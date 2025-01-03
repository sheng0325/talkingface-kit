from tkinter import N
import matplotlib
from matplotlib import image

# 设置matplotlib 使用无图形界面后端
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm, trange

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.diffusion.diffusion_generator import DiffusionGenerator
from modules.Facevid.keypoint_detector import KPDetector, HEEstimator
from modules.Facevid.generator_facevid import OcclusionAwareGenerator
from modules.Facevid.keypoint_transform import keypoint_transformation
from modules.deca.decalib.deca import DECA
# from arcface import ArcFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from einops import rearrange, reduce, repeat
# from pytorch_fid import fid_score, inception
import torch.nn.functional as F
import pdb

# 限制版本
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

# 初始化扩散模型
def set_diffusion(diffusion_path, img_size=256, time_steps=100):

    # 创建扩散模型实例
    diffusion_model = DiffusionGenerator(img_size=img_size, timesteps=time_steps)
    
    checkpoint = torch.load(diffusion_path, map_location='cpu') # 加载权重
    diffusion_model.load_state_dict(checkpoint) # 将权重应用到扩散模型
        
    print('load diffusion model successfull')
    diffusion_model = diffusion_model.cuda()
    diffusion_model.eval()
    return diffusion_model

# 生成结果图像
def get_result(s, d, deca, diffusion, test_noise, noise_list):

    # 使用 DECA 提取源图像和驱动图像的表情和姿态特征
    _, attr_s, attr_d = deca(source_pic=s, driving_pic=d)
    # exp：表情 pose：姿势
    attr = torch.cat((attr_s['exp'], attr_s['pose']), dim=1) - torch.cat((attr_d['exp'], attr_d['pose']), dim=1)

    # 构建不同分辨率的驱动特征
    low_feat = [F.interpolate(d, size=(32, 32)), 
                F.interpolate(d, size=(64, 64)), 
                F.interpolate(d, size=(128, 128))]

    # 生成32x32的初始特征图
    g = F.interpolate(d, size=(32, 32))
    g_dict = {'generated': F.interpolate(g, size=(256, 256)), 'driving': F.interpolate(g, size=(256, 256))}

    # 生成特征、驱动图像特征、表情和姿态差异、源图像
    condition_img = {'generated': g_dict, 'warping': low_feat, 'attribute': attr,
                     'source': s}
    # 用扩散模型生成图像
    img = diffusion.refer(test_noise, condition=condition_img, noise_list=noise_list)
    # 提取的结果是从第11个张量
    result_img = img[10:11, ...]
    # 限制图限制
    result_img = result_img.clamp(0, 1)

    return result_img


def set_noise():
    test_noise = torch.randn(1, 3, 256, 256).cuda()
    noise_list = []
    # 生成150个随机张量
    for i in range(150):
        noise_list.append(torch.randn(1, 3, 256, 256).cuda())

    return test_noise, noise_list

# 驱动视频生成动画
def make_animation_new(driving_video, diffusion_model, deca):
    with torch.no_grad():
        predictions = []
        # 初始化噪声张量
        test_noise, noise_list = set_noise()
        # 将驱动视频转换为pytorch张量，调整维度
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        s = driving[:, :, 0].cuda()
        # 遍历视频的每一帧
        for frame_idx in tqdm(range(driving.shape[2])):
            # 提取当前帧
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.cuda()
            # 调用get_result生辰当前帧
            out = get_result(s, driving_frame, deca, diffusion_model, test_noise, noise_list)
            predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])
            
    return predictions


def set_face_vid(load_path, config):
    # 加载权重
    checkpoint = torch.load(load_path, map_location='cpu')

    # 关键点检测器 检测器参数:kp_detertor_params 通用参数:common_params
    keypoint_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                   **config['model_params']['common_params'])
    keypoint_detector.load_state_dict(checkpoint['kp_detector'])
    keypoint_detector.cuda()

    # 头部表情估计器 
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    he_estimator.load_state_dict(checkpoint['he_estimator'])
    he_estimator.cuda()

    # 遮盖感知生成器
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.load_state_dict(checkpoint['generator'])
    generator.cuda()

    # 设置所有模型为评估模式
    keypoint_detector.eval()
    he_estimator.eval()
    generator.eval()

    return keypoint_detector, he_estimator, generator

# 根据源图像和驱动视频生成动画
def make_animation(source_image, driving_video, generator, kp_detector, he_estimator, diffusion_model, deca, relative=True,
                   adapt_movement_scale=True, estimate_jacobian=False, cpu=False, free_view=False, yaw=0, pitch=0,
                   roll=0):
    with torch.no_grad():
        predictions = []
        test_noise, noise_list = set_noise()
        # 输入形状为 (H, W, C)，输出形状为 (1, C, H, W)
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        # 将驱动视频转换为 PyTorch 张量，并调整维度顺序
        # 输入形状为 (T, H, W, C)，输出形状为 (1, C, T, H, W)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        # 提取源图像的关键点特征
        kp_canonical = kp_detector(source)
        # 使用头部表情估计器提取源图像的头部表情特征
        he_source = he_estimator(source)
        # 提取驱动视频第一帧的头部表情特征
        he_driving_initial = he_estimator(driving[:, :, 0].cuda())

        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, estimate_jacobian)
        # kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)

        # 遍历每一帧
        for frame_idx in tqdm(range(driving.shape[2])):
            # 提取当前帧
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            he_driving = he_estimator(driving_frame)
            kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian)
            kp_norm = kp_driving
            out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
            result = get_result(s=source, d=out['prediction'], deca=deca, diffusion=diffusion_model, test_noise=test_noise,
                                noise_list=noise_list)
            predictions.append(np.transpose(result.data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--inpainting_video_path", default='', help="path to coarse video")
    parser.add_argument("--source_image_path", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video_path", default='sup-mat/driving.mp4', help="path to driving video")
    parser.add_argument("--out_video_path", default='sup-mat/', help="path to output")
    parser.add_argument("--pattern", default='full', help="generation pattern including 'direct' and 'full' ")
    parser.add_argument("--log_dir", default='sup-mat/log', help="path to log")

    opt = parser.parse_args()

    if not os.path.exists(opt.out_video_path):
        os.makedirs(opt.out_video_path)
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    # set models
    config_path = 'config/vox-256.yaml' # 配置路径文件
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    load_path = 'modules/Facevid/ckpt/Facevid.pth'
    keypoint_detector, he_estimator, generator = set_face_vid(load_path, config=config)
    
    diffusion_path = 'modules/diffusion/ckpt/diffusion.pth'
    diffusion_model = set_diffusion(diffusion_path)

    deca = DECA()
    deca.eval()

    # generation
    if opt.pattern == 'full':
        source_image = imageio.imread(opt.source_image_path)
        reader = imageio.get_reader(opt.driving_video_path)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        out_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        predictions = make_animation(source_image=source_image, driving_video=driving_video, generator=generator, kp_detector=keypoint_detector,
                                        he_estimator=he_estimator, diffusion_model=diffusion_model, deca=deca)
        for frame_idx, frame in enumerate(predictions):

            frame_int = img_as_ubyte(frame)
            driving_int = img_as_ubyte(driving_video[frame_idx])

            out_video.append(frame_int)

        imageio.mimsave(os.path.join(opt.out_video_path, f'output.mp4'), out_video, fps=fps)

    elif opt.pattern == 'direct':
        reader = imageio.get_reader(opt.inpainting_video_path)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        out_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        predictions = make_animation_new(driving_video=driving_video, diffusion_model=diffusion_model, deca=deca)
        for frame_idx, frame in enumerate(predictions):

            frame_int = img_as_ubyte(frame)
            driving_int = img_as_ubyte(driving_video[frame_idx])

            out_video.append(frame_int)

        imageio.mimsave(os.path.join(opt.out_video_path, f'output.mp4'), out_video, fps=fps)
