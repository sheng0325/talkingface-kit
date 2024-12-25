import os
import cv2
import numpy as np
import torch
from scipy.io import loadmat  # 用于加载 .mat 文件
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_fid import fid_score  # 使用 pytorch-fid 来计算FID
from PIL import Image
from skimage.transform import resize
import scipy
import scipy.ndimage
import scipy.special
import math
import scipy.linalg
import glob
import librosa  # 音频处理
import dlib  # 人脸关键点检测

# --------------------- New NIQE Implementation ---------------------

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
    # flatten imdata
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.average(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if len(right_data) > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf

    imdata2_mean = np.mean(imdata2)
    r_hat = (np.mean(np.abs(imdata)) ** 2) / imdata2_mean if imdata2_mean != 0 else np.inf
    rhat_norm = r_hat * (((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2))

    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    N = (br - bl) * (gam2 / gam1)
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)

def gen_gauss_window(lw, sigma):
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum_weights = 1.0
    sd_squared = sd * sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd_squared)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum_weights += 2.0 * tmp
    weights = [w / sum_weights for w in weights]
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = image.astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return ( (image - mu_image) / (var_image + C), var_image, mu_image )

def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([
        alpha_m, (bl + br) / 2.0,
        alpha1, N1, bl1, br1,  # (V)
        alpha2, N2, bl2, br2,  # (H)
        alpha3, N3, bl3, br3,  # (D1)
        alpha4, N4, bl4, br4,  # (D2)
    ])

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    patch_features = [ _niqe_extract_subband_feats(p) for p in patches ]
    return np.array(patch_features)

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        raise ValueError("Input image is too small")
    
    # Ensure that the patch divides evenly into img
    hoffset = h % patch_size
    woffset = w % patch_size
    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]
    
    img = img.astype(np.float32)
    img2 = resize(img, (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5)), mode='constant', anti_aliasing=True)
    
    mscn1, _, _ = compute_image_mscn_transform(img)
    mscn2, _, _ = compute_image_mscn_transform(img2)
    
    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size // 2)  # Use integer division
    
    if feats_lvl1.size == 0 or feats_lvl2.size == 0:
        raise ValueError("No features extracted from patches.")
    
    feats = np.hstack((feats_lvl1, feats_lvl2))
    return feats

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def niqe(inputImgData, model_path):
    patch_size = 96
    # Load NIQE parameters
    params = loadmat(model_path)
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]
    
    M, N = inputImgData.shape
    if M <= (patch_size * 2 + 1) or N <= (patch_size * 2 + 1):
        raise ValueError(f"NIQE requires > {patch_size * 2 + 1}x{patch_size * 2 + 1} resolution images, got {M}x{N}")
    
    feats = get_patches_test_features(inputImgData, patch_size)
    if feats.size == 0:
        raise ValueError("No features extracted for NIQE.")
    
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    
    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
    
    return niqe_score

# --------------------- End of NIQE Implementation ---------------------

def extract_audio(video_path, audio_output_folder, video_name):
    """从视频中提取音频"""
    if not os.path.exists(audio_output_folder):
        os.makedirs(audio_output_folder)

    audio_path = os.path.join(audio_output_folder, f"{video_name}_audio.wav")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    
    # 使用 ffmpeg 命令提取音频
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    os.system(command)

    return audio_path, fps

def detect_landmarks(image, predictor):
    """检测人脸关键点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    else:
        return None

def extract_mouth_roi(image, landmarks):
    """提取嘴部区域"""
    # 嘴部关键点索引范围通常是 48 到 67
    mouth_points = landmarks[48:68]
    
    # 计算嘴部区域的边界框
    min_x = int(min(mouth_points[:, 0]))
    max_x = int(max(mouth_points[:, 0]))
    min_y = int(min(mouth_points[:, 1]))
    max_y = int(max(mouth_points[:, 1]))
    
    # 为嘴部区域增加一些 padding
    padding = int((max_x - min_x) * 0.2)  # 可根据需要调整 padding 的大小
    min_x = max(0, min_x - padding)
    max_x = min(image.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(image.shape[0], max_y + padding)
    
    # 提取嘴部区域
    mouth_roi = image[min_y:max_y, min_x:max_x]
    
    return mouth_roi

def compute_lip_sync_score(audio_path, video_path, predictor):
    """计算唇动同步性评分"""
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return float('nan')

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lip_sync_scores = []
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_landmarks(frame, predictor)
        if landmarks is not None:
            mouth_roi = extract_mouth_roi(frame, landmarks)
            
            if mouth_roi is None or mouth_roi.size == 0:
                print(f"Warning: Mouth ROI is empty for frame {frame_num} in video {video_path}")
                continue

            # 计算当前帧对应的音频特征的起始和结束时间（以秒为单位）
            start_time = frame_num / frame_rate
            end_time = (frame_num + 1) / frame_rate

            # 找到与当前视频帧对应的 MFCC 特征的帧索引
            start_mfcc_frame = int(start_time * sr / librosa.time_to_samples(1, sr=sr))  # 假设 hop_length=512
            end_mfcc_frame = int(end_time * sr / librosa.time_to_samples(1, sr=sr))

            # 提取与当前视频帧对应的 MFCC 特征
            frame_mfccs = mfccs[:, start_mfcc_frame:end_mfcc_frame + 1]

            # 计算嘴部区域的特征（例如，可以使用嘴部区域的平均颜色或更复杂的特征）
            mouth_feature = np.mean(mouth_roi)  # 示例：使用平均像素值作为特征

            # 计算 MFCC 特征与嘴部特征之间的相关性
            if frame_mfccs.shape[1] > 0:  # 检查 frame_mfccs 是否非空
                correlation = np.corrcoef(mouth_feature, np.mean(frame_mfccs, axis=1))[0, 1]
                lip_sync_scores.append(correlation)
            else:
                print(f"Warning: Empty MFCCs for frame {frame_num} in video {video_path}")

    cap.release()
    avg_lip_sync_score = np.nanmean(lip_sync_scores) if lip_sync_scores else float('nan')
    return avg_lip_sync_score

def compute_audio_visual_latency(audio_path, video_path, predictor):
    """计算音画一致性延迟"""
    y, sr = librosa.load(audio_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return float('nan')

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    latencies = []
    for onset_frame in onset_frames:
        # 音频事件对应的时间 (秒)
        audio_time = librosa.frames_to_time(onset_frame, sr=sr)
        
        # 找到最近的视频帧
        video_frame_num = int(audio_time * frame_rate)

        if 0 <= video_frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = detect_landmarks(frame, predictor)
            if landmarks is not None:
                # 假设的视觉事件：嘴部张开程度
                mouth_height = landmarks[66, 1] - landmarks[62, 1]  # 假设 62 和 66 是上下嘴唇中心的点
                
                # 简单的事件检测：嘴部高度变化超过阈值
                # 这里需要根据实际情况调整阈值和逻辑
                if video_frame_num > 0:  # 避免第一帧没有前一帧的情况
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num - 1)
                    ret, prev_frame = cap.read()
                    if ret:
                        prev_landmarks = detect_landmarks(prev_frame, predictor)
                        if prev_landmarks is not None:
                            prev_mouth_height = prev_landmarks[66, 1] - prev_landmarks[62, 1]
                            mouth_height_change = abs(mouth_height - prev_mouth_height)

                            # 设置一个阈值来判断是否为有效变化
                            threshold = 5  # 这个值需要根据实际情况调整
                            if mouth_height_change > threshold:
                                latencies.append(0)  # 如果检测到变化，延迟认为是0，表示同步
                            else:
                                latencies.append(1)  # 否则认为不同步，延迟设为1，或其他值
        
    cap.release()

    avg_latency = np.nanmean(latencies) if latencies else float('nan')
    return avg_latency

def compute_expression_richness(video_path, predictor):
    """计算表情丰富度"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return float('nan')

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 假设的表情数量 (需要根据实际使用的表情识别模型调整)
    num_expressions = 7  

    expression_counts = np.zeros(num_expressions)
    
    # 预先分配一个数组来存储每一帧的表情标签
    frame_expressions = np.full(total_frames, -1)  # 初始化为-1或其他无效标签

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_landmarks(frame, predictor)
        if landmarks is not None:
            # 使用 landmarks 计算表情特征 (这里只是一个占位符, 需要根据实际情况实现)
            # 例如，可以计算不同关键点之间的距离比例等
            
            # 假设的表情特征：嘴巴张开程度和眉毛上扬程度
            mouth_openness = landmarks[66, 1] - landmarks[62, 1]  # 62 和 66 分别是上下嘴唇中心的点
            eyebrow_raise = landmarks[21, 1] - landmarks[27, 1]  # 21 和 27 分别是左眉毛和鼻子中心的点

            # 根据特征判断表情 (这里只是一个简单的示例, 需要根据实际情况实现)
            # 可以使用机器学习模型来根据 landmarks 预测表情
            if mouth_openness > 10 and eyebrow_raise < -5:
                expression_label = 0  # 假设 0 代表 "高兴"
            elif mouth_openness > 10 and eyebrow_raise > 5:
                expression_label = 1  # 假设 1 代表 "惊讶"
            elif mouth_openness < 5 and eyebrow_raise > 5:
                expression_label = 2  # 假设 2 代表 "悲伤"
            # ... 其他表情的判断 ...
            else:
                expression_label = 6  # 假设 6 代表 "中性"

            expression_counts[expression_label] += 1
            
            # 记录当前帧的表情标签
            frame_expressions[frame_num] = expression_label

    cap.release()

    # 计算表情丰富度 (这里只是一个简单的示例, 需要根据实际情况实现)
    # 可以使用表情数量、表情转换次数、表情持续时间等指标
    
    # 计算表情转换次数
    expression_changes = 0
    for i in range(1, total_frames):
        if frame_expressions[i] != frame_expressions[i - 1] and frame_expressions[i] != -1 and frame_expressions[i - 1] != -1:
            expression_changes += 1

    # 计算表情丰富度评分
    richness_score = expression_changes / total_frames if total_frames > 0 else 0
    
    # 也可以考虑不同表情的持续时间等因素
    # ...

    return richness_score

def compute_motion_naturalness(video_path, predictor):
    """计算动作自然度"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return float('nan')

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_landmarks = []
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detect_landmarks(frame, predictor)
        if landmarks is not None:
            all_landmarks.append(landmarks)

    cap.release()

    if not all_landmarks:
        print(f"Warning: No landmarks detected in video {video_path}")
        return float('nan')

    # 计算关键点速度和加速度
    velocities = []
    accelerations = []
    for i in range(1, len(all_landmarks)):
        # 计算速度
        velocity = (all_landmarks[i] - all_landmarks[i - 1]) * frame_rate
        velocities.append(velocity)

        if i > 1:
            # 计算加速度
            acceleration = (velocities[i - 1] - velocities[i - 2]) * frame_rate
            accelerations.append(acceleration)

    if not velocities or not accelerations:
        print(f"Warning: Not enough landmarks to compute velocities or accelerations in video {video_path}")
        return float('nan')

    # 计算平均速度和加速度的统计量
    avg_velocity_magnitude = np.mean(np.linalg.norm(velocities, axis=2)) if velocities else 0.0
    avg_acceleration_magnitude = np.mean(np.linalg.norm(accelerations, axis=2)) if accelerations else 0.0

    # 计算速度和加速度的标准差
    std_velocity_magnitude = np.std(np.linalg.norm(velocities, axis=2)) if velocities else 0.0
    std_acceleration_magnitude = np.std(np.linalg.norm(accelerations, axis=2)) if accelerations else 0.0

    # 计算自然度评分（这里只是一个示例，需要根据实际情况调整）
    # 可以使用速度和加速度的统计量、频谱分析等
    naturalness_score = 1 / (1 + std_velocity_magnitude + std_acceleration_magnitude)
    
    # 也可以考虑其他因素，例如运动的平滑度、抖动程度等
    # ...

    return naturalness_score

def compute_psnr_metric(img1, img2):
    """计算PSNR指标"""
    return psnr(img1, img2, data_range=img2.max() - img2.min())

def compute_ssim_metric(img1, img2):
    """计算SSIM指标"""
    return ssim(img1, img2, channel_axis=2, win_size=3)

def compute_fid(path1, path2):
    """计算FID指标"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return fid_score.calculate_fid_given_paths(
        [path1, path2],
        batch_size=50,
        device=device,
        dims=2048
    )

def compute_lse_c(img1, img2):
    """计算LSE-C指标（简单的L2距离作为占位符）"""
    return np.linalg.norm(img1.astype(np.float32) - img2.astype(np.float32))

def compute_lse_d(img1, img2):
    """计算LSE-D指标（简单的L2距离作为占位符）"""
    return np.linalg.norm(img1.astype(np.float32) - img2.astype(np.float32))

def extract_frames(video_path, output_folder, fixed_size=(256, 256)):
    """从视频中提取帧并保存到指定文件夹，同时调整帧尺寸"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    pbar = tqdm(desc=f"Extracting frames from {os.path.basename(video_path)}", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to fixed size
        frame_resized = cv2.resize(frame, fixed_size, interpolation=cv2.INTER_LINEAR)
        frame_name = f"frame_{frame_count:05d}.jpg"
        success = cv2.imwrite(os.path.join(output_folder, frame_name), frame_resized)
        if not success:
            print(f"Warning: Failed to write frame {frame_name}")
        frame_count += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")

def resize_to_fixed_size(img, size=(256, 256)):
    """调整图像到固定尺寸"""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

# 初始化 dlib 的人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/root/code/123/FADM-main/video-quality/shape_predictor_68_face_landmarks.dat")  # 请替换为你的关键点预测器模型路径

def main(main_video_path, ref_videos_folder, frames_folder, ref_frames_parent_folder, plot_folder, niqe_model_path, audio_output_folder):
    # 步骤1：提取主视频帧
    extract_frames(main_video_path, frames_folder)
    
    main_video_name = os.path.splitext(os.path.basename(main_video_path))[0]
    main_audio_path, main_fps = extract_audio(main_video_path, audio_output_folder, main_video_name)
    if main_audio_path is None or main_fps is None:
        print("Error: Failed to extract audio from the main video. Exiting.")
        return
    
    # 步骤2：获取所有参考视频路径
    ref_video_paths = sorted(glob.glob(os.path.join(ref_videos_folder, '*.*')))
    ref_video_paths = [f for f in ref_video_paths if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not ref_video_paths:
        print(f"No reference videos found in {ref_videos_folder}")
        return
    
    # 步骤3：提取所有参考视频帧
    for ref_video_path in ref_video_paths:
        ref_video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
        current_ref_frames_folder = os.path.join(ref_frames_parent_folder, ref_video_name)
        extract_frames(ref_video_path, current_ref_frames_folder)
        
    for ref_video_path in ref_video_paths:
        ref_video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
        # 提取参考视频的音频
        ref_audio_path = os.path.join(audio_output_folder, f"{ref_video_name}_audio.wav") # Correct filename
        ref_audio_path, ref_fps = extract_audio(ref_video_path, audio_output_folder, ref_video_name)  # Corrected call
        
        # Check if audio extraction was successful for the reference video
        if ref_audio_path is None or ref_fps is None:
            print(f"Warning: Failed to extract audio for reference video {ref_video_name}. Skipping metrics that require audio.")
            # Don't `continue` here, as we still want to compute frame-based metrics.
            # We will handle missing audio later.
        else:
            print(f"Successfully extracted audio for {ref_video_name} to {ref_audio_path}")
    
    # 获取主帧文件列表并过滤无效文件（仅保留 .jpg 和 .png 文件）
    main_frames = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))])
    
    # 打印主帧数量
    print(f"Number of frames in {frames_folder}: {len(main_frames)}")
    
    # 计算NIQE for main frames
    print("Calculating NIQE for main video frames...")
    niqe_scores = []
    for frame_name in tqdm(main_frames, desc="Calculating NIQE"):
        img1_path = os.path.join(frames_folder, frame_name)
        img1 = cv2.imread(img1_path)
        if img1 is None:
            print(f"Warning: Unable to read {img1_path}")
            continue
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        try:
            current_niqe = niqe(img1_gray, niqe_model_path)
            niqe_scores.append(current_niqe)
        except Exception as e:
            print(f"Error computing NIQE for frame {frame_name}: {e}")
            continue
    niqe_mean = np.mean(niqe_scores) if niqe_scores else float('nan')
    print(f"Average NIQE for main video: {niqe_mean}")
    
    # Initialize a dictionary to store metrics per reference video
    metrics_dict = {}
    
    # Iterate over each reference video
    for ref_video_path in ref_video_paths:
        ref_video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
        current_ref_frames_folder = os.path.join(ref_frames_parent_folder, ref_video_name)
        
        # 提取参考视频的音频
        ref_audio_path = os.path.join(audio_output_folder, f"{ref_video_name}_audio.wav") # Correctly use audio_output_folder
        ref_audio_path, ref_fps = extract_audio(ref_video_path, audio_output_folder, ref_video_name)  # Corrected line
        
        # 获取参考帧列表
        ref_frames = sorted([f for f in os.listdir(current_ref_frames_folder) if f.lower().endswith(('.jpg', '.png'))])
        print(f"\nProcessing reference video: {ref_video_name}")
        print(f"Number of frames in {current_ref_frames_folder}: {len(ref_frames)}")
        
        # 确保两段视频帧数相同
        num_frames = min(len(main_frames), len(ref_frames))
        if len(main_frames) != len(ref_frames):
            print(f"Warning: Number of frames differ between main video and {ref_video_name}. Processing {num_frames} frames.")
        
        # Initialize metrics lists
        psnr_list = []
        ssim_list = []
        lse_c_list = []
        lse_d_list = []
        
        # 计算新指标
        lip_sync_score = compute_lip_sync_score(main_audio_path, main_video_path, predictor)
        av_latency = compute_audio_visual_latency(main_audio_path, main_video_path, predictor)
        expression_richness = compute_expression_richness(main_video_path, predictor)
        motion_naturalness = compute_motion_naturalness(main_video_path, predictor)
        
        # Process frames
        for i in tqdm(range(num_frames), desc=f"Computing metrics for {ref_video_name}"):
            # Get frame names
            frame_name = main_frames[i]
            ref_frame_name = ref_frames[i]
            
            # Read frames
            img1_path = os.path.join(frames_folder, frame_name)
            img2_path = os.path.join(current_ref_frames_folder, ref_frame_name)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            # Check if images are loaded
            if img1 is None:
                print(f"Warning: Unable to read {img1_path}")
                psnr_val = float('nan')
                ssim_val = float('nan')
                lse_c_val = float('nan')
                lse_d_val = float('nan')
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lse_c_list.append(lse_c_val)
                lse_d_list.append(lse_d_val)
                continue
            if img2 is None:
                print(f"Warning: Unable to read {img2_path}")
                psnr_val = float('nan')
                ssim_val = float('nan')
                lse_c_val = float('nan')
                lse_d_val = float('nan')
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lse_c_list.append(lse_c_val)
                lse_d_list.append(lse_d_val)
                continue
            
            # Resize images to fixed size
            img1 = resize_to_fixed_size(img1, size=(256, 256))
            img2 = resize_to_fixed_size(img2, size=(256, 256))
            
            # Convert BGR to RGB
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Debug: Print image shapes and data types
            if img1_rgb.shape != img2_rgb.shape:
                print(f"Error: Image shapes do not match for frame {i}: {img1_rgb.shape} vs {img2_rgb.shape}")
                psnr_val = float('nan')
                ssim_val = float('nan')
            else:
                # Compute PSNR and SSIM
                try:
                    psnr_val = compute_psnr_metric(img1_rgb, img2_rgb)
                    ssim_val = compute_ssim_metric(img1_rgb, img2_rgb)
                except Exception as e:
                    print(f"Error computing PSNR/SSIM for frame {i}: {e}")
                    psnr_val = float('nan')
                    ssim_val = float('nan')
            
            # Compute LSE-C and LSE-D
            try:
                lse_c_val = compute_lse_c(img1, img2)
                lse_d_val = compute_lse_d(img1, img2)
            except Exception as e:
                print(f"Error computing LSE-C/LSE-D for frame {i}: {e}")
                lse_c_val = float('nan')
                lse_d_val = float('nan')
            
            # Add to lists
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lse_c_list.append(lse_c_val)
            lse_d_list.append(lse_d_val)
        
        # Calculate average metrics
        avg_psnr = np.mean(psnr_list) if psnr_list else float('nan')
        avg_ssim = np.mean(ssim_list) if ssim_list else float('nan')
        avg_lse_c = np.mean(lse_c_list) if lse_c_list else float('nan')
        avg_lse_d = np.mean(lse_d_list) if lse_d_list else float('nan')
        
        # Compute FID
        try:
            fid_score_value = compute_fid(frames_folder, current_ref_frames_folder)
        except Exception as e:
            print(f"Error computing FID for {ref_video_name}: {e}")
            fid_score_value = float('nan')
        
        # Store metrics
        metrics_dict[ref_video_name] = {
            'PSNR': avg_psnr,
            'SSIM': avg_ssim,
            'NIQE': niqe_mean,
            'FID': fid_score_value,
            'LSE-C': avg_lse_c,
            'LSE-D': avg_lse_d,
            'Lip-sync Score': lip_sync_score,
            'Audio-Visual Latency': av_latency,
            'Expression Richness': expression_richness,
        }
    
    # Print all metrics
    print("\n=== Metrics Summary ===")
    print("{:<25} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>15} {:>25} {:>20} {:>20}".format(
        "Reference Video", "PSNR", "SSIM", "NIQE", "FID", "LSE-C", "LSE-D", "Lip-sync Score", "Audio-Visual Latency", "Expression Richness", "Motion Naturalness"))
    for ref_video, metrics in metrics_dict.items():
        print("{:<25} {:>8.2f} {:>8.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>15.4f} {:>25.4f} {:>20.4f} {:>20.4f}".format(
            ref_video,
            metrics['PSNR'],
            metrics['SSIM'],
            metrics['NIQE'],
            metrics['FID'],
            metrics['LSE-C'],
            metrics['LSE-D'],
            metrics['Lip-sync Score'],
            metrics['Audio-Visual Latency'],
            metrics['Expression Richness'],
            metrics['Motion Naturalness']
        ))
    
    # Optionally, save metrics to a file
    metrics_output_path = os.path.join(plot_folder, "metrics_summary.txt")
    with open(metrics_output_path, 'w') as f:
        f.write("{:<25} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>15} {:>25} {:>20} {:>20}\n".format(
            "Reference Video", "PSNR", "SSIM", "NIQE", "FID", "LSE-C", "LSE-D", "Lip-sync Score", "Audio-Visual Latency", "Expression Richness", "Motion Naturalness"))
        for ref_video, metrics in metrics_dict.items():
            f.write("{:<25} {:>8.2f} {:>8.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>15.4f} {:>25.4f} {:>20.4f} {:>20.4f}\n".format(
                ref_video,
                metrics['PSNR'],
                metrics['SSIM'],
                metrics['NIQE'],
                metrics['FID'],
                metrics['LSE-C'],
                metrics['LSE-D'],
                metrics['Lip-sync Score'],
                metrics['Audio-Visual Latency'],
                metrics['Expression Richness'],
                metrics['Motion Naturalness']
            ))
    print(f"Metrics summary saved to {metrics_output_path}")

    # Optionally, plot metrics
    # Skipping plotting in this multi-reference scenario
    # Alternatively, plot per metric with multiple lines
    # Implement if required

# 设置视频路径、参考视频文件夹、帧保存路径、模型路径和图表保存路径
main_video_path = "/root/code/123/FADM-main/video-quality/video/output.mp4"
ref_videos_folder = "/root/code/123/FADM-main/video-quality/reference/"  # Folder containing multiple reference videos
frames_folder = "/root/code/123/FADM-main/video-quality/frame/"
ref_frames_parent_folder = "/root/code/123/FADM-main/video-quality/frame_reference/"
plot_folder = "/root/code/123/FADM-main/video-quality/output/"  # For saving plots and metrics
niqe_model_path = "/root/code/123/FADM-main/video-quality/niqe_image_params.mat"  # Ensure this is 'niqe_image_params.mat'
audio_output_folder = "/root/code/123/FADM-main/video-quality/audio_output/" # 设置音频输出的文件夹

if __name__ == "__main__":
    main(main_video_path, ref_videos_folder, frames_folder, ref_frames_parent_folder, plot_folder, niqe_model_path, audio_output_folder)
