import os
import glob
import numpy as np
import pandas as pd
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import imageio.v3 as iio  # 使用 imageio.v3 替代 mimread
from augmentation import AllAugmentationTransform  # 确保该模块存在

# 视频读取函数
def read_video(name, frame_shape):
    """
    读取视频的函数，可以处理以下几种情况：
      - 帧序列图片
      - '.mp4' 或 '.gif' 视频文件
      - 包含视频帧的文件夹
    """
    if os.path.isdir(name):
        # 如果是文件夹，逐帧读取
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)]
        )
    elif name.lower().endswith(('.png', '.jpg')):
        # 如果是图片，处理为视频帧
        image = io.imread(name)

        # 如果是灰度图像，转换为 RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        # 如果有透明通道，去掉
        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        # 将图片分解为帧
        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith(('.gif', '.mp4', '.mov')):
        # 使用 imageio.v3 的 pyav 插件读取视频
        try:
            video = iio.imread(name, plugin="pyav")
            video_array = img_as_float32(video)

            # 如果是灰度帧，转换为 RGB
            if len(video_array.shape) == 3:
                video_array = np.array([gray2rgb(frame) for frame in video_array])
            # 如果有透明通道，去掉
            if video_array.shape[-1] == 4:
                video_array = video_array[..., :3]

            # 调整帧尺寸到目标形状
            target_h, target_w, target_c = frame_shape
            resized_frames = []
            for frame in video_array:
                resized_frame = resize(
                    frame, 
                    (target_h, target_w), 
                    order=1, 
                    preserve_range=True, 
                    anti_aliasing=True
                )
                resized_frame = resized_frame.astype('float32')  # 确保类型一致
                resized_frames.append(resized_frame)
            
            video_array = np.array(resized_frames, dtype='float32')
        except Exception as e:
            raise IOError(f"读取视频失败：{name}, 错误信息：{e}") from e
    else:
        raise Exception(f"未知的文件扩展名：{name}")

    return video_array


# 视频帧数据集类
class FramesDataset(Dataset):
    """
    视频帧数据集类，每个视频可以是：
      - 拼接帧的图片
      - '.mp4' 或 '.gif' 视频
      - 包含帧的文件夹
    """

    def __init__(
        self, 
        root_dir, 
        frame_shape=(256, 256, 3), 
        id_sampling=False, 
        is_train=True,
        random_seed=0, 
        pairs_list=None, 
        augmentation_params=None
    ):
        self.root_dir = root_dir
        print(f"数据集根目录：{root_dir}")  # 调试信息
        self.videos = os.listdir(root_dir)
        print(f"发现的文件：{self.videos}")  # 调试信息
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test')), "测试目录不存在"
            print("使用预定义的训练-测试分割")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("使用随机训练-测试分割")
            train_videos, test_videos = train_test_split(
                self.videos, 
                random_state=random_seed, 
                test_size=0.2
            )

        self.videos = train_videos if is_train else test_videos
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params) if augmentation_params else None
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            print(f"根目录：{self.root_dir}, 名称：{name}")  # 调试信息
            matched_files = glob.glob(os.path.join(self.root_dir, name + '*.mp4'))
            print(matched_files)  # 调试信息
            if not matched_files:
                raise FileNotFoundError(f"未找到匹配的文件：{name}*.mp4")
            path = np.random.choice(matched_files)
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [
                img_as_float32(io.imread(os.path.join(path, frames[idx].decode('UTF8')))) 
                for idx in frame_idx
            ]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = (
                np.sort(np.random.choice(num_frames, replace=True, size=2)) 
                if self.is_train else 
                range(num_frames)
            )
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            print(f"源帧形状：{source.shape}, 驱动帧形状：{driving.shape}")  # 调试信息

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out



class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
