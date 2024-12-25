import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_video_duration(file_path):
    """获取视频时长"""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", file_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        return file_path, duration
    except Exception as e:
        print(f"Error checking duration for {file_path}: {e}")
        return file_path, None

def delete_invalid_videos(directory):
    """删除时长为 0 的视频并打印路径"""
    files_to_process = []

    # 获取所有 .mp4 文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                files_to_process.append(file_path)

    total_files = len(files_to_process)
    processed_files = 0

    # 使用 ProcessPoolExecutor 并行处理文件
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(get_video_duration, file_path): file_path for file_path in files_to_process}

        for future in as_completed(futures):
            file_path, duration = future.result()
            processed_files += 1
            if duration is not None and duration == 0:
                print(f"Deleting: {file_path}")
                os.remove(file_path)

            # 输出已处理的文件数和剩余的文件数
            print(f"Processed {processed_files}/{total_files} files, {total_files - processed_files} remaining.")

if __name__ == "__main__":
    directory = "/root/autodl-tmp/vox-png/train/"  # 替换为你的目录路径
    delete_invalid_videos(directory)
