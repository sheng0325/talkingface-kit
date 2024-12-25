import os

def clean_non_mp4_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            if not file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    folder_to_clean = "/root/autodl-tmp/vox-png/train"  # 替换为你的目标文件夹路径
    clean_non_mp4_files(folder_to_clean)
