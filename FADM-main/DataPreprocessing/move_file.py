import os
import shutil

# 源文件夹路径（修改这里）
source_folder = '/root/autodl-tmp/data/'  # 修改为你的源文件夹路径

# 目标文件夹路径（修改这里）
destination_folder = '/root/autodl-tmp/vox-png/train/'  # 修改为你的目标文件夹路径

# 创建目标文件夹（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 初始化计数器
counter = 1

# 遍历源文件夹及其子文件夹
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # 获取文件的扩展名
        file_extension = file.split('.')[-1] if '.' in file else ''
        
        # 创建新的文件名（0000001、0000002...）
        new_name = f"{counter:07d}.{file_extension}"

        # 获取文件的完整路径
        old_file_path = os.path.join(root, file)
        new_file_path = os.path.join(destination_folder, new_name)

        # 移动并重命名文件
        shutil.move(old_file_path, new_file_path)

        # 打印文件的移动信息（可选）
        print(f"Moved: {old_file_path} -> {new_file_path}")

        # 增加计数器
        counter += 1

print("所有文件已成功提取并重命名！")
