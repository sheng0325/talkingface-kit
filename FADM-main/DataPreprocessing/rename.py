import os

def rename_files_in_directory(directory):
    """按照顺序给文件重命名，格式为 0000001, 0000002 等，保留原扩展名"""
    files = os.listdir(directory)
    
    # 过滤出文件，只保留文件（排除文件夹）
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    
    # 按文件名排序，或者按修改时间等排序
    files.sort()  # 如果文件名是数字，可以根据需要自定义排序
    
    # 打印总文件数
    print(f"Total files to rename: {len(files)}")

    # 重命名文件
    for index, file in enumerate(files, start=1):
        old_file_path = os.path.join(directory, file)
        
        # 获取文件名和扩展名
        file_name, file_extension = os.path.splitext(file)
        
        # 格式化新文件名，保持扩展名
        new_file_name = f"{index:07d}{file_extension}"
        new_file_path = os.path.join(directory, new_file_name)
        
        # 打印当前重命名的文件
        print(f"Renaming: {file} -> {new_file_name}")

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        
    # 完成后打印
    print("Renaming process completed.")

if __name__ == "__main__":
    directory = "/root/autodl-tmp/vox-png/train/"  # 替换为你的文件夹路径
    rename_files_in_directory(directory)
