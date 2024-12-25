import subprocess
import time

def monitor_gpu(interval=1):
    try:
        while True:
            # 执行 nvidia-smi 命令
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)

            # 打印结果
            print(result.stdout)

            # 等待指定时间间隔
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")

if __name__ == "__main__":
    monitor_gpu(interval=2)
