import torch


# 
def get_dataset_distributed_(_dataset, world_size, rank, batch_size, **kwargs):

    # 创建分布式采样器 (DistributedSampler)
    # 这个采样器会将数据集划分给不同的进程，以确保每个进程处理的数据不重复且均匀分布。
    sampler = torch.utils.data.distributed.DistributedSampler(
        _dataset,             # 输入数据集
        num_replicas=world_size,  # 总进程数，即世界大小
        rank=rank,               # 当前进程的 rank，决定此进程处理哪一部分数据
    )
 
    # 使用采样器创建数据加载器 (DataLoader)
    dataloader = torch.utils.data.DataLoader(
        _dataset,            # 输入数据集
        sampler=sampler,     # 分布式采样器，决定每个 rank 加载的具体数据
        batch_size=batch_size,  # 每个 rank 中的 batch 大小
        shuffle=False,       # 分布式模式下，shuffle 必须由采样器控制，所以这里设置为 False
        drop_last=True,      # 如果最后一个 batch 的大小不足，则丢弃它，确保所有 batch 尺寸一致
        pin_memory=True,    # 是否将数据加载到固定内存，适用于加速 CPU 到 GPU 的数据传输（默认关闭）
        num_workers=16,      # 数据加载时使用的子进程数，可以根据硬件性能调整
        persistent_workers=True,  # 如果为 True，则 DataLoader 的工作进程在多次迭代间保持存活，减少创建开销
    )

    print(f"Rank {rank}: DataLoader initialized with batch_size={batch_size}, num_workers={kwargs.get('num_workers', 4)}")
    return dataloader, 3
