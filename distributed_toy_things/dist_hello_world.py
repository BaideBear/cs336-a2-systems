import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3, ))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__=="__main__":
    world_size = 4
    # 这里的rank参数由spawn函数自动传入
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)

"""
Running on gloo:
[Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 3 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
rank 3 data (before all-reduce): tensor([8, 2, 8])
rank 2 data (before all-reduce): tensor([4, 7, 0])
rank 0 data (before all-reduce): tensor([8, 7, 7])
rank 1 data (before all-reduce): tensor([2, 3, 6])
rank 0 data (after all-reduce): tensor([22, 19, 21])
rank 3 data (after all-reduce): tensor([22, 19, 21])
rank 2 data (after all-reduce): tensor([22, 19, 21])
rank 1 data (after all-reduce): tensor([22, 19, 21])
"""