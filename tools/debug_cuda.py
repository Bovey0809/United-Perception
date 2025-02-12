import torch
import torch.multiprocessing as mp
import os

def check_cuda(rank):
    try:
        print(f"Process {rank} - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        print(f"Process {rank} - Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Process {rank} - torch.cuda.is_available(): {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(rank)
            print(f"Process {rank} - Current device: {torch.cuda.current_device()}")
            print(f"Process {rank} - Device name: {torch.cuda.get_device_name(rank)}")
    except Exception as e:
        print(f"Process {rank} - Error: {str(e)}")

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"Main process sees {n_gpus} GPUs")
    
    mp.spawn(check_cuda, nprocs=n_gpus, join=True)
