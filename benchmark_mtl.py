import os
import time
import torch
import numpy as np
import PIL.Image as pil
from torchvision import transforms
import torch.nn.functional as F
import networks
from layers import disp_to_depth
import cv2

def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Benchmarking on: {device}")

    # 1. Load Models
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    encoder = networks.LiteMono(model=args.model, height=192, width=640, in_chans=3).to(device).eval()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3)).to(device).eval()
    depth_decoder.load_state_dict(torch.load(os.path.join(args.load_weights_folder, "depth.pth"), map_location=device))

    seg_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=9, is_seg=True).to(device).eval()
    seg_decoder.load_state_dict(torch.load(os.path.join(args.load_weights_folder, "segmentation.pth"), map_location=device))
    
    centers_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=1, is_seg=False).to(device).eval()
    centers_decoder.load_state_dict(torch.load(os.path.join(args.load_weights_folder, "centers.pth"), map_location=device))

    # 2. Prepare Dummy Data (To avoid Disk I/O bottleneck)
    # We use random data of the correct input size
    input_tensor = torch.randn(1, 3, 192, 640).to(device)

    # 3. WARM-UP (Crucial for GPU)
    print("-> Warming up GPU...")
    for _ in range(50):
        with torch.no_grad():
            features = encoder(input_tensor)
            _ = depth_decoder(features)
            _ = seg_decoder(features)
            _ = centers_decoder(features)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 4. ACTUAL TEST (Pure Inference)
    num_frames = 500
    print(f"-> Running Inference Benchmark ({num_frames} frames)...")
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_frames):
            features = encoder(input_tensor)
            d = depth_decoder(features)
            s = seg_decoder(features)
            c = centers_decoder(features)
            
            # If on GPU, we must force synchronization to get real time
            if device.type == 'cuda':
                torch.cuda.synchronize()

    end_time = time.perf_counter()
    
    # 5. CALCULATE STATS
    total_time = end_time - start_time
    avg_latency = (total_time / num_frames) * 1000 # in milliseconds
    fps = num_frames / total_time

    print("\n" + "="*30)
    print(f"RESULTS FOR {args.model.upper()}")
    print("="*30)
    print(f"Total Time:     {total_time:.2f} seconds")
    print(f"Avg Latency:    {avg_latency:.2f} ms per frame")
    print(f"Throughput:     {fps:.2f} FPS")
    print("="*30)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--model', type=str, default="lite-mono-8m")
    args = parser.parse_args()
    benchmark(args)