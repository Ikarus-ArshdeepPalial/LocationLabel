"""Step 1: does VGGT load and run on this machine, and how fast / how much GPU memory?"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "vggt_src"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
images = [str(p) for p in sys.argv[1:]]
print("images:", images)

started = time.perf_counter()
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
print(f"model loaded in {time.perf_counter() - started:.0f}s")

batch = load_and_preprocess_images(images).to(device)
print("input batch:", tuple(batch.shape))
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
    out = model(batch)
torch.cuda.synchronize()
print(f"inference {time.perf_counter() - started:.2f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
print({k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})

extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"], batch.shape[-2:])
print("intrinsics of view 0:\n", intrinsic[0, 0].cpu().numpy().round(1))
print("camera centres (relative scale):")
for i, e in enumerate(extrinsic[0].cpu()):
    rotation, translation = e[:, :3], e[:, 3]
    print(i, (-(rotation.T @ translation)).numpy().round(3))
