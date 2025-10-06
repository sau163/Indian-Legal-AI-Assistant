
import argparse
import subprocess
import sys
from pathlib import Path

TRAIN_PY = Path(__file__).parent / "train.py"

parser = argparse.ArgumentParser(description="GPU-friendly launcher for train.py")
parser.add_argument("--gpu-id", type=int, default=0, help="GPU id to use (default 0)")
parser.add_argument("--no-quant", action="store_true", help="Disable quantization")
parser.add_argument("--no-flash", action="store_true", help="Disable Flash Attention")
parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Arguments passed through to train.py")

args = parser.parse_args()

cmd = [sys.executable, str(TRAIN_PY), "--device-map", f"cuda:{args.gpu_id}"]

if args.no_quant:
    cmd.append("--disable-quantization")
else:
    # enable quantization by not passing disable; if user wants flash disabled they can
    pass

if args.no_flash:
    cmd.extend(["--force-quantization"])  # no-op, but keeps options explicit

if args.extra_args:
    cmd.extend(args.extra_args)

print("Running:", " ".join(cmd))
subprocess.run(cmd)
