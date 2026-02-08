#!/usr/bin/env python3
"""
Quick test of the evaluation script on a small subset
"""

import subprocess
import sys

# Test with a small number of samples
cmd = [
    sys.executable,
    "evaluate_flow_matching.py",
    "--checkpoint", "runs/smoke_dataset_run-flow_smoke_vqgan/checkpoints/step_40000.pth",
    "--config", "configs/smoke_dataset.yaml",
    "--num-samples", "3",
    "--num-test-videos", "5",
    "--output-dir", "test_evaluation",
]

print("Running quick evaluation test...")
print(" ".join(cmd))
print()

result = subprocess.run(cmd)
sys.exit(result.returncode)
