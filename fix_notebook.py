import json

file_path = "/run/media/santhankumar/New Volume/ContainerScaler-RL/kaggle-training-ppo.ipynb"

with open(file_path, "r") as f:
    notebook = json.load(f)

# Update cell 3 (index 2)
cell_source = """# 3. Verify PyTorch and GPU Availability
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device: {gpu_name}")
    
    if "P100" in gpu_name:
        print("\\n" + "="*80)
        print("🚨 CRITICAL ERROR: INCOMPATIBLE GPU DETECTED 🚨")
        print("The current Kaggle PyTorch environment does not support the older Tesla P100 GPU.")
        print("Please change your Kaggle Accelerator setting to T4x2:")
        print("  1. Go to the right sidebar in Kaggle.")
        print("  2. Click on 'Settings' -> 'Accelerator'.")
        print("  3. Change 'GPU P100' to 'GPU T4x2'.")
        print("  4. Restart the session and Run All again.")
        print("="*80 + "\\n")
        sys.exit("Execution stopped: Please switch to GPU T4x2.")
else:
    print("WARNING: No GPU detected! Make sure you enabled the T4x2 GPU accelerator in the Kaggle session settings.")
"""

for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "3. Verify PyTorch" in "".join(cell["source"]):
        cell["source"] = cell_source
        break

with open(file_path, "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated.")
