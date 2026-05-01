from datasets import load_from_disk
import subprocess

print("=== DATASET STATS ===")
d = load_from_disk("/workspace/data/gutenberg")
print(d)
print()

print("=== PRETRAIN LOSS (first 5 logs) ===")
result = subprocess.run(
    ["grep", "-E", "Training Loss|Validation Loss", "/workspace/train.log"],
    capture_output=True, text=True
)
lines = result.stdout.strip().split("\n")
for line in lines[:5]:
    print(line)
print("...")
print("=== PRETRAIN LOSS (last 5 logs) ===")
for line in lines[-5:]:
    print(line)
print()

print("=== SFT LOSS (first 5 logs) ===")
result2 = subprocess.run(
    ["grep", "-E", "Training Loss|Validation Loss", "/tmp/sft.log"],
    capture_output=True, text=True
)
lines2 = result2.stdout.strip().split("\n")
for line in lines2[:5]:
    print(line)
print("...")
print("=== SFT LOSS (last 5 logs) ===")
for line in lines2[-5:]:
    print(line)
