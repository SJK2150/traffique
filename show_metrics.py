import pandas as pd
from pathlib import Path

print("📊 Training Metrics Summary\n")

# Read results CSV
results_csv = Path('runs/finetune/visdrone_finetuned/results.csv')
df = pd.read_csv(results_csv)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

print("Training completed in", len(df), "epochs")
print("\n" + "="*60)

# Get final epoch metrics
final_epoch = df.iloc[-1]

print("\n🎯 Final Metrics (Last Epoch):")
print(f"  Epoch: {int(final_epoch['epoch'])}")

# Box metrics
if 'metrics/mAP50(B)' in df.columns:
    print(f"  mAP50: {final_epoch['metrics/mAP50(B)']:.4f}")
if 'metrics/mAP50-95(B)' in df.columns:
    print(f"  mAP50-95: {final_epoch['metrics/mAP50-95(B)']:.4f}")

# Precision and Recall
if 'metrics/precision(B)' in df.columns:
    print(f"  Precision: {final_epoch['metrics/precision(B)']:.4f}")
if 'metrics/recall(B)' in df.columns:
    print(f"  Recall: {final_epoch['metrics/recall(B)']:.4f}")

# Losses
print(f"\n📉 Final Losses:")
if 'train/box_loss' in df.columns:
    print(f"  Box Loss: {final_epoch['train/box_loss']:.4f}")
if 'train/cls_loss' in df.columns:
    print(f"  Class Loss: {final_epoch['train/cls_loss']:.4f}")
if 'train/dfl_loss' in df.columns:
    print(f"  DFL Loss: {final_epoch['train/dfl_loss']:.4f}")

# Best metrics
print(f"\n🏆 Best Metrics Across All Epochs:")
if 'metrics/mAP50(B)' in df.columns:
    best_map50 = df['metrics/mAP50(B)'].max()
    best_map50_epoch = df['metrics/mAP50(B)'].idxmax() + 1
    print(f"  Best mAP50: {best_map50:.4f} (epoch {best_map50_epoch})")

if 'metrics/mAP50-95(B)' in df.columns:
    best_map = df['metrics/mAP50-95(B)'].max()
    best_map_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
    print(f"  Best mAP50-95: {best_map:.4f} (epoch {best_map_epoch})")

print("\n" + "="*60)
print("\n✅ Fine-tuning complete!")
print(f"\n📁 Results saved to: runs/finetune/visdrone_finetuned/")
print(f"🎯 Best model: visdrone_finetuned.pt")
print(f"\n📊 View training curves: runs/finetune/visdrone_finetuned/results.png")
