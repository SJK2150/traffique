"""
Analyze training results
"""

import pandas as pd
import os

def analyze_results():
    results_file = 'runs/detect/train/results.csv'
    
    if not os.path.exists(results_file):
        print("❌ No results file found. Training may not have started.")
        return
    
    df = pd.read_csv(results_file)
    df.columns = df.columns.str.strip()
    
    current_epoch = len(df)
    
    print("\n" + "="*70)
    print("📊 TRAINING RESULTS ANALYSIS")
    print("="*70)
    
    print(f"\n🏋️ Progress: Epoch {current_epoch}/50 ({(current_epoch/50)*100:.1f}%)")
    
    # Loss trends
    print(f"\n📉 Loss Trends (Lower is better):")
    print(f"  Box Loss:   {df['train/box_loss'].iloc[0]:.4f} → {df['train/box_loss'].iloc[-1]:.4f} (↓ {((df['train/box_loss'].iloc[0] - df['train/box_loss'].iloc[-1])/df['train/box_loss'].iloc[0]*100):.1f}%)")
    print(f"  Class Loss: {df['train/cls_loss'].iloc[0]:.4f} → {df['train/cls_loss'].iloc[-1]:.4f} (↓ {((df['train/cls_loss'].iloc[0] - df['train/cls_loss'].iloc[-1])/df['train/cls_loss'].iloc[0]*100):.1f}%)")
    print(f"  DFL Loss:   {df['train/dfl_loss'].iloc[0]:.4f} → {df['train/dfl_loss'].iloc[-1]:.4f} (↓ {((df['train/dfl_loss'].iloc[0] - df['train/dfl_loss'].iloc[-1])/df['train/dfl_loss'].iloc[0]*100):.1f}%)")
    
    # Validation metrics
    print(f"\n✅ Validation Performance (Current):")
    print(f"  Precision:    {df['metrics/precision(B)'].iloc[-1]:.4f} ({df['metrics/precision(B)'].iloc[-1]*100:.1f}%)")
    print(f"  Recall:       {df['metrics/recall(B)'].iloc[-1]:.4f} ({df['metrics/recall(B)'].iloc[-1]*100:.1f}%)")
    print(f"  mAP@0.5:      {df['metrics/mAP50(B)'].iloc[-1]:.4f} ({df['metrics/mAP50(B)'].iloc[-1]*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {df['metrics/mAP50-95(B)'].iloc[-1]:.4f} ({df['metrics/mAP50-95(B)'].iloc[-1]*100:.1f}%)")
    
    # Best performance
    best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
    best_map = df['metrics/mAP50(B)'].max()
    print(f"\n🏆 Best Performance:")
    print(f"  Best mAP@0.5: {best_map:.4f} ({best_map*100:.1f}%) at epoch {best_epoch}")
    
    # Check if still improving
    recent_map = df['metrics/mAP50(B)'].iloc[-5:].values
    if len(recent_map) >= 5:
        improving = recent_map[-1] > recent_map[0]
        trend = "📈 Still improving" if improving else "📉 Plateauing"
        print(f"  Trend: {trend}")
    
    print("\n" + "="*70)
    print("💡 WHAT THIS MEANS:")
    print("="*70)
    
    # Interpret results
    current_map = df['metrics/mAP50(B)'].iloc[-1]
    
    if current_map > 0.7:
        print("✅ EXCELLENT: Model is performing very well!")
        print("   - mAP > 70% means it's detecting vehicles accurately")
        print("   - Should work much better than the pre-trained model")
    elif current_map > 0.5:
        print("✅ GOOD: Model is learning well")
        print("   - mAP > 50% is solid for drone footage")
        print("   - Will detect more vehicles than before")
    elif current_map > 0.3:
        print("⚠️  FAIR: Model is improving but needs more training")
        print("   - May need more annotated frames or more epochs")
    else:
        print("⚠️  POOR: Model struggling to learn")
        print("   - May need more diverse training data")
    
    # Loss analysis
    if df['train/box_loss'].iloc[-1] < 1.5:
        print("\n✅ Box localization: Good - model finding vehicles well")
    else:
        print("\n⚠️  Box localization: Needs improvement")
    
    if df['train/cls_loss'].iloc[-1] < 0.5:
        print("✅ Classification: Good - distinguishing car/truck/bus well")
    else:
        print("⚠️  Classification: Needs improvement")
    
    print("\n" + "="*70)
    print("📈 VISUAL ANALYSIS:")
    print("="*70)
    print("\nOpen: runs/detect/train/results.png")
    print("\nLook for:")
    print("  1. Loss curves going DOWN (all 3: box, cls, dfl)")
    print("  2. mAP curves going UP (both mAP50 and mAP50-95)")
    print("  3. Precision/Recall balanced (both should be high)")
    print("  4. No sudden spikes (would indicate training issues)")
    print("="*70)

if __name__ == '__main__':
    analyze_results()
