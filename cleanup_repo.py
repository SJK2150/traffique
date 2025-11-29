"""
Clean up repository by removing unnecessary files before pushing
"""
import os
from pathlib import Path

def cleanup_repo():
    """Remove unnecessary files and keep only essential ones"""
    
    base_dir = Path(__file__).parent
    
    # Files to keep (essential documentation and code)
    keep_md_files = {
        'README.md',
        'LICENSE',
    }
    
    # MD files to remove (excessive documentation)
    md_files_to_remove = [
        'ANNOTATION_TIME_GUIDE.md',
        'CHECKLIST.md',
        'COMMANDS.md',
        'DESIGN_UPDATE.md',
        'DIAGRAMS.md',
        'ENVIRONMENT.md',
        'FINE_TUNING_GUIDE.md',
        'FRONTEND_IMPLEMENTATION.md',
        'FRONTEND_QUICKSTART.md',
        'GETTING_STARTED.md',
        'IMPLEMENTATION_SUMMARY.md',
        'INSTALLATION_TESTING.md',
        'MANUAL_ANNOTATION_GUIDE.md',
        'MULTICAMERA_QUICKSTART.md',
        'MULTICAMERA_SETUP.md',
        'MULTI_CAMERA_GUIDE.md',
        'QUICKSTART.md',
        'README_MULTICAM.md',
        'STRUCTURE.md',
        'VISUAL_GUIDE.md',
    ]
    
    # Temporary/test files to remove
    temp_files = [
        'temp_frame.jpg',
        'test_frame_comparison.jpg',
    ]
    
    # Old/unused Python scripts to remove
    unused_scripts = [
        'annotate_footage.py',
        'auto_annotate.py',
        'check_training.py',
        'check_videos.py',
        'create_diagram.py',
        'debug_model.py',
        'extract_frames.py',
        'find_aerial_model.py',
        'prepare_training_data.py',
        'setup_annotation_tool.py',
        'setup_manual_annotation.py',
        'setup_multicam_annotation.py',
        'show_busy_frames.py',
        'smart_annotate.py',
        'test_finetuned_model.py',
        'test_visdrone_model.py',
        'train_model.py',
        'train_with_manual_data.py',
        'train_yolo.py',
        'verify_detections.py',
    ]
    
    # Old YOLO model files (keep only what's needed)
    old_models = [
        'yolo11n.pt',
        'yolov8l.pt',
        'yolov8x.pt',
        # Keep: yolov8n.pt, yolov8s.pt, yolov8m.pt
    ]
    
    removed_files = []
    
    print("="*70)
    print("🧹 CLEANING UP REPOSITORY")
    print("="*70)
    
    # Remove MD files
    print("\n📄 Removing excessive documentation files...")
    for md_file in md_files_to_remove:
        file_path = base_dir / md_file
        if file_path.exists():
            os.remove(file_path)
            removed_files.append(md_file)
            print(f"   ✓ Removed: {md_file}")
    
    # Remove temporary files
    print("\n🗑️  Removing temporary files...")
    for temp_file in temp_files:
        file_path = base_dir / temp_file
        if file_path.exists():
            os.remove(file_path)
            removed_files.append(temp_file)
            print(f"   ✓ Removed: {temp_file}")
    
    # Remove unused scripts
    print("\n🐍 Removing unused Python scripts...")
    for script in unused_scripts:
        file_path = base_dir / script
        if file_path.exists():
            os.remove(file_path)
            removed_files.append(script)
            print(f"   ✓ Removed: {script}")
    
    # Remove old model files
    print("\n🤖 Removing old YOLO model files...")
    for model in old_models:
        file_path = base_dir / model
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            os.remove(file_path)
            removed_files.append(model)
            print(f"   ✓ Removed: {model} ({size_mb:.1f} MB)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 CLEANUP SUMMARY")
    print("="*70)
    print(f"\n✅ Removed {len(removed_files)} files")
    
    print("\n📦 Essential files kept:")
    essential_files = [
        'README.md',
        'LICENSE',
        'requirements.txt',
        'config.yaml',
        'main.py',
        'process_multicam_sahi.py',
        'process_multicam_sahi_visdrone.py',
        'test_sahi.py',
        'test_visdrone_comparison.py',
        'compare_models.py',
        'analyze_results.py',
        'analytics.py',
        'api_server.py',
        'calibration.py',
        'fusion.py',
        'vehicle.py',
        'yolov8n.pt',
        'yolov8s.pt',
        'yolov8m.pt',
    ]
    
    for f in essential_files:
        if (base_dir / f).exists():
            print(f"   ✓ {f}")
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE!")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Review the changes")
    print("   2. git add .")
    print("   3. git commit -m 'Clean up repository and add VisDrone support'")
    print("   4. git push")
    
    return removed_files

if __name__ == '__main__':
    removed = cleanup_repo()
