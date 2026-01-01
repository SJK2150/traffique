"""
Label Studio Setup Script
Initializes Label Studio project for VisDrone vehicle annotation correction
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_label_studio_installed():
    """Check if Label Studio is installed"""
    try:
        import label_studio
        print("✅ Label Studio is already installed")
        return True
    except ImportError:
        print("❌ Label Studio not found")
        return False

def install_label_studio():
    """Install Label Studio via pip"""
    print("\n📦 Installing Label Studio...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "label-studio", "label-studio-converter"])
        print("✅ Label Studio installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def create_label_config():
    """Create Label Studio annotation configuration for VisDrone vehicle detection"""
    
    # VisDrone vehicle classes
    config = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pedestrian" background="#FF6B6B"/>
    <Label value="people" background="#FFA07A"/>
    <Label value="bicycle" background="#4ECDC4"/>
    <Label value="car" background="#45B7D1"/>
    <Label value="van" background="#96CEB4"/>
    <Label value="truck" background="#FFEAA7"/>
    <Label value="tricycle" background="#DFE6E9"/>
    <Label value="awning-tricycle" background="#A29BFE"/>
    <Label value="bus" background="#FD79A8"/>
    <Label value="motor" background="#FDCB6E"/>
  </RectangleLabels>
</View>"""
    
    return config

def create_project_config():
    """Create Label Studio project configuration"""
    
    project_dir = Path("label_studio_project")
    project_dir.mkdir(exist_ok=True)
    
    config = {
        "title": "VisDrone Vehicle Annotation Correction",
        "description": "Correct vehicle classifications from VisDrone model predictions",
        "label_config": create_label_config(),
        "sampling": "Sequential",
        "show_instruction": True,
        "instruction": """
# VisDrone Vehicle Annotation Correction

## Instructions:
1. Review the pre-annotated bounding boxes from the model
2. Correct any misclassifications by changing the label
3. Add missing vehicles that the model didn't detect
4. Delete false positive detections
5. Adjust bounding boxes if needed for better accuracy

## Vehicle Classes:
- **car**: Standard passenger vehicles
- **van**: Vans and minivans
- **truck**: Trucks and large vehicles
- **bus**: Buses
- **motor**: Motorcycles and scooters
- **bicycle**: Bicycles
- **tricycle**: Three-wheeled vehicles
- **awning-tricycle**: Covered tricycles
- **pedestrian**: Single person
- **people**: Group of people

## Tips:
- Focus on correcting misclassifications first
- Ensure bounding boxes are tight around vehicles
- Mark unclear/occluded vehicles if uncertain
"""
    }
    
    config_path = project_dir / "project_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Project configuration saved to: {config_path}")
    return config_path

def create_startup_script():
    """Create a convenient startup script for Label Studio"""
    
    startup_script = """@echo off
echo Starting Label Studio for VisDrone Annotation Correction...
echo.
echo Label Studio will open at: http://localhost:8080
echo.
echo To stop Label Studio, press Ctrl+C in this window
echo.

label-studio start label_studio_project --port 8080
"""
    
    script_path = Path("start_labelstudio.bat")
    with open(script_path, 'w') as f:
        f.write(startup_script)
    
    print(f"✅ Startup script created: {script_path}")
    print(f"   Run this script to start Label Studio anytime!")
    return script_path

def main():
    """Main setup function"""
    print("=" * 60)
    print("Label Studio Setup for VisDrone Model Improvement")
    print("=" * 60)
    
    # Check/Install Label Studio
    if not check_label_studio_installed():
        print("\nLabel Studio needs to be installed.")
        response = input("Install now? (y/n): ").lower()
        if response == 'y':
            if not install_label_studio():
                print("\n❌ Setup failed. Please install manually:")
                print("   pip install label-studio label-studio-converter")
                return
        else:
            print("\n⚠️  Please install Label Studio manually:")
            print("   pip install label-studio label-studio-converter")
            return
    
    # Create project configuration
    print("\n📝 Creating project configuration...")
    config_path = create_project_config()
    
    # Create startup script
    print("\n📝 Creating startup script...")
    script_path = create_startup_script()
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run the startup script:")
    print(f"   {script_path}")
    print("\n2. Open your browser to: http://localhost:8080")
    print("\n3. Create a new project using the configuration in:")
    print(f"   {config_path}")
    print("\n4. Use the inference scripts to generate predictions:")
    print("   python batch_inference.py --input <video_or_folder>")
    print("\n5. Import predictions to Label Studio:")
    print("   python import_to_labelstudio.py --predictions <predictions_folder>")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
