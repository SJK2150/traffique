"""
Import Predictions to Label Studio
Converts model predictions to Label Studio JSON format and imports them
"""

import json
import base64
import argparse
from pathlib import Path
from tqdm import tqdm
import requests

class LabelStudioImporter:
    """Convert predictions to Label Studio format and import"""
    
    def __init__(self, label_studio_url="http://localhost:8080", api_key=None):
        """
        Initialize importer
        
        Args:
            label_studio_url: Label Studio server URL
            api_key: API key for Label Studio (optional for local use)
        """
        self.url = label_studio_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Token {api_key}'
    
    def convert_to_labelstudio_format(self, predictions_file, image_base_path=None):
        """
        Convert predictions JSON to Label Studio import format
        
        Args:
            predictions_file: Path to predictions.json from batch_inference.py
            image_base_path: Base path for images (if different from predictions)
        
        Returns:
            List of Label Studio tasks with pre-annotations
        """
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        tasks = []
        
        print(f"📝 Converting {len(predictions)} predictions to Label Studio format...")
        
        for img_path, pred_data in tqdm(predictions.items(), desc="Converting"):
            img_path = Path(img_path)
            
            # Use custom base path if provided
            if image_base_path:
                img_path = Path(image_base_path) / img_path.name
            
            # Create task with pre-annotations
            task = {
                "data": {
                    "image": f"/data/local-files/?d={img_path.as_posix()}"
                },
                "predictions": [{
                    "result": [],
                    "score": 0.0
                }]
            }
            
            # Convert detections to Label Studio format
            total_score = 0
            for det in pred_data['detections']:
                x1, y1, x2, y2 = det['bbox']
                width = pred_data['width']
                height = pred_data['height']
                
                # Convert to percentage coordinates (Label Studio format)
                x_percent = (x1 / width) * 100
                y_percent = (y1 / height) * 100
                width_percent = ((x2 - x1) / width) * 100
                height_percent = ((y2 - y1) / height) * 100
                
                # Create annotation
                annotation = {
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                        "rotation": 0,
                        "rectanglelabels": [det['class_name']]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "manual"
                }
                
                task["predictions"][0]["result"].append(annotation)
                total_score += det['confidence']
            
            # Average confidence score
            if len(pred_data['detections']) > 0:
                task["predictions"][0]["score"] = total_score / len(pred_data['detections'])
            
            tasks.append(task)
        
        return tasks
    
    def save_import_file(self, tasks, output_file):
        """
        Save tasks to JSON file for Label Studio import
        
        Args:
            tasks: List of Label Studio tasks
            output_file: Path to save import file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print(f"✅ Label Studio import file saved: {output_file}")
        print(f"   Tasks: {len(tasks)}")
        return output_file
    
    def import_to_project(self, tasks, project_id):
        """
        Import tasks directly to Label Studio project via API
        
        Args:
            tasks: List of Label Studio tasks
            project_id: Label Studio project ID
        
        Returns:
            Response from Label Studio API
        """
        import_url = f"{self.url}/api/projects/{project_id}/import"
        
        print(f"\n📤 Importing {len(tasks)} tasks to Label Studio project {project_id}...")
        
        try:
            response = requests.post(
                import_url,
                headers={**self.headers, 'Content-Type': 'application/json'},
                json=tasks
            )
            
            if response.status_code == 201:
                print(f"✅ Successfully imported {len(tasks)} tasks!")
                return response.json()
            else:
                print(f"❌ Import failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Import error: {e}")
            print(f"\n💡 Manual import instructions:")
            print(f"   1. Open Label Studio: {self.url}")
            print(f"   2. Go to your project")
            print(f"   3. Click 'Import' button")
            print(f"   4. Upload the generated JSON file")
            return None

def main():
    parser = argparse.ArgumentParser(description='Import predictions to Label Studio')
    parser.add_argument('--predictions', required=True, 
                       help='Path to predictions directory (containing predictions.json)')
    parser.add_argument('--output', default='labelstudio_import.json',
                       help='Output import file path')
    parser.add_argument('--image-path', default=None,
                       help='Custom base path for images (optional)')
    parser.add_argument('--url', default='http://localhost:8080',
                       help='Label Studio URL')
    parser.add_argument('--api-key', default=None,
                       help='Label Studio API key (optional)')
    parser.add_argument('--project-id', type=int, default=None,
                       help='Project ID for direct import (optional)')
    
    args = parser.parse_args()
    
    # Find predictions.json
    predictions_dir = Path(args.predictions)
    predictions_file = predictions_dir / 'predictions.json'
    
    if not predictions_file.exists():
        raise FileNotFoundError(f"predictions.json not found in: {predictions_dir}")
    
    # Initialize importer
    importer = LabelStudioImporter(label_studio_url=args.url, api_key=args.api_key)
    
    # Convert predictions
    tasks = importer.convert_to_labelstudio_format(
        predictions_file, 
        image_base_path=args.image_path
    )
    
    # Save import file
    import_file = importer.save_import_file(tasks, args.output)
    
    # Try direct import if project ID provided
    if args.project_id:
        importer.import_to_project(tasks, args.project_id)
    else:
        print(f"\n📋 Manual Import Instructions:")
        print(f"   1. Open Label Studio: {args.url}")
        print(f"   2. Create or open your project")
        print(f"   3. Go to Settings → Cloud Storage")
        print(f"   4. Add local storage path: {predictions_dir / 'frames'}")
        print(f"   5. Click 'Import' and upload: {import_file}")
        print(f"\n✅ Your predictions will appear as pre-annotations!")

if __name__ == "__main__":
    main()
