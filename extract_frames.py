"""
Frame Extraction Script for Dataset Preparation
Extracts 1 frame per second from drone footage for labeling
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, 
                   output_dir: str, 
                   fps: int = 1,
                   max_frames: int = None,
                   prefix: str = "frame"):
    """
    Extract frames from video at specified interval.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (1 = one frame per second)
        max_frames: Maximum number of frames to extract (None = all)
        prefix: Prefix for output filenames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video Properties:")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    expected_frames = int(duration * fps)
    
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    
    print(f"\nExtraction Settings:")
    print(f"  Extract 1 frame every {frame_interval} frames")
    print(f"  Expected output: ~{expected_frames} frames")
    print(f"  Output directory: {output_dir}\n")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                output_filename = f"{prefix}_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                output_filepath = output_path / output_filename
                
                cv2.imwrite(str(output_filepath), frame)
                saved_count += 1
                pbar.update(1)
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_count += 1
    
    cap.release()
    
    print(f"\n✓ Extraction complete!")
    print(f"  Processed: {frame_count} frames")
    print(f"  Saved: {saved_count} frames")
    print(f"  Output: {output_dir}")
    
    return saved_count


def extract_from_multiple_videos(video_paths: list, 
                                 output_dir: str, 
                                 fps: int = 1,
                                 frames_per_video: int = 150):
    """
    Extract frames from multiple video files.
    
    Args:
        video_paths: List of video file paths
        output_dir: Base output directory
        fps: Frames per second to extract
        frames_per_video: Number of frames to extract per video
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*60}")
        print(f"Processing Video {i}/{len(video_paths)}: {video_path}")
        print(f"{'='*60}")
        
        # Create subfolder for each video
        video_name = Path(video_path).stem
        video_output_dir = output_path / video_name
        
        try:
            saved = extract_frames(
                video_path=video_path,
                output_dir=str(video_output_dir),
                fps=fps,
                max_frames=frames_per_video,
                prefix=f"{video_name}"
            )
            total_saved += saved
            
        except Exception as e:
            print(f"✗ Error processing {video_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All videos processed!")
    print(f"Total frames extracted: {total_saved}")
    print(f"{'='*60}")


def create_dataset_structure(base_dir: str):
    """
    Create standard dataset directory structure.
    
    Args:
        base_dir: Base directory for dataset
    """
    base_path = Path(base_dir)
    
    # Create directories
    directories = [
        "raw_frames",        # Extracted frames
        "labeled/images",    # Images for training
        "labeled/labels",    # YOLO format labels
        "splits/train",      # Training split
        "splits/val",        # Validation split
        "splits/test"        # Test split
    ]
    
    for dir_name in directories:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Dataset structure created at: {base_dir}")
    for dir_name in directories:
        print(f"  - {dir_name}/")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from drone footage for dataset preparation"
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to video file"
    )
    
    parser.add_argument(
        "--videos", "-vs",
        nargs="+",
        help="Paths to multiple video files"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/dataset/raw_frames",
        help="Output directory for extracted frames"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to extract (default: 1)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: unlimited)"
    )
    
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=150,
        help="Frames to extract per video when processing multiple videos (default: 150)"
    )
    
    parser.add_argument(
        "--create-structure",
        action="store_true",
        help="Create dataset directory structure"
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/dataset",
        help="Base directory for dataset structure"
    )
    
    args = parser.parse_args()
    
    # Create dataset structure if requested
    if args.create_structure:
        create_dataset_structure(args.dataset_dir)
        if not args.video and not args.videos:
            return
    
    # Extract frames
    if args.video:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            max_frames=args.max_frames
        )
    elif args.videos:
        extract_from_multiple_videos(
            video_paths=args.videos,
            output_dir=args.output,
            fps=args.fps,
            frames_per_video=args.frames_per_video
        )
    else:
        print("Error: Please provide --video or --videos argument")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
