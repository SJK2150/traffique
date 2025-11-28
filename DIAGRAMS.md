# System Architecture Diagrams

This document provides visual representations of the system architecture.

---

## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-CAMERA TRAFFIC ANALYSIS                  │
│                           SYSTEM                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Camera 1   │  │  Camera 2   │  │  Camera N   │
│   Video     │  │   Video     │  │   Video     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
            ┌───────────▼───────────┐
            │  YOLOv8 Detection     │
            │  + ByteTrack          │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  Homography Transform │
            │  (Camera → Global)    │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  Multi-Camera Fusion  │
            │  (Coordinate-based)   │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼─────┐ ┌──────▼──────┐
│ CSV Logging  │ │   Zones    │ │Visualization│
│              │ │  Counting  │ │             │
└──────────────┘ └────────────┘ └─────────────┘
```

---

## 2. Data Flow Architecture

```
INPUT LAYER
┌──────────────────────────────────────────────────────────┐
│  Camera 1 Stream    Camera 2 Stream    Camera N Stream   │
└────────┬──────────────────┬──────────────────┬───────────┘
         │                  │                  │
         
PROCESSING LAYER
┌────────▼──────────┐ ┌─────▼──────────┐ ┌───▼───────────┐
│ Frame Reader      │ │ Frame Reader   │ │ Frame Reader  │
│ + Resize          │ │ + Resize       │ │ + Resize      │
└────────┬──────────┘ └─────┬──────────┘ └───┬───────────┘
         │                  │                  │
┌────────▼──────────┐ ┌─────▼──────────┐ ┌───▼───────────┐
│ YOLO Detection    │ │ YOLO Detection │ │ YOLO Detection│
│ (bbox, class,     │ │ (bbox, class,  │ │ (bbox, class, │
│  confidence)      │ │  confidence)   │ │  confidence)  │
└────────┬──────────┘ └─────┬──────────┘ └───┬───────────┘
         │                  │                  │
┌────────▼──────────┐ ┌─────▼──────────┐ ┌───▼───────────┐
│ ByteTrack         │ │ ByteTrack      │ │ ByteTrack     │
│ (local_id)        │ │ (local_id)     │ │ (local_id)    │
└────────┬──────────┘ └─────┬──────────┘ └───┬───────────┘
         │                  │                  │
         
TRANSFORMATION LAYER
┌────────▼──────────┐ ┌─────▼──────────┐ ┌───▼───────────┐
│ Homography H₁     │ │ Homography H₂  │ │ Homography Hₙ │
│ (x,y) → (X,Y)     │ │ (x,y) → (X,Y)  │ │ (x,y) → (X,Y) │
└────────┬──────────┘ └─────┬──────────┘ └───┬───────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
FUSION LAYER                │
┌───────────────────────────▼────────────────────────────┐
│              COORDINATE FUSION ENGINE                   │
│  • Spatial Distance Matching                           │
│  • Temporal Prediction                                 │
│  • Confidence Weighting                                │
│  • Cross-Camera ID Management                          │
└───────────────────────────┬────────────────────────────┘
                            │
VEHICLE MANAGEMENT          │
┌───────────────────────────▼────────────────────────────┐
│              VEHICLE MANAGER                            │
│  • Global ID Assignment                                │
│  • Trajectory History (Array t)                        │
│  • Speed & Distance Calculation                        │
│  • Active/Inactive State                               │
└───────────────────────────┬────────────────────────────┘
                            │
OUTPUT LAYER                │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼──────┐  ┌────────▼──────┐  ┌───────▼────────┐
│ CSV Logger    │  │ Zone Counter  │  │ Visualizer     │
│               │  │               │  │                │
│ • Timestamp   │  │ • Polygon     │  │ • Global Map   │
│ • Position    │  │   Test        │  │ • Trajectories │
│ • Class       │  │ • Entry/Exit  │  │ • Statistics   │
│ • Speed       │  │   Counting    │  │ • Real-time    │
└───────────────┘  └───────────────┘  └────────────────┘
```

---

## 3. Coordinate Transformation

```
CAMERA VIEW                          GLOBAL MAP
┌─────────────────┐                 ┌──────────────────┐
│                 │                 │                  │
│    📹           │    Homography   │    🗺️           │
│   (x,y)         │  ─────────────▶ │   (X,Y)         │
│                 │    Matrix H     │                  │
│  ┌────┐         │                 │     ●            │
│  │Car │ ●       │                 │   Vehicle        │
│  └────┘         │                 │   Position       │
│                 │                 │                  │
└─────────────────┘                 └──────────────────┘

Mathematical Transformation:
┌───┐   ┌─────────┐   ┌───┐
│ X │   │h₁₁ h₁₂ h₁₃│   │ x │
│ Y │ = │h₂₁ h₂₂ h₂₃│ × │ y │
│ w │   │h₃₁ h₃₂ h₃₃│   │ 1 │
└───┘   └─────────┘   └───┘

Then normalize: X' = X/w, Y' = Y/w
```

---

## 4. Fusion Process

```
STEP 1: LOCAL DETECTION
┌──────────┐  ┌──────────┐
│ Camera 1 │  │ Camera 2 │
│          │  │          │
│  🚗 ID:5 │  │  🚗 ID:3 │
│  📍(x₁,y₁)│  │  📍(x₂,y₂)│
└────┬─────┘  └────┬─────┘
     │            │
     
STEP 2: TRANSFORM TO GLOBAL
     │            │
┌────▼─────┐ ┌────▼─────┐
│H₁ Matrix │ │H₂ Matrix │
└────┬─────┘ └────┬─────┘
     │            │
┌────▼─────┐ ┌────▼─────┐
│ 📍(X₁,Y₁)│ │ 📍(X₂,Y₂)│
└────┬─────┘ └────┬─────┘
     │            │
     
STEP 3: DISTANCE CHECK
     └──────┬─────┘
            │
      ┌─────▼──────┐
      │  d(P₁,P₂)  │
      │  < 2.0m ?  │
      └─────┬──────┘
            │
    ┌───────┴────────┐
    │ YES            │ NO
    │                │
    ▼                ▼
┌───────────┐  ┌──────────┐
│MERGE IDs  │  │KEEP BOTH │
│Global:42  │  │Cam1:5    │
│           │  │Cam2:3    │
└───────────┘  └──────────┘
```

---

## 5. Vehicle Lifecycle

```
┌───────────────────────────────────────────────────────────┐
│                    VEHICLE LIFECYCLE                       │
└───────────────────────────────────────────────────────────┘

 BIRTH                    TRACKING                   DEATH
   │                         │                         │
   ▼                         ▼                         ▼
┌──────┐                ┌─────────┐              ┌─────────┐
│First │                │ Update  │              │Lost for │
│Detect│───────────────▶│Position │──────────────▶│>30      │
└──────┘                │Trajectory│              │frames   │
   │                    └─────────┘              └─────────┘
   │                         │                         │
   ▼                         ▼                         ▼
┌──────────┐           ┌──────────┐             ┌──────────┐
│Assign    │           │Camera    │             │Archive   │
│Global ID │           │Transition│             │Vehicle   │
└──────────┘           └──────────┘             └──────────┘

STATE DIAGRAM:
                    ┌──────────┐
          ┌────────▶│  ACTIVE  │◀─────────┐
          │         └────┬─────┘          │
          │              │                │
     Detected        No Detection     Detected
          │              │                │
          │         ┌────▼─────┐          │
          └─────────│   LOST   │──────────┘
                    └────┬─────┘
                         │
                    Lost >30 frames
                         │
                    ┌────▼─────┐
                    │ REMOVED  │
                    └──────────┘
```

---

## 6. Zone Counting System

```
GLOBAL MAP WITH ZONES
┌──────────────────────────────────────┐
│                                      │
│   Zone 1: "Intersection A"          │
│   ╔═══════════╗                     │
│   ║  🚗 🚲   ║                     │
│   ║          ║                     │
│   ╚═══════════╝                     │
│   Count: 2                          │
│                                      │
│           Zone 2: "Highway Entry"   │
│           ┌───────────────┐         │
│           │  🚗 🚗 🚗     │         │
│           └───────────────┘         │
│           Count: 3                  │
│                                      │
└──────────────────────────────────────┘

POINT-IN-POLYGON TEST:
         Zone Polygon
    ╔═════════════════╗
    ║                 ║
    ║    ● Point      ║ ──▶ Inside
    ║                 ║
    ╚═════════════════╝

  ● Point                  ──▶ Outside
```

---

## 7. Analytics Pipeline

```
┌────────────────────────────────────────────────────┐
│              REAL-TIME ANALYTICS                    │
└────────────────────────────────────────────────────┘

Vehicle Data Stream
      │
      ▼
┌──────────────┐
│ Frame Update │
└──────┬───────┘
       │
   ┌───┴────┬────────┬─────────┐
   │        │        │         │
   ▼        ▼        ▼         ▼
┌────┐  ┌────┐  ┌──────┐  ┌────────┐
│Zone│  │CSV │  │Heat  │  │Stats   │
│Count  │Log │  │map   │  │Calc    │
└──┬─┘  └──┬─┘  └───┬──┘  └───┬────┘
   │       │        │         │
   ▼       ▼        ▼         ▼
Output  Output  Output    Display
Buffer  File    Image     Overlay

POST-PROCESSING
      │
      ▼
┌──────────────────┐
│ analyze_results  │
└────────┬─────────┘
         │
    ┌────┴────┬──────────┬─────────┐
    │         │          │         │
    ▼         ▼          ▼         ▼
 Plots   Statistics  Heatmap   HTML
                                Report
```

---

## 8. Training Pipeline

```
┌────────────────────────────────────────────────────┐
│              TRAINING WORKFLOW                      │
└────────────────────────────────────────────────────┘

Raw Videos
    │
    ▼
┌──────────────┐
│Extract Frames│
│ (1 fps)      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Label      │
│(Roboflow/    │
│ CVAT)        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Organize    │
│  Dataset     │
│ Train/Val/   │
│  Test        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│YOLOv8-Nano   │
│  Training    │
│  (50 epochs) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validation  │
│  & Testing   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   best.pt    │
│  (Custom     │
│   Model)     │
└──────────────┘
```

---

## 9. System Components

```
┌──────────────────────────────────────────────────────────┐
│                    MAIN.PY                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │         MultiCameraTrafficSystem                   │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │      CameraProcessor x N                    │  │  │
│  │  │  • Video Reading                            │  │  │
│  │  │  • YOLO Detection                           │  │  │
│  │  │  • ByteTrack                                │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │      MultiCameraFusion                      │  │  │
│  │  │  • Homography Transform                     │  │  │
│  │  │  • Coordinate Fusion                        │  │  │
│  │  │  • Vehicle Manager                          │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │      TrafficAnalytics                       │  │  │
│  │  │  • Zone Manager                             │  │  │
│  │  │  • CSV Logger                               │  │  │
│  │  │  • Heatmap Generator                        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

SUPPORTING MODULES:

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  vehicle.py  │  │  fusion.py   │  │analytics.py  │
│              │  │              │  │              │
│• Vehicle     │  │• Coordinate  │  │• Zone        │
│• Manager     │  │  Fusion      │  │• Logger      │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 10. Performance Optimization

```
┌────────────────────────────────────────────────┐
│         OPTIMIZATION STRATEGIES                 │
└────────────────────────────────────────────────┘

INPUT OPTIMIZATION
┌──────────────┐
│  Frame Skip  │──▶ Process every Nth frame
│  Resize      │──▶ Reduce resolution
│  ROI Crop    │──▶ Process only relevant area
└──────────────┘

MODEL OPTIMIZATION
┌──────────────┐
│  YOLOv8n     │──▶ Smallest model variant
│  ONNX Export │──▶ Faster inference
│  TensorRT    │──▶ GPU optimization
└──────────────┘

PROCESSING OPTIMIZATION
┌──────────────┐
│  Batch Proc  │──▶ Process multiple frames
│  Parallel    │──▶ Multi-camera parallel
│  GPU Accel   │──▶ CUDA utilization
└──────────────┘

MEMORY OPTIMIZATION
┌──────────────┐
│  Buffer Size │──▶ Limit trajectory length
│  Cleanup     │──▶ Remove old vehicles
│  Streaming   │──▶ Process without loading all
└──────────────┘
```

---

These diagrams illustrate the complete system architecture, data flow, and component interactions in the Multi-Camera Traffic Analysis System.
