from ultralytics import YOLO

# Load your custom trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Set the input video
video_path = "./As.mp4"

# Choose tracker: 'bytetrack.yaml' (default) or 'deepsort.yaml'
tracker_config = "bytetrack.yaml"  # or "deepsort.yaml" if configured

# Run tracking
model.track(
    source=video_path,
    conf=0.2,               # Confidence threshold
    iou=0.5,                # IOU threshold
    tracker=tracker_config,
    save=False,              # Save output video to 'runs/track'
    show=True,              # Show the tracking live
    verbose=False           # Reduce console output
)
