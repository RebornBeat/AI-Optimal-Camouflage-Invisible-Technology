import cv2
import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as TF

# Camera and Display Configuration
NUM_CAMERAS = 4  # Adjust based on the number of cameras used
DISPLAY_RESOLUTION = (1920, 1080)  # Full HD resolution for displays
CAMERA_ANGLES = [0, 90, 180, 270]  # Four cameras placed at 90° intervals

class PerspectiveTransformer(nn.Module):
    """AI Model to adjust video feed for observer perspective."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * DISPLAY_RESOLUTION[0] * DISPLAY_RESOLUTION[1], 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def initialize_cameras():
    """Initialize camera feeds."""
    cameras = [cv2.VideoCapture(i) for i in range(NUM_CAMERAS)]
    for idx, cam in enumerate(cameras):
        if not cam.isOpened():
            print(f"Camera {idx} failed to initialize.")
            exit(1)
    return cameras

def capture_frames(cameras):
    """Capture frames from all cameras."""
    frames = []
    for cam in cameras:
        ret, frame = cam.read()
        if ret:
            frames.append(cv2.resize(frame, DISPLAY_RESOLUTION))
    return frames

def transform_frame(frame, angle, model):
    """Apply AI-based perspective transformation."""
    tensor_frame = TF.to_tensor(frame).unsqueeze(0)
    transformed_frame = model(tensor_frame)
    return transformed_frame.detach().numpy()

def display_frames(frames):
    """Send frames to connected displays."""
    for idx, frame in enumerate(frames):
        # Simulate display output with OpenCV windows (replace with actual display drivers)
        cv2.imshow(f"Display {idx}", frame)

def main():
    # Load or initialize AI model
    model = PerspectiveTransformer()
    model.eval()

    # Initialize cameras
    cameras = initialize_cameras()

    try:
        while True:
            # Capture frames from all cameras
            raw_frames = capture_frames(cameras)

            # Process and transform frames
            transformed_frames = [
                transform_frame(frame, angle, model)
                for frame, angle in zip(raw_frames, CAMERA_ANGLES)
            ]

            # Display processed frames
            display_frames(transformed_frames)

            # Break loop on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release all resources
        for cam in cameras:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
