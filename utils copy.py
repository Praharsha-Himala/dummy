import cv2
import torch
import torch.nn as nn
import torch
import csv
import os
from pathlib import Path

#--------------------------------------------------------------------------------------------------------------------------
def yolo_to_corners(pred, image_height, image_width):
    cx, cy, w, h = pred[:4]
    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)
    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)
    return [x1, y1, x2, y2]
#--------------------------------------------------------------------------------------------------------------------------

def process_video_with_model(video_path, model, trim_seconds=10, output_dir="processed_videos", device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Error opening video file.")
        return []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps
    num_segments = duration // trim_seconds
    video_name = Path(video_path).stem
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[✓] Processing '{video_name}': {duration}s at {fps} FPS -> {num_segments} segments of {trim_seconds}s")
    predictions_all = []
    segment_idx = 0
    while segment_idx < num_segments:
        segment_frames = []
        for _ in range(trim_seconds * fps):
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames.append(frame)
        if not segment_frames:
            break
        segment_preds = []
        out_path = os.path.join(output_dir, f"{video_name}_trim_{segment_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        for frame in segment_frames:
            orig_frame = frame.copy()
            height, width = orig_frame.shape[:2]
            resized = cv2.resize(orig_frame, (224, 224))
            input_tensor = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                model.eval()
                pred = model(input_tensor).squeeze(0).cpu()
            x1, y1, x2, y2 = map(int, yolo_to_corners(pred, height, width))
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            out.write(orig_frame)
            segment_preds.append([x1, y1, x2, y2])
        out.release()
        print(f"[✓] Saved: {out_path}")
        predictions_all.append(segment_preds)
        segment_idx += 1
    cap.release()
    return predictions_all

#--------------------------------------------------------------------------------------------------------------------------

def plot_predictions_on_full_video(original_video_path, preds):
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print("[!] Error opening video file.")
        return
    save_dir = os.path.dirname(original_video_path)
    video_name = os.path.splitext(os.path.basename(original_video_path))[0]
    output_path = os.path.join(save_dir, f"{video_name}_with_boxes.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_index = 0
    segment_index = 0
    while True:
        ret, frame = cap.read()
        if not ret or segment_index >= len(preds):
            break
        if frame_index >= len(preds[segment_index]):
            segment_index += 1
            frame_index = 0
            continue
        x1, y1, x2, y2 = preds[segment_index][frame_index]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)
        frame_index += 1
    cap.release()
    out.release()
    print(f"[✓] Annotated video saved at: {output_path}")

#--------------------------------------------------------------------------------------------------------------------------

def write_predictions_to_csv(preds, original_video_path):
    video_name = os.path.splitext(os.path.basename(original_video_path))[0]
    save_dir = os.path.dirname(original_video_path)
    csv_path = os.path.join(save_dir, f"{video_name}_predictions.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video_file", "frame_index", "x1", "y1", "x2", "y2"])
        for i, segment_preds in enumerate(preds):
            video_segment_name = f"{video_name}_trim_{i}"
            for frame_index, (x1, y1, x2, y2) in enumerate(segment_preds):
                writer.writerow([video_segment_name, frame_index, x1, y1, x2, y2])
    print(f"[✓] Predictions CSV saved at: {csv_path}")

#--------------------------------------------------------------------------------------------------------------------------

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8)) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 128 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.global_avg_pool(self.pool3(torch.relu(self.conv3(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
   
def load_checkpoint(checkpoint, architecture):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    print("loading checkpoint...")
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    model = architecture()
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval().to(device)

#--------------------------------------------------------------------------------------------------------------------------


def rpi():
    path = r"E:\Downloads\1Lcrop_1.mp4"
    model_path = r""
    model = load_checkpoint(model_path, CNNModel) 
    preds = process_video_with_model(path, model, trim_seconds=10)
    write_predictions_to_csv(preds, path)
    



