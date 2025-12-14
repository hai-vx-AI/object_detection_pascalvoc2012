from voc_dataset import VOCDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import argparse
import os
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description= "Train model")
    parser.add_argument("--video_path", type = str, default = None)
    parser.add_argument("--save_checkpoint", type=str, default="trained_model/best.pt")
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    args = parser.parse_args()
    return args

def test(args):
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"
    ]

    args.video_path = r"D:\deep_learning_object_detection\video_test_od.mp4"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = fasterrcnn_resnet50_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    checkpoint = torch.load(args.save_checkpoint, map_location = "cpu")
    model.load_state_dict(checkpoint)
    model.to(device)

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    out = cv2.VideoWriter("filename.mp4", cv2.VideoWriter_fourcc(*"MJPG"),
                          int(cap.get(cv2.CAP_PROP_FPS)), size)
    model.eval()
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))/255.0
        image = [torch.from_numpy(image).to(device)]

        with torch.no_grad():
            output = model(image)[0]
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]
            for bbox, label, score in zip(boxes, labels, scores):
                if score > args.conf_threshold:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    category = classes[int(label)]
                    cv2.putText(frame, category, (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    args = get_args()
    test(args)

