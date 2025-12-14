import torch
import cv2
import numpy as np
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Cấu hình thay cho argparse ---
class Args:
    # Đổi tên file này thành tên file video bạn upload lên Colab
    video_path = "/content/drive/MyDrive/face-demographics-walking.mp4"
    # Đổi tên file này thành đường dẫn file model của bạn
    save_checkpoint = "/content/drive/MyDrive/VOC_Training/best.pt"
    conf_threshold = 0.7 # Tăng lên 0.5 để giảm bớt khung hình rác
    output_name = "output_result.mp4"

args = Args()

def test_on_colab(args):
    # Danh sách class (đảm bảo khớp với lúc train)
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"
    ]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on: {device}")

    # 1. Khởi tạo Model
    model = fasterrcnn_resnet50_fpn(pretrained=False) # False vì mình sẽ load weight riêng
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)

    # 2. Load weights
    if os.path.exists(args.save_checkpoint):
        checkpoint = torch.load(args.save_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print("Đã load checkpoint thành công.")
    else:
        print(f"LỖI: Không tìm thấy file {args.save_checkpoint}. Hãy upload file model lên Colab.")
        return

    model.to(device)
    model.eval()

    # 3. Xử lý Video
    if not os.path.exists(args.video_path):
        print(f"LỖI: Không tìm thấy file video {args.video_path}.")
        return

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (frame_width, frame_height)

    # Trên Colab/Linux, codec 'mp4v' thường ổn định cho .mp4
    out = cv2.VideoWriter(args.output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    print("Đang xử lý video... (Vui lòng chờ)")
    frame_count = 0

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        # Preprocessing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image_tensor = [torch.from_numpy(image).to(device)]

        with torch.no_grad():
            output = model(image_tensor)[0]
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            for bbox, label, score in zip(boxes, labels, scores):
                if score > args.conf_threshold:
                    x_min, y_min, x_max, y_max = bbox

                    # Vẽ hình chữ nhật
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                    # Xử lý label an toàn (tránh lỗi index out of range)
                    label_idx = int(label)
                    if 0 <= label_idx < len(classes):
                        category = classes[label_idx]
                    else:
                        category = f"Class {label_idx}"

                    cv2.putText(frame, f"{category} {score:.2f}", (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Hoàn tất! Video đã được lưu tại: {args.output_name}")

# Chạy hàm test
test_on_colab(args)