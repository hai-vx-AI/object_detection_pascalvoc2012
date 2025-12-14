import torch
import gc

# Xóa cache của Python
gc.collect()

# Xóa cache của PyTorch trên GPU
torch.cuda.empty_cache()

print("Đã giải phóng bộ nhớ GPU!")
import torch.cuda.amp as amp
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.autonotebook import tqdm
from voc_dataset import VOCDataset

# --- CẤU HÌNH ---
class Args:
    num_epochs = 5
    batch_size = 8
    data_path = "/content/my_dataset/VOC2012_train_val/VOC2012_train_val"
    learning_rate = 0.005
    momentum = 0.9
    log_path = "tensorboard_logs"
    check_point = "/content/drive/MyDrive/VOC_Training"
    save_checkpoint = "/content/drive/MyDrive/VOC_Training/last.pt"

args = Args()

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on: {device}")

    # 1. Transform tối giản
    transform = Compose([ToTensor()])
    test_transform = Compose([ToTensor()])

    # 2. Dataset & DataLoader
    train_dataset = VOCDataset(root_dir=args.data_path, split="train", transform=transform)
    val_dataset = VOCDataset(root_dir=args.data_path, split="val", transform=test_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. Model (Layer=1)
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=1
    )
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(train_dataset.classes))
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    scaler = amp.GradScaler()

    # Load Checkpoint
    start_epoch = 0
    best_map = -1
    if os.path.isfile(args.save_checkpoint):
        print(f"Loading checkpoint from {args.save_checkpoint}...")
        checkpoint = torch.load(args.save_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_map = checkpoint.get("map", 0.0)

    # Tensorboard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path, exist_ok=True)
    writer = SummaryWriter(args.log_path)

    # Training Loop
    for epoch in range(start_epoch, args.num_epochs):
        # --- TRAIN ---
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan", desc=f"Epoch {epoch+1}/{args.num_epochs}")
        train_loss = []

        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]

            optimizer.zero_grad()

            # Fix cú pháp autocast mới để hết báo warning đỏ
            with torch.amp.autocast('cuda', enabled=True):
                loss_dict = model(images, labels)
                final_loss = sum(loss for loss in loss_dict.values())

            scaler.scale(final_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(final_loss.item())
            progress_bar.set_postfix({"loss": np.mean(train_loss)})
            writer.add_scalar("Train/step_loss", final_loss.item(), epoch * len(train_dataloader) + iter)

        mean_loss = np.mean(train_loss)
        writer.add_scalar("Train/epoch_loss", mean_loss, epoch)

        # --- VALIDATION ---
        print("Running Validation...")
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, colour="green", desc="Validating"):
                images = [image.to(device) for image in images]

                with torch.amp.autocast('cuda', enabled=True):
                    outputs = model(images)

                preds = [{k: v.cpu() for k, v in t.items()} for t in outputs]

                # --- SỬA LỖI Ở ĐÂY ---
                # Thay k bằng "boxes"
                targets = [{"boxes": v["boxes"].cpu(), "labels": v["labels"].cpu()} for v in labels]

                metric.update(preds, targets)

        result = metric.compute()
        map_value = result["map"].item()
        print(f"Validation mAP: {map_value:.4f}")

        writer.add_scalar("Val/mAP", map_value, epoch)

        # Save checkpoint
        check_point_data = {
            "model_state_dict": model.state_dict(),
            "map": map_value,
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict()
        }
        torch.save(check_point_data, args.save_checkpoint)

        if map_value > best_map:
            best_map = map_value
            torch.save(model.state_dict(), os.path.join(args.check_point, "best.pt"))
            print(f"Saved BEST model (mAP: {best_map:.4f})")

    writer.close()

if __name__ == '__main__':
    train(args)