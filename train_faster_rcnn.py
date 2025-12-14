from voc_dataset import VOCDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import argparse
import os
import shutil
import numpy as np
from tqdm.autonotebook import tqdm

def get_args():
    parser = argparse.ArgumentParser(description= "Train model")
    parser.add_argument("--num_epochs", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--data_path", "-d", type = str,
                        default=r"D:\deep_learning_object_detection\pascal_voc_2012\VOC2012_train_val\VOC2012_train_val")
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--momentum", type = float, default = 0.9)
    parser.add_argument("--log_path", type = str, default = "tensorboard")
    parser.add_argument("--check_point", type = str, default = "trained_model")
    parser.add_argument("--save_checkpoint", type=str, default="trained_model/last.pt")
    args = parser.parse_args()
    return args


def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train(args):
    num_epochs = args.num_epochs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = Compose([
        RandomAffine(
            degrees=5,
            translate = (0.15, 0.15),
            scale = (0.85, 1.15),
            shear = 10,
        ),
        ColorJitter(
            brightness = 0.125,
            contrast = 0.5,
            saturation = 0.5,
            hue = 0.05,
        ),
        ToTensor(),
        # Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    test_transform = Compose([
        ToTensor(),
        # Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    train_dataset = VOCDataset(root_dir = args.data_path, split = "train",
                         transform = transform)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 2,
        collate_fn = collate_fn
    )

    val_dataset = VOCDataset(root_dir = args.data_path, split = "val",
                         transform = test_transform)
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 2,
        collate_fn = collate_fn
    )

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers = 4)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(train_dataset.classes))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    if args.save_checkpoint and os.path.isfile(args.save_checkpoint):
        print(f"Loading checkpoint from {args.save_checkpoint}...")
        checkpoint = torch.load(args.save_checkpoint, map_location = "cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_map = checkpoint["best_map"]
    else:
        start_epoch = 0
        best_map = -1

    model.to(device)

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.check_point):
        os.makedirs(args.check_point)

    writer = SummaryWriter(args.log_path)
    best_map = -1
    for epoch in range(start_epoch, num_epochs):

        # TRAINING PHASE
        model.train()
        progress_bar = tqdm(train_dataloader, colour = "cyan")
        num_iters = len(train_dataloader)
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]
            # forward
            loss = model(images, labels)
            final_loss = sum([loss_value for loss_value in loss.values()])

            # backward
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            train_loss.append(final_loss.item())
            mean_loss = np.mean(train_loss)

            progress_bar.set_description("epoch {}/{}. loss: {:0.4f}".format(epoch + 1, num_epochs, mean_loss))
            writer.add_scalar("Train/loss", mean_loss, epoch * num_iters + iter)

        # VALIDATION PHASE
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="cyan")
        metric = MeanAveragePrecision(iou_type="bbox")
        val_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu"),
                })
            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"].to("cpu"),
                    "labels": label["labels"].to("cpu"),
                })
            metric.update(preds, targets)

        result = metric.compute()
        print(result)
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)

        check_point = {
            "model_state_dict": model.state_dict(),
            "map": result["map"],
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(check_point, os.path.join(args.check_point, "last.pt"))
        if result["map"] > best_map:
            best_map = result["map"]
            torch.save(model.state_dict(), os.path.join(args.check_point, "best.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)