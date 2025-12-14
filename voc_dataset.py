import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, RandomAffine, ColorJitter
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Đường dẫn tới folder chứa 'JPEGImages' và 'Annotations'.
                               (Ví dụ: .../VOC2012_train_val/VOC2012)
            split (string): 'train', 'val', hoặc 'trainval'.
            transform (callable, optional): Transform ảnh.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Định nghĩa đường dẫn con
        self.annotation_dir = os.path.join(self.root_dir, 'Annotations')
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')

        # Đường dẫn file txt chứa danh sách ảnh (train.txt / val.txt)
        # Cấu trúc Kaggle thường là: root_dir/ImageSets/Main/train.txt
        split_file_path = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{self.split}.txt')

        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Không tìm thấy file danh sách ảnh tại: {split_file_path}")

        with open(split_file_path, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        # PASCAL VOC Classes
        self.classes = (
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        # return len(self.file_names)
        return 10

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # 1. Load Ảnh
        img_path = os.path.join(self.image_dir, f"{file_name}.jpg")
        image = Image.open(img_path).convert("RGB")

        # 2. Load XML Annotation
        xml_path = os.path.join(self.annotation_dir, f"{file_name}.xml")
        boxes, labels = self._parse_xml(xml_path)

        # Convert sang Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # 3. Transform
        if self.transform:
            image = self.transform(image)

        return image, target

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_idx:
                continue

            label = self.class_to_idx[name]
            bndbox = obj.find("bndbox")

            # PASCAL VOC là định dạng [xmin, ymin, xmax, ymax]
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels


# Hàm Collate (BẮT BUỘC cho Object Detection)
# def collate_fn(batch):
#     return tuple(zip(*batch))
#
if __name__ == '__main__':
    transform = Compose([
        RandomAffine(
            degrees=5,
            translate=(0.15, 0.15),
            # scale=(0.85, 1.15),
            shear=10,
        ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05,
        ),
    ])
    dataset = VOCDataset(root_dir = r"D:\deep_learning_object_detection\pascal_voc_2012\VOC2012_train_val\VOC2012_train_val",
                         transform = transform)
    image, target = dataset[2345]
    image.show()

# from torchvision.datasets import VOCDataset, Dataset
# import torch
#
# class VOCDataset(Dataset):
#     def __init__(self, root, year, image_set, transform=None):
#         super().__init__(root, year, image_set, transform)
#         self.classes = (
#                     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#                     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#                     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
#                 )
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#
#     def __getitem__(self, item):
#         image, data = super().__getitem__(item)
#         all_bboxes = []
#         all_labels = []
#         for obj in data["annotation"]["object"]:
#             x_min = int(obj["bndbox"]["xmin"])
#             y_min = int(obj["bndbox"]["ymin"])
#             x_max = int(obj["bndbox"]["xmax"])
#             y_max = int(obj["bndbox"]["ymax"])
#             all_bboxes.append([x_min, y_min, x_max, y_max])
#             all_labels.append(self.class_to_idx[obj["name"]])
#         all_boxes = torch.FloatTensor(all_bboxes)
#         all_labels = torch.LongTensor(all_labels)
#         target = {
#             "boxes": all_boxes,
#             "labels": all_labels,
#         }
#         return image, target