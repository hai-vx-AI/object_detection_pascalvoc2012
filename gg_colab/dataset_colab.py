%%writefile voc_dataset.py
import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.annotation_dir = os.path.join(self.root_dir, 'Annotations')
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')

        # Sửa đường dẫn cho đúng cấu trúc giải nén trên Colab
        split_file_path = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{self.split}.txt')

        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Không tìm thấy file danh sách ảnh tại: {split_file_path}")

        with open(split_file_path, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.classes = (
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor", "background"
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        # --- THAY ĐỔI Ở ĐÂY ---
        # Trả về toàn bộ dữ liệu thay vì 10
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, f"{file_name}.jpg")

        # Xử lý trường hợp ảnh không load được (optional but recommended)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
             print(f"Lỗi load ảnh: {img_path}")
             return self.__getitem__((idx + 1) % len(self))

        xml_path = os.path.join(self.annotation_dir, f"{file_name}.xml")
        boxes, labels = self._parse_xml(xml_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

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

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels