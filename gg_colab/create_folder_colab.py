# Mount Google Drive để lưu model (Optional - nhưng khuyên dùng)
from google.colab import drive
drive.mount('/content/drive')

# Tạo folder lưu model trên Drive nếu chưa có
import os
if not os.path.exists('/content/drive/MyDrive/VOC_Training'):
    os.makedirs('/content/drive/MyDrive/VOC_Training')