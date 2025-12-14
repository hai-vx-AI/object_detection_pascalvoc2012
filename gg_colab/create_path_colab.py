import os
from google.colab import drive

# 1. Kết nối Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Đường dẫn file trên Drive (Dựa theo ảnh bạn gửi)
zip_path = '/content/drive/MyDrive/pascal_voc_2012.zip'
extract_path = '/content/my_dataset'

# 3. Thực hiện giải nén
if os.path.exists(zip_path):
    print(f"✅ Tìm thấy file: {zip_path}")
    print("⏳ Đang giải nén... (Vui lòng đợi 1-2 phút)")

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # Lệnh giải nén (dùng unzip vì file của bạn đuôi .zip)
    !unzip -q "$zip_path" -d "$extract_path"

    print("✅ Giải nén hoàn tất!")
else:
    print(f"❌ Không tìm thấy file tại {zip_path}. Bạn hãy kiểm tra lại tên file trên Drive xem có chính xác từng ký tự không.")