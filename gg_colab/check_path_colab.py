import os

# Tìm kiếm file train.txt bắt đầu từ thư mục content
print("Đang quét tìm file train.txt...")
found = False
for root, dirs, files in os.walk("/content"):
    if "train.txt" in files and "ImageSets" in root:
        # Tìm thấy thư mục chứa ImageSets, đây chính là root_dir cần tìm
        # Chúng ta cần lấy thư mục cha của 'ImageSets'
        actual_data_path = os.path.dirname(os.path.dirname(root))
        print(f"\n✅ ĐÃ TÌM THẤY! Đường dẫn chuẩn của bạn là:\n--> {actual_data_path}")
        found = True
        break

if not found:
    print("\n❌ Vẫn chưa tìm thấy file. Bạn hãy kiểm tra lại bước giải nén xem folder có rỗng không.")