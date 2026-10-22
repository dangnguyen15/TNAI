import os
import shutil
from PIL import Image, ImageOps
import pillow_heif

# Register plugin đọc file HEIC/HEIF cho Pillow
pillow_heif.register_heif_opener()

# 1. Thiết lập đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset01")
TEMP_DIR = os.path.join(PROJECT_ROOT, "dataset01_temp")

os.makedirs(TEMP_DIR, exist_ok=True)

# 2. Lấy danh sách toàn bộ ảnh (bao gồm HEIC, PNG, JPG, WEBP...)
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif')

all_files = [
    f for f in os.listdir(SOURCE_DIR) 
    if f.lower().endswith(VALID_EXTENSIONS)
]
all_files.sort()

print(f"=== BẮT ĐẦU CHUYỂN ĐỔI VÀ ĐỔI TÊN {len(all_files)} ANH TRONG DATASET01 ===")

success_count = 0

for idx, filename in enumerate(all_files):
    old_path = os.path.join(SOURCE_DIR, filename)
    
    try:
        # Mở ảnh bằng Pillow (hỗ trợ mọi định dạng bao gồm HEIC)
        with Image.open(old_path) as img:
            # Tự động xoay ảnh đúng chiều theo thông tin EXIF của điện thoại
            img = ImageOps.exif_transpose(img)
            
            # Chuyển đổi sang hệ màu RGB (tránh lỗi khi lưu JPG từ ảnh RGBA/HEIC)
            rgb_img = img.convert('RGB')
            
            # Đường dẫn ảnh mới: 0.jpg, 1.jpg, 2.jpg...
            new_filename = f"{idx}.jpg"
            new_path = os.path.join(TEMP_DIR, new_filename)
            
            # Lưu lại dạng JPG chất lượng cao
            rgb_img.save(new_path, 'JPEG', quality=95)
            success_count += 1
            
    except Exception as e:
        print(f"⚠️ Lỗi xử lý file {filename}: {e}")

# 3. Thay thế folder dataset01 cũ bằng dataset01 mới đã chuẩn hóa
shutil.rmtree(SOURCE_DIR)
os.rename(TEMP_DIR, SOURCE_DIR)

print(f"\n✅ HOÀN THÀNH!")
print(f"- Đã chuyển đổi thành công {success_count}/{len(all_files)} ảnh về dạng `.jpg`.")
print(f"- Toàn bộ ảnh đã được đổi tên từ `0.jpg` đến `{success_count - 1}.jpg`.")