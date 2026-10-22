import os
import random
import shutil

# -------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN & TỶ LỆ CHIA DATASET
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset01")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset01-final")

# Tỷ lệ phân chia: 80% Train, 10% Validation, 10% Test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Xóa thư mục dataset01-final cũ (nếu có) để đảm bảo dữ liệu sạch
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Tạo lại cấu trúc thư mục YOLO chuẩn
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)


# -------------------------------------------------------------------
# 2. QUÉT VÀ LỌC CÁC CẶP ẢNH (.JPG) VÀ NHÃN (.TXT) HỢP LỆ
# -------------------------------------------------------------------
print("=== BẮT ĐẦU QUÉT THƯ MỤC DATASET01 ===")

valid_pairs = []
all_files = os.listdir(SOURCE_DIR)

for filename in all_files:
    if filename.lower().endswith('.jpg'):
        base_name = os.path.splitext(filename)[0]
        txt_name = f"{base_name}.txt"
        
        img_path = os.path.join(SOURCE_DIR, filename)
        txt_path = os.path.join(SOURCE_DIR, txt_name)
        
        # Chỉ lấy những ảnh có đi kèm file nhãn .txt
        if os.path.exists(txt_path):
            valid_pairs.append((filename, txt_name))
        else:
            print(f"⚠️ Bỏ qua {filename} (Không tìm thấy file nhãn {txt_name})")

# Sắp xếp cố định theo tên số và trộn ngẫu nhiên
valid_pairs.sort(key=lambda x: int(os.path.splitext(x[0])[0]) if os.path.splitext(x[0])[0].isdigit() else x[0])
random.seed(42)
random.shuffle(valid_pairs)


# -------------------------------------------------------------------
# 3. CHIA TẬP DỮ LIỆU & COPY SANG DATASET01-FINAL
# -------------------------------------------------------------------
total_count = len(valid_pairs)
train_end = int(total_count * TRAIN_RATIO)
val_end = train_end + int(total_count * VAL_RATIO)

train_pairs = valid_pairs[:train_end]
val_pairs = valid_pairs[train_end:val_end]
test_pairs = valid_pairs[val_end:]

def copy_data(pair_list, split_type):
    for img_name, txt_name in pair_list:
        src_img = os.path.join(SOURCE_DIR, img_name)
        src_txt = os.path.join(SOURCE_DIR, txt_name)
        
        dst_img = os.path.join(OUTPUT_DIR, 'images', split_type, img_name)
        dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_type, txt_name)
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_txt, dst_txt)
        
    print(f"-> Tập '{split_type}': Hoàn thành {len(pair_list)} cặp ảnh & nhãn.")

print("\n--- Tiến hành phân chia ---")
copy_data(train_pairs, 'train')
copy_data(val_pairs, 'val')
copy_data(test_pairs, 'test')


# -------------------------------------------------------------------
# 4. TẠO FILE DATA.YAML DÙNG CHO PHẦN TRAIN YOLO
# -------------------------------------------------------------------
yaml_content = """path: /content/dataset01-final
train: images/train
val: images/val
test: images/test

names:
  0: answer_sheet
"""

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"\n✅ ĐÃ HOÀN THÀNH!")
print(f"- Tổng số cặp ảnh/nhãn hợp lệ: {total_count}")
print(f"- Thư mục đầu ra: {OUTPUT_DIR}")
print(f"- Đã tạo cấu hình YOLO: {yaml_path}")