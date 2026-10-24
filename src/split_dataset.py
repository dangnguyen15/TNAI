import os
import shutil
import random

# Fix seed để kết quả phân chia ngẫu nhiên có thể tái lập lại (reproducible)
random.seed(42)

# -------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
SPLIT_ROOT_DIR = os.path.join(PROJECT_ROOT, "dataset", "dataset_split")

SUBFOLDERS = ['empty_a', 'empty_b', 'empty_c', 'empty_d', 'empty_e', 'filled', 'crossed']
NUM_VAL_PER_CLASS = 30
NUM_TEST_PER_CLASS = 20


# -------------------------------------------------------------------
# 2. CHƯƠNG TRÌNH PHÂN CHIA DATASET
# -------------------------------------------------------------------
def split_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Không tìm thấy thư mục dữ liệu nguồn tại: {SOURCE_DIR}")
        return

    print("=== BẮT ĐẦU PHÂN CHIA DATASET (VAL: 30, TEST: 20) ===\n")

    # Tạo cấu trúc thư mục train, val, test trong dataset_split/
    for split_type in ['train', 'val', 'test']:
        for folder in SUBFOLDERS:
            os.makedirs(os.path.join(SPLIT_ROOT_DIR, split_type, folder), exist_ok=True)

    total_train_count = 0
    total_val_count = 0
    total_test_count = 0

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for folder in SUBFOLDERS:
        src_folder_path = os.path.join(SOURCE_DIR, folder)
        if not os.path.exists(src_folder_path):
            print(f"⚠️ Không tìm thấy thư mục '{folder}', bỏ qua.")
            continue

        files = [f for f in os.listdir(src_folder_path) if f.lower().endswith(valid_exts)]
        
        # Bắt buộc mỗi thư mục phải có ít nhất 100 ảnh (50 test + 50 val)
        required_min = NUM_VAL_PER_CLASS + NUM_TEST_PER_CLASS
        if len(files) < required_min:
            print(f"❌ Thư mục [{folder}] chỉ có {len(files)} ảnh (Yêu cầu tối thiểu {required_min} ảnh để chia val/test).")
            continue

        # Xáo trộn danh sách ảnh ngẫu nhiên
        random.shuffle(files)

        # Phân chia danh sách file
        test_files = files[:NUM_TEST_PER_CLASS]
        val_files = files[NUM_TEST_PER_CLASS : NUM_TEST_PER_CLASS + NUM_VAL_PER_CLASS]
        train_files = files[NUM_TEST_PER_CLASS + NUM_VAL_PER_CLASS:]

        # 1. Copy sang tập TEST
        for f in test_files:
            shutil.copy2(os.path.join(src_folder_path, f), os.path.join(SPLIT_ROOT_DIR, "test", folder, f))

        # 2. Copy sang tập VAL
        for f in val_files:
            shutil.copy2(os.path.join(src_folder_path, f), os.path.join(SPLIT_ROOT_DIR, "val", folder, f))

        # 3. Copy phần còn lại sang tập TRAIN
        for f in train_files:
            shutil.copy2(os.path.join(src_folder_path, f), os.path.join(SPLIT_ROOT_DIR, "train", folder, f))

        total_test_count += len(test_files)
        total_val_count += len(val_files)
        total_train_count += len(train_files)

        print(f"✅ [{folder.upper()}]: Tổng {len(files)} ảnh ➔ Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print("\n" + "="*50)
    print(f"🎉 HOÀN THÀNH PHÂN CHIA DATASET!")
    print(f"📂 Tập TRAIN tổng cộng : {total_train_count} ảnh")
    print(f"📂 Tập VAL   tổng cộng : {total_val_count} ảnh (30 ảnh / nhãn)")
    print(f"📂 Tập TEST  tổng cộng : {total_test_count} ảnh (20 ảnh / nhãn)")
    print(f"📍 Lưu trữ tại thư mục : {SPLIT_ROOT_DIR}")


if __name__ == "__main__":
    split_dataset()