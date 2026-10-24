import os

# -------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

# Danh sách 7 thư mục nhãn cần đánh lại số
SUBFOLDERS = ['empty_a', 'empty_b', 'empty_c', 'empty_d', 'empty_e', 'filled', 'crossed']


# -------------------------------------------------------------------
# 2. CHƯƠNG TRÌNH ĐÁNH SỐ TỰ ĐỘNG
# -------------------------------------------------------------------
def prepare_and_rename_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Không tìm thấy thư mục dataset tại: {TRAIN_DIR}")
        return

    print("=== BẮT ĐẦU CHUẨN HÓA VÀ ĐÁNH SỐ LẠI DATASET ===\n")
    total_renamed = 0

    for folder in SUBFOLDERS:
        folder_path = os.path.join(TRAIN_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Thư mục '{folder}' chưa tồn tại, bỏ qua.")
            continue

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        files.sort()  # Sắp xếp tên file cũ để đánh số theo thứ tự

        if not files:
            print(f"📁 [{folder}]: Không có ảnh nào.")
            continue

        # Bước 1: Đổi sang tên tạm thời để tránh trùng lặp/xung đột tên file
        temp_list = []
        for idx, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1].lower()
            old_path = os.path.join(folder_path, filename)
            temp_name = f"_temp_rename_{idx}{ext}"
            temp_path = os.path.join(folder_path, temp_name)
            
            os.rename(old_path, temp_path)
            temp_list.append((temp_path, ext))

        # Bước 2: Đổi tên chính thức theo định dạng chuẩn hóa (VD: filled_0001.jpg)
        for idx, (temp_path, ext) in enumerate(temp_list, start=1):
            new_filename = f"{folder}_{idx:04d}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(temp_path, new_path)

        count = len(files)
        total_renamed += count
        first_file = f"{folder}_0001{temp_list[0][1]}"
        last_file = f"{folder}_{count:04d}{temp_list[-1][1]}"
        print(f"✅ [{folder}]: Đã đánh số {count} ảnh ({first_file}  ➔  {last_file})")

    print(f"\n🎉 HOÀN THÀNH! Tổng cộng đã chuẩn hóa {total_renamed} ảnh trong {TRAIN_DIR}")


if __name__ == "__main__":
    prepare_and_rename_dataset()