import os
import cv2
import numpy as np
import random

# -------------------------------------------------------------------
# THIẾT LẬP ĐƯỜNG DẪN & THAM SỐ
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PAPER_PATH = os.path.join(PROJECT_ROOT, "paper.png")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset01")

START_INDEX = 2839      # Tiếp nối sau 700 ảnh nghiêng (1639 -> 2338)
NUM_IMAGES = 300       # Số lượng ảnh thẳng cần tạo thêm

os.makedirs(DATASET_DIR, exist_ok=True)


# -------------------------------------------------------------------
# HÀM TẠO BACKGROUND NGẪU NHIÊN
# -------------------------------------------------------------------
def create_background(w=1200, h=1600):
    bg_type = random.choice(['wood', 'tile', 'texture', 'gradient'])
    
    if bg_type == 'wood':
        bg = np.zeros((h, w, 3), dtype=np.float32)
        base = np.array([random.randint(20, 80), random.randint(50, 120), random.randint(90, 180)], dtype=np.float32)
        bg[:, :] = base
        
        noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (1, 51), 0)
        noise = np.expand_dims(noise, axis=-1)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
    elif bg_type == 'tile':
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :] = [random.randint(150, 230)] * 3
        grid_color = [random.randint(50, 110)] * 3
        step = random.randint(150, 250)
        for y in range(0, h, step):
            cv2.line(bg, (0, y), (w, y), grid_color, random.randint(2, 4))
        for x in range(0, w, step):
            cv2.line(bg, (x, 0), (x, h), grid_color, random.randint(2, 4))
            
    elif bg_type == 'gradient':
        c1 = np.array([random.randint(30, 180), random.randint(30, 180), random.randint(30, 180)], dtype=np.float32)
        c2 = np.array([random.randint(30, 180), random.randint(30, 180), random.randint(30, 180)], dtype=np.float32)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            alpha = y / float(h)
            bg[y, :] = ((1 - alpha) * c1 + alpha * c2).astype(np.uint8)
            
    else: # texture
        bg = np.random.randint(50, 210, (h, w, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (35, 35), 0)

    return bg


# -------------------------------------------------------------------
# HÀM GHÉP TỜ PHIẾU THẲNG GÓC & CÂN XỨNG
# -------------------------------------------------------------------
def paste_straight_paper_and_get_label(paper_img, bg_img):
    bg_h, bg_w = bg_img.shape[:2]
    p_h, p_w = paper_img.shape[:2]
    
    # 1. Resize giữ nguyên tỷ lệ A4 (chiếm từ 55% đến 75% chiều rộng nền)
    target_w = int(bg_w * random.uniform(0.55, 0.75))
    scale = target_w / float(p_w)
    target_h = int(p_h * scale)
    paper_resized = cv2.resize(paper_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 2. Đặt vị trí THẲNG & CÂN XỨNG (Căn giữa kèm xê dịch nhẹ 1-3% để đa dạng)
    center_x = (bg_w - target_w) // 2
    center_y = (bg_h - target_h) // 2
    
    offset_x = random.randint(-int(bg_w * 0.03), int(bg_w * 0.03))
    offset_y = random.randint(-int(bg_h * 0.03), int(bg_h * 0.03))
    
    tx = max(0, min(bg_w - target_w, center_x + offset_x))
    ty = max(0, min(bg_h - target_h, center_y + offset_y))
    
    # 3. Trộn ảnh lên background
    result_bg = bg_img.copy()
    y1, y2 = ty, ty + target_h
    x1, x2 = tx, tx + target_w
    
    if paper_resized.shape[2] == 4:
        alpha = (paper_resized[:, :, 3] / 255.0)[:, :, np.newaxis]
        paper_rgb = paper_resized[:, :, :3]
        blended = (1 - alpha) * result_bg[y1:y2, x1:x2] + alpha * paper_rgb
        result_bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        result_bg[y1:y2, x1:x2] = paper_resized
        
    # 4. Tính toán nhãn YOLO chuẩn tuyệt đối (vì khung thẳng)
    xc = ((tx + target_w / 2.0)) / bg_w
    yc = ((ty + target_h / 2.0)) / bg_h
    nw = target_w / float(bg_w)
    nh = target_h / float(bg_h)
    
    yolo_line = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
    return result_bg, yolo_line


# -------------------------------------------------------------------
# CHẠY CHƯƠNG TRÌNH
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(PAPER_PATH):
        print(f"❌ Không tìm thấy file 'paper.png' tại: {PAPER_PATH}")
    else:
        paper = cv2.imread(PAPER_PATH, cv2.IMREAD_UNCHANGED)
        if paper.shape[2] == 3:
            b, g, r = cv2.split(paper)
            a = np.ones(b.shape, dtype=b.dtype) * 255
            paper = cv2.merge((b, g, r, a))

        print(f"Bắt đầu sinh {NUM_IMAGES} ảnh THẲNG GÓC và lưu vào thư mục dataset01...")

        for i in range(NUM_IMAGES):
            idx = START_INDEX + i
            
            bg = create_background(w=1200, h=1600)
            img_out, label = paste_straight_paper_and_get_label(paper, bg)
            
            img_path = os.path.join(DATASET_DIR, f"{idx}.jpg")
            txt_path = os.path.join(DATASET_DIR, f"{idx}.txt")
            
            cv2.imwrite(img_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(txt_path, 'w') as f:
                f.write(label)

            if (i + 1) % 100 == 0 or (i + 1) == NUM_IMAGES:
                print(f"-> Đã sinh {i + 1}/{NUM_IMAGES} ảnh (File hiện tại: {idx}.jpg)")

        print(f"\n✅ ĐÃ HOÀN THÀNH!")
        print(f"Đã thêm {NUM_IMAGES} ảnh THẲNG GÓC mới từ {START_INDEX}.jpg / {START_INDEX}.txt đến {START_INDEX + NUM_IMAGES - 1}.jpg / {START_INDEX + NUM_IMAGES - 1}.txt vào trực tiếp folder 'dataset01'.")