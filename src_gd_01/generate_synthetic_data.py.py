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

START_INDEX = 1639      # Tên file bắt đầu (1639.jpg, 1639.txt)
NUM_IMAGES = 400        # Số lượng ảnh sinh ra

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
        
        # Tạo nhiễu cho vân gỗ
        noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (1, 51), 0)
        
        # Mở rộng kích thước chiều thành (h, w, 1) để khớp phép cộng với bg (h, w, 3)
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
# HÀM GHÉP TỜ PHIẾU VÀ TÍNH TỌA ĐỘ YOLO
# -------------------------------------------------------------------
def paste_paper_and_get_label(paper_img, bg_img):
    bg_h, bg_w = bg_img.shape[:2]
    p_h, p_w = paper_img.shape[:2]
    
    # 1. Resize ngẫu nhiên (chiếm 45% - 75% chiều rộng nền)
    target_w = int(bg_w * random.uniform(0.45, 0.75))
    scale = target_w / float(p_w)
    target_h = int(p_h * scale)
    paper_resized = cv2.resize(paper_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 2. Biến dạng góc nghiêng
    src_pts = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.float32)
    tilt = random.uniform(0.02, 0.12)
    dst_pts = np.array([
        [random.uniform(0, target_w * tilt), random.uniform(0, target_h * tilt)],
        [target_w - random.uniform(0, target_w * tilt), random.uniform(0, target_h * tilt)],
        [target_w - random.uniform(0, target_w * tilt), target_h - random.uniform(0, target_h * tilt)],
        [random.uniform(0, target_w * tilt), target_h - random.uniform(0, target_h * tilt)]
    ], dtype=np.float32)
    
    M_p = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(paper_resized, M_p, (target_w, target_h), borderValue=(0,0,0,0))
    
    # 3. Xoay ngẫu nhiên (-180 đến 180 độ)
    angle = random.uniform(-180, 180)
    M_r = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
    
    cos, sin = np.abs(M_r[0, 0]), np.abs(M_r[0, 1])
    new_w = int((target_h * sin) + (target_w * cos))
    new_h = int((target_h * cos) + (target_w * sin))
    M_r[0, 2] += (new_w / 2) - (target_w / 2)
    M_r[1, 2] += (new_h / 2) - (target_h / 2)
    
    rotated = cv2.warpAffine(warped, M_r, (new_w, new_h), borderValue=(0,0,0,0))
    
    # 4. Tính toán tọa độ các đỉnh sau khi biến đổi
    corners_p = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), M_p).reshape(-1, 2)
    corners_h = np.hstack([corners_p, np.ones((4, 1), dtype=np.float32)])
    corners_r = (M_r @ corners_h.T).T
    
    # 5. Chọn vị trí dán ngẫu nhiên
    tx = random.randint(0, max(0, bg_w - new_w))
    ty = random.randint(0, max(0, bg_h - new_h))
    abs_corners = corners_r + np.array([tx, ty])
    
    # 6. Trộn ảnh lên background
    result_bg = bg_img.copy()
    y1, y2 = ty, min(bg_h, ty + new_h)
    x1, x2 = tx, min(bg_w, tx + new_w)
    
    py1, py2 = 0, y2 - y1
    px1, px2 = 0, x2 - x1
    
    alpha = (rotated[py1:py2, px1:px2, 3] / 255.0)[:, :, np.newaxis]
    paper_rgb = rotated[py1:py2, px1:px2, :3]
    blended = (1 - alpha) * result_bg[y1:y2, x1:x2] + alpha * paper_rgb
    result_bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    
    # 7. Tạo nhãn YOLO
    min_x, max_x = max(0, np.min(abs_corners[:, 0])), min(bg_w, np.max(abs_corners[:, 0]))
    min_y, max_y = max(0, np.min(abs_corners[:, 1])), min(bg_h, np.max(abs_corners[:, 1]))
    
    xc = ((min_x + max_x) / 2.0) / bg_w
    yc = ((min_y + max_y) / 2.0) / bg_h
    nw = (max_x - min_x) / bg_w
    nh = (max_y - min_y) / bg_h
    
    yolo_line = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"
    return result_bg, yolo_line


# -------------------------------------------------------------------
# CHẠY CHƯƠNG TRÌNH
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(PAPER_PATH):
        print(f"❌ Không tìm thấy file 'paper.png' tại: {PAPER_PATH}")
    else:
        # Đọc file paper.png (đảm bảo có kênh Alpha)
        paper = cv2.imread(PAPER_PATH, cv2.IMREAD_UNCHANGED)
        if paper.shape[2] == 3:
            b, g, r = cv2.split(paper)
            a = np.ones(b.shape, dtype=b.dtype) * 255
            paper = cv2.merge((b, g, r, a))

        print(f"Bắt đầu sinh {NUM_IMAGES} ảnh và lưu thẳng vào thư mục dataset01...")

        for i in range(NUM_IMAGES):
            idx = START_INDEX + i
            
            bg = create_background(w=1200, h=1600)
            img_out, label = paste_paper_and_get_label(paper, bg)
            
            # Lưu trực tiếp vào dataset01
            img_path = os.path.join(DATASET_DIR, f"{idx}.jpg")
            txt_path = os.path.join(DATASET_DIR, f"{idx}.txt")
            
            cv2.imwrite(img_path, img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(txt_path, 'w') as f:
                f.write(label)

            if (i + 1) % 100 == 0 or (i + 1) == NUM_IMAGES:
                print(f"-> Đã sinh {i + 1}/{NUM_IMAGES} ảnh (File hiện tại: {idx}.jpg)")

        print(f"\n✅ ĐÃ HOÀN THÀNH!")
        print(f"Đã thêm {NUM_IMAGES} ảnh và file nhãn mới từ {START_INDEX}.jpg / {START_INDEX}.txt đến {START_INDEX + NUM_IMAGES - 1}.jpg / {START_INDEX + NUM_IMAGES - 1}.txt vào trực tiếp folder 'dataset01'.")