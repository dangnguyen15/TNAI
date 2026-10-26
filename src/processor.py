import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    return rect


def scan_and_flatten_paper(image_path_or_img, target_w=1200, target_h=1600):
    """
    TẦNG 1: NẮN PHẲNG TOÀN BỘ TỜ GIẤY BÀI THI (1200x1600 PX)
    """
    if isinstance(image_path_or_img, str):
        image = cv2.imread(image_path_or_img)
    else:
        image = image_path_or_img

    if image is None:
        raise ValueError("Không thể mở file ảnh đầu vào!")

    h_img, w_img = image.shape[:2]
    total_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < (total_area * 0.20):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            paper_pts = approx.reshape(4, 2)
            break
        elif len(approx) > 4:
            rect_rot = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect_rot)
            paper_pts = np.int32(box)
            break

    if paper_pts is None:
        paper_pts = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype="float32")

    rect = order_points(paper_pts)
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    scanned_paper = cv2.warpPerspective(image, M, (target_w, target_h))
    return scanned_paper


def extract_exact_table_roi(scanned_img, target_w=1000, target_h=600):
    """
    TẦNG 2: CẮT RIÊNG KHUNG BẢNG BÀI LÀM (1000x600 PX)
    """
    h, w = scanned_img.shape[:2]
    
    y1_rough = int(h * 0.52)
    y2_rough = int(h * 0.96)
    x1_rough = int(w * 0.05)
    x2_rough = int(w * 0.95)

    rough_crop = scanned_img[y1_rough:y2_rough, x1_rough:x2_rough]
    crop_h, crop_w = rough_crop.shape[:2]

    gray = cv2.cvtColor(rough_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(crop_w * 0.3), 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    row_sums = np.sum(horizontal_lines, axis=1)
    valid_rows = np.where(row_sums > (crop_w * 255 * 0.15))[0]

    if len(valid_rows) >= 2:
        y1_table = valid_rows[0]
        y2_table = valid_rows[-1]
        table_crop = rough_crop[y1_table:y2_table, 0:crop_w]
    else:
        table_crop = rough_crop

    table_roi = cv2.resize(table_crop, (target_w, target_h))
    return table_roi


def extract_and_draw_300_bubbles(table_roi, target_size=(32, 32)):
    """
    CHIA LƯỚI TOẠ ĐỘ CHUẨN XÁC CHỦ DỘNG TRÊN BẢNG 1000x600 PX
    """
    roi_h, roi_w = table_roi.shape[:2]
    overlay_img = table_roi.copy()

    options = ['a', 'b', 'c', 'd', 'e']
    bubbles_list = []

    # Bảng 1000px được chia đều làm 4 cột nhóm (mỗi nhóm rộng 250px)
    group_w = roi_w / 4.0        # 250.0 px
    row_h = roi_h / 15.0         # 40.0 px

    # Trong 250px của mỗi nhóm:
    # - 54px đầu tiên là dành riêng cho phần số thứ tự câu (vd: 1), 16), 31)...)
    # - 188px tiếp theo chia đều cho 5 ô đáp án (mỗi ô rộng ~37.6px)
    # - 8px cuối cùng là khoảng trống chừa vạch kẻ đứng phân cách nhóm
    num_margin = int(group_w * 0.22)               # 55 px
    opt_area_w = group_w - num_margin - 8.0         # 187.0 px
    single_opt_w = opt_area_w / 5.0                # 37.4 px

    for col in range(4):
        group_x1 = int(col * group_w)
        opt_start_x = group_x1 + num_margin

        for row in range(15):
            q_num = col * 15 + row + 1
            by1 = int(row * row_h)
            by2 = int((row + 1) * row_h)

            for opt_idx, opt_char in enumerate(options):
                bx1 = int(opt_start_x + opt_idx * single_opt_w)
                bx2 = int(opt_start_x + (opt_idx + 1) * single_opt_w)

                crop_patch = table_roi[by1:by2, bx1:bx2]
                bubble_resized = cv2.resize(crop_patch, target_size, interpolation=cv2.INTER_AREA)

                bubbles_list.append({
                    'question': q_num,
                    'option': opt_char,
                    'col': col,
                    'row': row,
                    'opt_idx': opt_idx,
                    'image': bubble_resized,
                    'bbox': (bx1, by1, bx2, by2)
                })

                # Vẽ khung màu xanh lá đè trực tiếp lên vị trí ô tròn
                cv2.rectangle(
                    overlay_img, 
                    (bx1 + 1, by1 + 1), 
                    (bx2 - 1, by2 - 1), 
                    (0, 255, 0), 
                    1
                )

    return bubbles_list, overlay_img