import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. CÁC HÀM SẴN CÓ CỦA BẠN (ORDER_POINTS, SCAN, EXTRACT)
# -------------------------------------------------------------------
def order_points(pts):
    """
    Sắp xếp 4 tọa độ góc theo đúng thứ tự: 
    [0]: Top-Left (Trên-Trái), [1]: Top-Right (Trên-Phải), 
    [2]: Bottom-Right (Dưới-Phải), [3]: Bottom-Left (Dưới-Trái)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sắp xếp theo trục X (Trái / Phải)
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Sắp xếp 2 điểm bên trái theo trục Y (Trên / Dưới)
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Sắp xếp 2 điểm bên phải theo trục Y (Trên / Dưới)
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    return rect


def scan_and_flatten_paper(image_path_or_img, target_w=1200, target_h=1600):
    if isinstance(image_path_or_img, str):
        image = cv2.imread(image_path_or_img)
    else:
        image = image_path_or_img

    if image is None:
        raise ValueError("Không thể mở file ảnh đầu vào!")

    h_img, w_img = image.shape[:2]
    total_area = h_img * w_img

    # 1. Tiền xử lý lọc nhiễu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Dùng Canny để bắt viền mép giấy sắc nét hơn
    edged = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 3. Tìm Contour khung giấy lớn nhất
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < (total_area * 0.25): # Yêu cầu viền giấy chiếm ít nhất 25% diện tích
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            paper_pts = approx.reshape(4, 2)
            break

    # Dự phòng: Nếu không bắt được góc giấy rõ ràng, lấy toàn bộ khung ảnh gốc
    if paper_pts is None:
        paper_pts = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype="float32")

    # 4. Nắn phẳng perspective
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

def extract_answer_region(scanned_img):
    h, w = scanned_img.shape[:2]
    
    # 🛠️ TINH CHỈNH TỌA ĐỘ LỀ NGANGB
    # roi_x1: Đẩy sang phải 0.090 để cắt sát vào phần chữ số 1)
    roi_x1 = int(w * 0.090)
    
    # roi_x2: Thu gọn về 0.935 để bỏ khoảng trắng lề ngoài bên phải
    roi_x2 = int(w * 0.935)

    # Giữ nguyên khung chiều cao rất chuẩn
    roi_y1 = int(h * 0.538)  
    roi_y2 = int(h * 0.955)  

    answer_roi = scanned_img[roi_y1:roi_y2, roi_x1:roi_x2]
    return answer_roi


def crop_300_bubbles(answer_roi, target_size=(32, 32)):
    h, w = answer_roi.shape[:2]
    col_w = w / 4.0
    row_h = h / 15.0

    options = ['a', 'b', 'c', 'd', 'e']
    bubbles_list = []

    for col in range(4):
        for row in range(15):
            q_num = col * 15 + row + 1
            
            x1_col = int(col * col_w)
            x2_col = int((col + 1) * col_w)
            y1_row = int(row * row_h)
            y2_row = int((row + 1) * row_h)

            cell_roi = answer_roi[y1_row:y2_row, x1_col:x2_col]
            cell_h, cell_w = cell_roi.shape[:2]

            # 🛠️ ĐIỀU CHỈNH CHÍNH: Tăng lên 25% để vượt qua số thứ tự 1), 16), 31), 46)
            opt_start_x = int(cell_w * 0.25)
            opt_total_w = cell_w - opt_start_x
            single_opt_w = opt_total_w / 5.0

            for opt_idx, opt_char in enumerate(options):
                bx1 = opt_start_x + int(opt_idx * single_opt_w)
                bx2 = opt_start_x + int((opt_idx + 1) * single_opt_w)

                bubble_crop = cell_roi[0:cell_h, bx1:bx2]
                bubble_resized = cv2.resize(bubble_crop, target_size, interpolation=cv2.INTER_AREA)

                bubbles_list.append({
                    'question': q_num,
                    'option': opt_char,
                    'col': col,
                    'row': row,
                    'opt_idx': opt_idx,
                    'image': bubble_resized
                })

    return bubbles_list

def build_grid_visualization(bubbles_list, patch_size=32, pad=2):
    grid_rows = 15
    grid_cols = 20

    cell_dim = patch_size + pad * 2
    gap_between_groups = 8

    canvas_h = grid_rows * cell_dim
    canvas_w = grid_cols * cell_dim + (3 * gap_between_groups)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 40

    for item in bubbles_list:
        r = item['row']
        c_group = item['col']
        opt_i = item['opt_idx']
        
        c_flat = c_group * 5 + opt_i
        extra_gap = c_group * gap_between_groups
        x_pos = c_flat * cell_dim + pad + extra_gap
        y_pos = r * cell_dim + pad

        img = item['image']
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        canvas[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size] = img

        cv2.rectangle(
            canvas,
            (x_pos - 1, y_pos - 1),
            (x_pos + patch_size, y_pos + patch_size),
            (80, 80, 80),
            1
        )

    return canvas


# -------------------------------------------------------------------
# 3. CHƯƠNG TRÌNH CHẠY DÙNG THỬ
# -------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "dataset/raw/001.jpg"  # Thay bằng đường dẫn ảnh bài thi của bạn

    try:
        # 1. Quét và nắn phẳng bài thi
        scanned = scan_and_flatten_paper(IMAGE_PATH)

        # 2. Trích xuất vùng bài làm 60 câu
        answer_roi = extract_answer_region(scanned)

        # 3. Cắt 300 ô trắc nghiệm nhỏ
        bubbles = crop_300_bubbles(answer_roi)
        print(f"✅ Đã trích xuất thành công: {len(bubbles)} ô trắc nghiệm!")

        # 4. Tạo ảnh hiển thị dạng lưới
        grid_img = build_grid_visualization(bubbles)

        # 5. Hiển thị bằng OpenCV
        cv2.imshow("Answer ROI", answer_roi)
        cv2.imshow("300 Bubbles Grid (15x20)", grid_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Hoặc lưu lại file ảnh lưới nếu cần
        # cv2.imwrite("bubbles_grid_output.jpg", grid_img)

    except Exception as e:
        print(f"❌ Lỗi xử lý: {e}")