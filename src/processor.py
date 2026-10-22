import cv2
import numpy as np

def order_points(pts):
    """Sắp xếp 4 tọa độ theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    return rect

def scan_and_flatten_paper(image_path, target_w=1200, target_h=1600):
    """
    1. Nhận diện 4 góc tờ giấy và nắn phẳng (Perspective Transform) về chuẩn 1200x1600.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể mở ảnh từ đường dẫn: {image_path}")

    h_orig, w_orig = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Tìm contour lớn nhất đại diện cho tờ giấy
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper_pts = approx.reshape(4, 2)
            break

    # Dự phòng: Nếu không bắt được 4 góc contour, lấy toàn bộ viền ảnh gốc
    if paper_pts is None:
        paper_pts = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype="float32")

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
    """
    2. Trích xuất riêng vùng chứa 60 câu trắc nghiệm dựa trên tọa độ tỷ lệ chuẩn.
    """
    h, w = scanned_img.shape[:2]
    
    # Tọa độ vùng trắc nghiệm trên mẫu phiếu Đại học Hoa Sen (Y: 42% -> 92%, X: 8% -> 92%)
    roi_x1 = int(w * 0.08)
    roi_y1 = int(h * 0.42)
    roi_x2 = int(w * 0.92)
    roi_y2 = int(h * 0.92)

    answer_roi = scanned_img[roi_y1:roi_y2, roi_x1:roi_x2]
    return answer_roi