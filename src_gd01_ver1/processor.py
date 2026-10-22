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

def scan_and_flatten_paper(image_path, target_w=1200, target_h=1600):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể mở ảnh: {image_path}")

    h_img, w_img = image.shape[:2]
    total_area = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Phép biến đổi Morphological đóng liền viền mép giấy
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    for c in contours:
        area = cv2.contourArea(c)
        
        # ĐIỀU KIỆN LỌC QUAN TRỌNG: Khung bắt được phải chiếm ít nhất 30% diện tích ảnh
        if area < (total_area * 0.30):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            paper_pts = approx.reshape(4, 2)
            break

    # Dự phòng: Nếu nền quá nhiễu không tìm thấy viền giấy > 30%, lấy toàn bộ khung ảnh gốc
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

def extract_answer_region(scanned_img):
    h, w = scanned_img.shape[:2]
    
    roi_x1 = int(w * 0.07)
    roi_y1 = int(h * 0.50)  
    roi_x2 = int(w * 0.93)
    roi_y2 = int(h * 0.96)

    answer_roi = scanned_img[roi_y1:roi_y2, roi_x1:roi_x2]
    return answer_roi