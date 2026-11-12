import os
import cv2
import numpy as np

# PyTorch là tùy chọn: nếu không có model, chương trình dùng fallback OMR.
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

# OCR là tùy chọn. Cần cài pytesseract và Tesseract OCR riêng trên Windows.
HAS_OCR = False
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    pass


# =============================================================================
# CNN - GIỮ NGUYÊN KIẾN TRÚC MODEL CŨ
# =============================================================================
if HAS_TORCH:
    class OMRPatchNet(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)


# =============================================================================
# CẤU HÌNH TEMPLATE PHIẾU HOA SEN
# Các tỷ lệ được đo theo ảnh mẫu người dùng gửi. Trang đã được chuẩn hóa A4.
# =============================================================================
PAGE_W = 1200
PAGE_H = 1697

# Tâm các lựa chọn A-E theo tỷ lệ chiều rộng toàn trang.
# Mỗi hàng có 5 lựa chọn; 4 nhóm tương ứng câu 1-15, 16-30, 31-45, 46-60.
BUBBLE_X_RATIOS = [
    [0.133, 0.164, 0.197, 0.231, 0.265],
    [0.352, 0.386, 0.418, 0.450, 0.482],
    [0.576, 0.608, 0.642, 0.674, 0.706],
    [0.784, 0.816, 0.849, 0.882, 0.916],
]
BUBBLE_Y_TOP = 0.571
BUBBLE_Y_BOTTOM = 0.925
BUBBLE_PATCH = 40

# Vùng bảng câu trả lời sau khi nắn trang (x1, y1, x2, y2), tỷ lệ trang.
ANSWER_TABLE_BOX = (0.055, 0.545, 0.945, 0.955)


def order_points(pts):
    """Sắp xếp 4 điểm: trái-trên, phải-trên, phải-dưới, trái-dưới."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _find_page_contour(image):
    """Tìm vùng giấy sáng trên nền tối; có fallback minAreaRect."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu thích nghi ngưỡng theo độ sáng của từng ảnh.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Đóng các khe nhỏ ở viền giấy, không dùng kernel quá lớn để tránh dính nền.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Không chỉ lấy contour lớn nhất một cách mù quáng: lọc diện tích và tỷ lệ.
    candidates = []
    image_area = float(h * w)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.12:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        ratio = cw / max(ch, 1)
        # A4 dọc trong ảnh có thể bị phối cảnh; cho phép tỷ lệ rộng hơn A4.
        if 0.42 <= ratio <= 1.15:
            candidates.append((area, contour))

    if not candidates:
        return None

    contour = max(candidates, key=lambda item: item[0])[1]
    perimeter = cv2.arcLength(contour, True)

    # Thử nhiều epsilon để tìm tứ giác ổn định.
    quad = None
    for eps in (0.012, 0.018, 0.025, 0.035):
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image_area * 0.12:
            quad = approx.reshape(4, 2)
            break

    if quad is None:
        # Khi cạnh giấy cong/nhăn, contour có nhiều đỉnh. Min-area rectangle
        # là fallback an toàn hơn việc lấy nhầm một contour nhỏ bên trong.
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect)

    return order_points(quad)


def _refine_page_corners(image, corners):
    """Tinh chỉnh nhẹ 4 góc theo cạnh gần nhất; không dịch quá xa."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 120)
    h, w = gray.shape
    refined = corners.copy()

    for i, (x, y) in enumerate(corners):
        x0, x1 = max(0, int(x) - 18), min(w, int(x) + 19)
        y0, y1 = max(0, int(y) - 18), min(h, int(y) + 19)
        patch = edges[y0:y1, x0:x1]
        ys, xs = np.where(patch > 0)
        if len(xs) < 5:
            continue
        pts = np.column_stack((xs + x0, ys + y0)).astype(np.float32)
        # Chọn điểm cạnh gần góc contour nhất để tránh nhảy sang chữ/đường kẻ.
        d = np.sum((pts - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
        j = int(np.argmin(d))
        candidate = pts[j]
        if np.linalg.norm(candidate - [x, y]) <= 12:
            refined[i] = candidate
    return refined


def scan_and_flatten_paper(image_path_or_img, target_w=PAGE_W, target_h=PAGE_H):
    """
    Phát hiện trang giấy và nắn về tỷ lệ A4 dọc.
    Trả về ảnh gốc nếu không phát hiện được viền đáng tin cậy.
    """
    if isinstance(image_path_or_img, (str, os.PathLike)):
        image = cv2.imread(str(image_path_or_img))
    else:
        image = image_path_or_img

    if image is None:
        raise ValueError("Không thể mở ảnh đầu vào.")

    image = image.copy()
    corners = _find_page_contour(image)
    if corners is None:
        raise ValueError(
            "Không tìm thấy tờ giấy. Hãy chụp trên nền tối, đủ sáng và thấy đủ 4 cạnh."
        )

    corners = _refine_page_corners(image, corners)
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        image, matrix, (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Cân bằng sáng nhẹ để ổn định bước phát hiện đường kẻ/ô.
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    warped = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return warped


def extract_exact_table_roi(scanned_img, target_w=1068, target_h=696):
    """Cắt bảng 60 câu theo template đã hiệu chỉnh, không dò contour lớn nhất."""
    h, w = scanned_img.shape[:2]
    x1, y1, x2, y2 = ANSWER_TABLE_BOX
    left, top = int(x1 * w), int(y1 * h)
    right, bottom = int(x2 * w), int(y2 * h)
    roi = scanned_img[top:bottom, left:right].copy()
    if roi.size == 0:
        raise ValueError("Không cắt được vùng bảng trắc nghiệm.")
    return cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def _detect_circle_centers(gray):
    """Phát hiện vòng tròn in sẵn; chỉ dùng để hiệu chỉnh tâm quanh template."""
    work = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        work, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=18,
        param1=100, param2=19,
        minRadius=7, maxRadius=19
    )
    if circles is None:
        return np.empty((0, 2), dtype=np.float32)
    return circles[0, :, :2].astype(np.float32)


def extract_and_draw_300_bubbles(table_roi, target_size=(32, 32)):
    """
    Định vị 300 ô theo template Hoa Sen.
    Hough Circle chỉ tinh chỉnh tâm trong bán kính nhỏ; nếu không phát hiện được
    thì vẫn dùng tọa độ template để tránh mất ô.
    """
    h, w = table_roi.shape[:2]
    overlay = table_roi.copy()
    gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
    detected = _detect_circle_centers(gray)

    options = ["a", "b", "c", "d", "e"]
    bubbles = []
    row_ys = np.linspace(BUBBLE_Y_TOP, BUBBLE_Y_BOTTOM, 15) * h

    # Tọa độ X đang tính theo toàn trang; chuyển về ROI bảng.
    page_w = PAGE_W
    page_h = PAGE_H
    table_x1 = ANSWER_TABLE_BOX[0] * page_w
    table_y1 = ANSWER_TABLE_BOX[1] * page_h
    table_x2 = ANSWER_TABLE_BOX[2] * page_w
    table_y2 = ANSWER_TABLE_BOX[3] * page_h
    roi_scale_x = w / (table_x2 - table_x1)
    roi_scale_y = h / (table_y2 - table_y1)

    for group_idx in range(4):
        for row_idx in range(15):
            q_num = group_idx * 15 + row_idx + 1
            page_y = row_ys[row_idx] + table_y1 * roi_scale_y
            # BUBBLE_Y_TOP/BOTTOM là tỷ lệ trang, chuyển trực tiếp sang ROI.
            cy_expected = (BUBBLE_Y_TOP + row_idx *
                           (BUBBLE_Y_BOTTOM - BUBBLE_Y_TOP) / 14) * PAGE_H
            cy = (cy_expected - table_y1) * roi_scale_y

            for opt_idx, opt_char in enumerate(options):
                page_x = BUBBLE_X_RATIOS[group_idx][opt_idx] * PAGE_W
                cx = (page_x - table_x1) * roi_scale_x

                # Tinh chỉnh tâm bằng vòng tròn Hough gần vị trí kỳ vọng.
                if len(detected):
                    distances = np.linalg.norm(
                        detected - np.array([cx, cy], dtype=np.float32), axis=1
                    )
                    nearest = int(np.argmin(distances))
                    if distances[nearest] <= 10:
                        cx, cy = detected[nearest]

                half = BUBBLE_PATCH // 2
                bx1, bx2 = max(0, int(round(cx)) - half), min(w, int(round(cx)) + half)
                by1, by2 = max(0, int(round(cy)) - half), min(h, int(round(cy)) + half)
                patch = table_roi[by1:by2, bx1:bx2]
                if patch.size == 0:
                    patch = np.full((BUBBLE_PATCH, BUBBLE_PATCH, 3), 255, np.uint8)
                else:
                    patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)

                bubbles.append({
                    "question": q_num,
                    "option": opt_char,
                    "col": group_idx,
                    "row": row_idx,
                    "opt_idx": opt_idx,
                    "image": patch,
                    "bbox": (bx1, by1, bx2, by2),
                    "center": (float(cx), float(cy))
                })
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 220, 0), 1)
                cv2.circle(overlay, (int(cx), int(cy)), 2, (0, 0, 255), -1)

    return bubbles, overlay


# =============================================================================
# OCR VÙNG THÔNG TIN - TỌA ĐỘ TEMPLATE SAU KHI NẮN TRANG A4
# Các ô OCR là vùng trống sau nhãn in sẵn, không gồm phần chữ nhãn.
# =============================================================================
INFO_FIELDS = {
    "Họ và tên": (0.205, 0.140, 0.435, 0.164),
    "MSSV": (0.105, 0.183, 0.245, 0.204),
    "Lớp": (0.285, 0.183, 0.435, 0.204),
    "Kỳ thi / loại kỳ": (0.485, 0.140, 0.735, 0.164),
    "Ngày thi": (0.535, 0.162, 0.735, 0.183),
    "Phòng thi": (0.755, 0.162, 0.920, 0.183),
    "Môn thi": (0.485, 0.185, 0.755, 0.207),
    "Mã đề": (0.790, 0.185, 0.920, 0.207),
    "Giám thị 1": (0.105, 0.224, 0.425, 0.245),
    "Giám thị 2": (0.485, 0.224, 0.650, 0.245),
    "Số phách": (0.790, 0.224, 0.920, 0.245),
}


def _prepare_ocr_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def extract_student_information(scanned_img, tesseract_cmd=None, language="vie+eng"):
    """
    Cắt các trường thông tin và OCR nếu Tesseract đã được cài.
    Trả về dict {field: text}, dict {field: crop}, trạng thái OCR.
    Lưu ý: OCR chữ viết tay phụ thuộc chất lượng ảnh và không thể đảm bảo chính xác tuyệt đối.
    """
    if HAS_OCR:
        # Có thể cấu hình đường dẫn Tesseract bằng biến môi trường TESSERACT_CMD.
        # Tự dò vị trí cài đặt phổ biến trên Windows nếu chưa cấu hình.
        configured_cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if not configured_cmd:
            common_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(common_path):
                configured_cmd = common_path
        if configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd

    h, w = scanned_img.shape[:2]
    values, crops = {}, {}
    ocr_ready = HAS_OCR

    for field, (x1, y1, x2, y2) in INFO_FIELDS.items():
        left, top = int(x1 * w), int(y1 * h)
        right, bottom = int(x2 * w), int(y2 * h)
        crop = scanned_img[top:bottom, left:right].copy()
        crops[field] = crop

        if not HAS_OCR or crop.size == 0:
            values[field] = "(Chưa OCR)"
            continue

        prepared = _prepare_ocr_crop(crop)
        config = "--oem 3 --psm 7"
        try:
            text = pytesseract.image_to_string(
                prepared, lang=language, config=config
            ).strip()
            values[field] = " ".join(text.split()) if text else ""
        except Exception as exc:
            values[field] = f"(Lỗi OCR: {exc})"
            ocr_ready = False

    return values, crops, ocr_ready


# =============================================================================
# LOAD MODEL + DỰ ĐOÁN 300 Ô
# =============================================================================
def load_pytorch_model(model_path):
    if not HAS_TORCH or not model_path or not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = OMRPatchNet(num_classes=7)
        class_to_idx = None

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            class_to_idx = checkpoint.get("class_to_idx")
        else:
            model = checkpoint

        model.to(device)
        model.eval()
        return model, class_to_idx
    except Exception as exc:
        print(f"Không thể nạp model {model_path}: {exc}")
        return None, None


def _find_default_model():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "models", "omr_model_best.pth"),
        os.path.join(here, "models", "model_v1.pth"),
        os.path.join(os.path.dirname(here), "models", "omr_model_best.pth"),
        os.path.join(os.path.dirname(here), "models", "model_v1.pth"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def predict_bubbles_with_cnn(bubbles_list, model_path=None):
    model_path = model_path or _find_default_model()
    model, class_to_idx = load_pytorch_model(model_path)

    default_classes = [
        "crossed", "empty_a", "empty_b", "empty_c",
        "empty_d", "empty_e", "filled"
    ]
    idx_to_class = {i: name for i, name in enumerate(default_classes)}
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    predictions = []
    if model is not None and HAS_TORCH:
        device = next(model.parameters()).device
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors = []

        for item in bubbles_list:
            rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - mean) / std
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
    else:
        # Fallback đơn giản; không thay thế được CNN đã huấn luyện.
        for item in bubbles_list:
            gray = cv2.cvtColor(item["image"], cv2.COLOR_BGR2GRAY)
            # Bỏ viền ngoài để giảm ảnh hưởng vòng tròn in sẵn.
            inner = gray[7:-7, 7:-7]
            _, th = cv2.threshold(
                inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            fill_ratio = cv2.countNonZero(th) / float(th.size)
            predictions.append(6 if fill_ratio > 0.34 else 1)

    options = ["A", "B", "C", "D", "E"]
    by_question = {}

    for i, item in enumerate(bubbles_list):
        q = item["question"]
        cls_name = idx_to_class.get(predictions[i], "empty_a")
        if cls_name == "filled":
            state = "Tô"
        elif cls_name == "crossed":
            state = "Gạch X"
        else:
            state = "Trắng"

        info = by_question.setdefault(q, {
            "options_detail": {}, "valid_selected": [],
            "crossed_selected": [], "col": item["col"] + 1
        })
        opt = options[item["opt_idx"]]
        info["options_detail"][opt] = state
        if state == "Tô":
            info["valid_selected"].append(opt)
        elif state == "Gạch X":
            info["crossed_selected"].append(opt)

    results = []
    for q in range(1, 61):
        info = by_question.get(q, {
            "options_detail": {}, "valid_selected": [],
            "crossed_selected": [], "col": (q - 1) // 15 + 1
        })
        details = info["options_detail"]
        selected = info["valid_selected"]
        detail_str = " | ".join(
            f"{o}:{details.get(o, 'Trắng')}" for o in options
        )

        if len(selected) == 1:
            answer, status = selected[0], "Hợp lệ"
        elif len(selected) > 1:
            answer, status = ",".join(selected), "Lỗi (Tô nhiều ô)"
        else:
            answer, status = "—", "Bỏ câu"

        results.append({
            "question": q,
            "col": info["col"],
            "detail_str": detail_str,
            "answer": answer,
            "status": status,
            "details": details,
            "valid_selected": selected,
            "crossed_selected": info["crossed_selected"]
        })
    return results