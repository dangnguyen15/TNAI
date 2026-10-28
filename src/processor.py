import os
import cv2
import numpy as np

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# KIẾN TRÚC MẠNG OMRPATCHNET ĐỒNG BỘ NGUYÊN BẢN TỪ FILE HUẤN LUYỆN COLAB
# ==============================================================================
if HAS_TORCH:
    class OMRPatchNet(nn.Module):
        def __init__(self, num_classes=7):
            super(OMRPatchNet, self).__init__()

            # Block 1: Input (3, 32, 32) -> Output (32, 16, 16)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)

            # Block 2: Input (32, 16, 16) -> Output (64, 8, 8)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)

            # Block 3: Input (64, 8, 8) -> Output (128, 4, 4)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.3)

            # Fully Connected Layers
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))

            x = x.view(x.size(0), -1)  # Flatten
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def scan_and_flatten_paper(image_path_or_img, target_w=1200, target_h=1600):
    """TẦNG 1: Nắn phẳng toàn bộ tờ giấy bài thi."""
    if isinstance(image_path_or_img, str):
        image = cv2.imread(image_path_or_img)
    else:
        image = image_path_or_img

    if image is None:
        raise ValueError("Không thể mở file ảnh đầu vào!")

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    for c in contours:
        if cv2.contourArea(c) < (h_img * w_img * 0.20):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper_pts = approx.reshape(4, 2)
            break
        elif len(approx) > 4:
            rect_rot = cv2.minAreaRect(c)
            paper_pts = np.int32(cv2.boxPoints(rect_rot))
            break

    if paper_pts is None:
        paper_pts = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype="float32")

    rect = order_points(paper_pts)
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (target_w, target_h))


def extract_exact_table_roi(scanned_img, target_w=1000, target_h=600):
    """TẦNG 2: Cắt bảng 60 câu nắn về 1000x600 px."""
    h, w = scanned_img.shape[:2]
    rough_crop = scanned_img[int(h * 0.50):int(h * 0.97), int(w * 0.03):int(w * 0.97)]
    crop_h, crop_w = rough_crop.shape[:2]

    gray = cv2.cvtColor(rough_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(crop_w * 0.25), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(crop_h * 0.25)))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_pts = None
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c_max) > (crop_h * crop_w * 0.30):
            peri = cv2.arcLength(c_max, True)
            approx = cv2.approxPolyDP(c_max, 0.02 * peri, True)
            if len(approx) == 4:
                table_pts = approx.reshape(4, 2)
            else:
                rect_rot = cv2.minAreaRect(c_max)
                table_pts = np.int32(cv2.boxPoints(rect_rot))

    if table_pts is not None:
        rect = order_points(table_pts)
        dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        table_roi = cv2.warpPerspective(rough_crop, M, (target_w, target_h))
    else:
        table_roi = cv2.resize(rough_crop, (target_w, target_h))

    return table_roi


def extract_and_draw_300_bubbles(table_roi, target_size=(32, 32)):
    """TẦNG 3: TRÍCH XUẤT 300 Ô VỚI BỘ TỌA ĐỘ CÂN CHỈNH MỚI NHẤT CỦA BẠN."""
    roi_h, roi_w = table_roi.shape[:2]
    overlay_img = table_roi.copy()

    options = ['a', 'b', 'c', 'd', 'e']
    bubbles_list = []

    group_w = roi_w / 4.0      # 250.0 px mỗi nhóm
    y_margin_top = 10
    y_margin_bottom = 31
    active_h = roi_h - y_margin_top - y_margin_bottom
    row_h = active_h / 15.0

    box_w = 40
    box_h = 40

    col_opt_x_ratios = [
        [0.327, 0.476, 0.621, 0.768, 0.915],
        [0.311, 0.466, 0.600, 0.750, 0.895],
        [0.288, 0.430, 0.567, 0.714, 0.877],
        [0.241, 0.420, 0.536, 0.688, 0.856]
    ]

    for col_idx in range(4):
        group_x1 = col_idx * group_w
        current_ratios = col_opt_x_ratios[col_idx]

        for row_idx in range(15):
            q_num = col_idx * 15 + row_idx + 1
            cy = int(y_margin_top + (row_idx + 0.5) * row_h)

            for opt_idx, opt_char in enumerate(options):
                cx = int(group_x1 + group_w * current_ratios[opt_idx])

                bx1 = max(0, cx - box_w // 2)
                bx2 = min(roi_w, cx + box_w // 2)
                by1 = max(0, cy - box_h // 2)
                by2 = min(roi_h, cy + box_h // 2)

                crop_patch = table_roi[by1:by2, bx1:bx2]
                if crop_patch.size > 0:
                    bubble_resized = cv2.resize(crop_patch, target_size, interpolation=cv2.INTER_AREA)
                else:
                    bubble_resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

                bubbles_list.append({
                    'question': q_num,
                    'option': opt_char,
                    'col': col_idx,
                    'row': row_idx,
                    'opt_idx': opt_idx,
                    'image': bubble_resized,
                    'bbox': (bx1, by1, bx2, by2)
                })

                cv2.rectangle(overlay_img, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

    return bubbles_list, overlay_img


def load_pytorch_model(model_path):
    """Nạp PyTorch Checkpoint (.pth) đã train từ Colab."""
    if not HAS_TORCH or not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = OMRPatchNet(num_classes=7)

        class_to_idx = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            class_to_idx = checkpoint.get('class_to_idx', None)
        else:
            model = checkpoint

        model.to(device)
        model.eval()
        return model, class_to_idx
    except Exception as e:
        print(f"Không thể nạp PyTorch Model từ {model_path}: {e}")
        return None, None


def predict_bubbles_with_cnn(bubbles_list, model_path=None):
    """
    SUY LUẬN CHÍNH XÁC VỚI MÔ HÌNH TRAIN TỪ COLAB (OMRPatchNet)
    - Xử lý ảnh: RGB + ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    - Mapping 7 nhãn: 'crossed' -> Gạch X, 'filled' -> Tô, 'empty_*' -> Trắng
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path_best = os.path.join(base_dir, "models", "omr_model_best.pth")
        model_path_v1 = os.path.join(base_dir, "models", "model_v1.pth")
        model_path = model_path_best if os.path.exists(model_path_best) else model_path_v1

    model, class_to_idx = load_pytorch_model(model_path)
    predictions = []

    # Map mặc định 7 nhãn chuẩn từ PyTorch ImageFolder
    # 0: crossed, 1..5: empty_a..e, 6: filled
    default_classes = ['crossed', 'empty_a', 'empty_b', 'empty_c', 'empty_d', 'empty_e', 'filled']
    idx_to_class = {i: cls_name for i, cls_name in enumerate(default_classes)}

    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    if model is not None and HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_list = []

        # BỘ CHUẨN HÓA IMAGENET GIỐNG HỆT COLAB TRAIN
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for item in bubbles_list:
            img_bgr = item['image']  # (32, 32, 3) BGR từ OpenCV
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Normalize [0, 1] -> Standardize ImageNet
            norm_img = img_rgb.astype("float32") / 255.0
            norm_img = (norm_img - mean) / std

            # Chuyển HWC (32, 32, 3) -> CHW (3, 32, 32)
            tensor_img = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1)
            tensor_list.append(tensor_img)

        batch_x = torch.stack(tensor_list).to(device)  # Shape (300, 3, 32, 32)

        with torch.no_grad():
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            predictions = preds.cpu().numpy().tolist()
    else:
        # Fallback đếm pixel mực nếu chưa nạp được PyTorch
        for item in bubbles_list:
            gray = cv2.cvtColor(item['image'], cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fill_ratio = cv2.countNonZero(thresh) / float(thresh.shape[0] * thresh.shape[1])
            predictions.append(6 if fill_ratio > 0.40 else 1)

    options_map = ['A', 'B', 'C', 'D', 'E']
    results_by_question = {}

    for i, item in enumerate(bubbles_list):
        q = item['question']
        opt_idx = item['opt_idx']
        pred_cls_idx = predictions[i]
        cls_name = idx_to_class.get(pred_cls_idx, 'empty_a')

        # QUY ĐỔI 7 NHÃN CỦA MÔ HÌNH VỀ 3 TRẠNG THÁI HIỂN THỊ
        if cls_name == 'filled':
            state_name = "Tô"
        elif cls_name == 'crossed':
            state_name = "Gạch X"
        else:
            state_name = "Trắng"

        if q not in results_by_question:
            results_by_question[q] = {
                'options_detail': {},
                'valid_selected': [],
                'crossed_selected': [],
                'col': item['col'] + 1
            }

        opt_char = options_map[opt_idx]
        results_by_question[q]['options_detail'][opt_char] = state_name

        if state_name == "Tô":
            results_by_question[q]['valid_selected'].append(opt_char)
        elif state_name == "Gạch X":
            results_by_question[q]['crossed_selected'].append(opt_char)

    final_results = []
    for q in range(1, 61):
        info = results_by_question.get(q, {'options_detail': {}, 'valid_selected': [], 'crossed_selected': [], 'col': 1})
        details = info['options_detail']
        valid_sel = info['valid_selected']
        crossed_sel = info['crossed_selected']

        detail_str = " | ".join([f"{opt}:{details.get(opt, 'Trắng')}" for opt in options_map])

        if len(valid_sel) == 1:
            ans_str = valid_sel[0]
            status = "Hợp lệ"
        elif len(valid_sel) > 1:
            ans_str = ",".join(valid_sel)
            status = "Lỗi (Tô nhiều ô)"
        else:
            ans_str = "—"
            status = "Bỏ câu"

        final_results.append({
            'question': q,
            'col': info['col'],
            'detail_str': detail_str,
            'answer': ans_str,
            'status': status,
            'details': details,
            'valid_selected': valid_sel,
            'crossed_selected': crossed_sel
        })

    return final_results