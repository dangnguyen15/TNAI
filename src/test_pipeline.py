import os
import json
import glob
import cv2
import numpy as np


class A4GridScanner:
    """
    Pipeline xử lý quét tờ giấy A4 và phân vùng ô kẻ sẵn.
    Chạy thuần OpenCV kết hợp cấu trúc On-Device AI Placeholder.
    """
    def __init__(self, target_width=1650, target_height=2338, onnx_model_path=None):
        self.target_w = target_width
        self.target_h = target_height
        self.onnx_model_path = onnx_model_path
        self.session = None
        
        # Nối với mô hình AI (YOLOv8-OBB INT8) nếu có file onnx
        if self.onnx_model_path and os.path.exists(self.onnx_model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.onnx_model_path)
            except Exception as e:
                print(f"[WARN] Không thể khởi tạo ONNX runtime: {e}. Chuyển sang OpenCV fallback.")

    def _order_points(self, pts):
        """ Sắp xếp 4 tọa độ đỉnh theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_and_warp(self, image):
        """ Giai đoạn 1: Phát hiện 4 góc A4 và nắn phẳng (Perspective Transform) """
        orig_h, orig_w = image.shape[:2]
        
        # Nếu có ONNX Session, ưu tiên nhận diện 4 góc bằng AI
        if self.session is not None:
            # Code suy luận từ mô hình ONNX sẽ xử lý tại đây
            pass

        # Thuật toán OpenCV Fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

        if doc_cnt is None:
            # Lấy toàn bộ biên ảnh nếu không tìm thấy đa giác 4 đỉnh
            pts1 = np.float32([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]])
        else:
            pts1 = self._order_points(doc_cnt.reshape(4, 2))

        # Khung A4 phẳng chuẩn
        pts2 = np.float32([
            [0, 0],
            [self.target_w - 1, 0],
            [self.target_w - 1, self.target_h - 1],
            [0, self.target_h - 1]
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(image, matrix, (self.target_w, self.target_h))
        return warped

    def preprocess(self, warped_img):
        """ Giai đoạn 2: Cân bằng ánh sáng, khử bóng đổ & nhị phân hóa thích ứng """
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        # Triệt tiêu bóng bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Nhị phân hóa Adaptive Gaussian
        binary = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 8
        )
        return binary

    def extract_grid_cells(self, binary_img, min_area=1000, max_area=500000):
        """ Giai đoạn 3: Phân vùng ô kẻ bằng Morphological Operations """
        h, w = binary_img.shape

        # Lọc nét kẻ ngang
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
        horiz_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

        # Lọc nét kẻ dọc
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))
        vert_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vert_kernel, iterations=2)

        # Hợp nhất đường kẻ tạo lưới ô
        grid_mask = cv2.add(horiz_lines, vert_lines)

        # Tìm các đường viền khép kín đại diện cho ô
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            
            # Lọc nhiễu theo diện tích ô và kích thước tối thiểu
            if min_area < area < max_area and bw > 25 and bh > 25:
                cells.append((x, y, bw, bh))

        # Sắp xếp thứ tự các ô từ trên xuống dưới, từ trái sang phải
        cells = sorted(cells, key=lambda b: (b[1] // 30, b[0]))
        return cells, grid_mask

    def process_file(self, img_path, output_dir):
        """ Giai đoạn 4: Đưa qua Pipeline, cắt vùng ảnh ô & xuất kết quả JSON """
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Không đọc được ảnh từ: {img_path}")
            return

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(output_dir, base_name)
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        # 1. Nắn phẳng A4
        warped = self.detect_and_warp(image)

        # 2. Khử bóng & nhị phân hóa
        binary = self.preprocess(warped)

        # 3. Phân vùng ô lưới
        cells, _ = self.extract_grid_cells(binary)

        # 4. Trích xuất vùng và lưu kết quả
        debug_img = warped.copy()
        metadata = {
            "source_image": img_path,
            "target_size": [self.target_w, self.target_h],
            "total_cells_found": len(cells),
            "regions": []
        }

        for idx, (x, y, w, h) in enumerate(cells):
            cell_id = f"cell_{idx+1:03d}"
            
            # Cắt ảnh nhỏ của ô
            crop = warped[y:y+h, x:x+w]
            crop_filename = f"{cell_id}.png"
            crop_path = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            # Vẽ ô lên ảnh debug
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(idx+1), (x+5, y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            metadata["regions"].append({
                "id": cell_id,
                "bbox": [x, y, w, h],
                "crop_path": os.path.relpath(crop_path, save_dir)
            })

        # Xuất file ảnh & JSON
        cv2.imwrite(os.path.join(save_dir, "warped_a4.png"), warped)
        cv2.imwrite(os.path.join(save_dir, "debug_grid.png"), debug_img)
        
        with open(os.path.join(save_dir, "layout_info.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] {base_name}: Đã trích xuất {len(cells)} ô -> {save_dir}")


def main():
    # Tự động định vị thư mục dựa trên đường dẫn file script (TNAI/src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(src_dir, "test")
    output_dir = os.path.join(src_dir, "output")

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        print(f"[INFO] Đã tạo thư mục test tại '{test_dir}'. Hãy thả ảnh vào đây để chạy thử.")
        return

    # Lấy danh sách ảnh trong TNAI/src/test/
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))

    if not image_paths:
        print(f"[WARN] Thư mục '{test_dir}' đang trống. Hãy thêm file ảnh vào.")
        return

    # Khởi tạo scanner
    scanner = A4GridScanner(target_width=1650, target_height=2338)

    print(f"Thực thi pipeline trên {len(image_paths)} file ảnh...\n")
    for img_path in image_paths:
        scanner.process_file(img_path, output_dir)


if __name__ == "__main__":
    main()