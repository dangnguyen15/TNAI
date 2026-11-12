import cv2
import numpy as np

def main():
    # 1. Khởi tạo Webcam (0 là camera mặc định của laptop)
    cap = cv2.VideoCapture(0)

    # Cấu hình độ phân giải camera (1280x720 HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Không thể mở Webcam. Vui lòng kiểm tra quyền truy cập camera!")
        return

    print("🚀 Đã mở Camera!")
    print("📌 Hướng dẫn thao tác:")
    print("   - Đưa tờ giấy bài thi vào Khung Canh Góc")
    print("   - Khung đổi màu XANH LÁ khi phát hiện góc chuẩn")
    print("   - Nhấn phím [SPACE] (Phím cách) để chụp ảnh")
    print("   - Nhấn phím [Q] hoặc [ESC] để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lỗi đọc dữ liệu từ camera!")
            break

        h, w = frame.shape[:2]
        display_frame = frame.copy()

        # 2. Định nghĩa kích thước & vị trí Khung Chuẩn (Tỉ lệ 3:4 cho tờ giấy)
        target_w = int(w * 0.50)
        target_h = int(target_w * (4 / 3))
        if target_h > int(h * 0.85):
            target_h = int(h * 0.85)
            target_w = int(target_h * (3 / 4))

        x1 = (w - target_w) // 2
        y1 = (h - target_h) // 2
        x2 = x1 + target_w
        y2 = y1 + target_h

        # 3. Phân tích Real-time tìm viền tờ giấy trong khung
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 30, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_aligned = False
        for c in contours:
            area = cv2.contourArea(c)
            # Kiểm tra diện tích contour có đủ lớn và nằm trong khung hay không
            if area > (target_w * target_h * 0.35):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                if len(approx) == 4:
                    is_aligned = True
                    break

        # 4. Cập nhật màu sắc & trạng thái khung canh góc
        if is_aligned:
            color = (0, 255, 0)  # Xanh lá: Đã vuông góc / Chuẩn vị trí
            status_text = "KHUNG CHUAN! NHAN [SPACE] DE CHUP"
        else:
            color = (0, 0, 255)  # Đỏ: Chưa cân / Chưa phát hiện đủ 4 góc
            status_text = "CANH GIAY VAO TRONG KHUNG..."

        # Vẽ 4 tai định hình ở 4 góc (Corner Markers)
        line_len = 35
        thick = 3
        # Góc trên-trái
        cv2.line(display_frame, (x1, y1), (x1 + line_len, y1), color, thick)
        cv2.line(display_frame, (x1, y1), (x1, y1 + line_len), color, thick)
        # Góc trên-phải
        cv2.line(display_frame, (x2, y1), (x2 - line_len, y1), color, thick)
        cv2.line(display_frame, (x2, y1), (x2, y1 + line_len), color, thick)
        # Góc dưới-trái
        cv2.line(display_frame, (x1, y2), (x1 + line_len, y2), color, thick)
        cv2.line(display_frame, (x1, y2), (x1, y2 - line_len), color, thick)
        # Góc dưới-phải
        cv2.line(display_frame, (x2, y2), (x2 - line_len, y2), color, thick)
        cv2.line(display_frame, (x2, y2), (x2, y2 - line_len), color, thick)

        # Khung viền mảnh kết nối các góc
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

        # Hiển thị văn bản hướng dẫn
        cv2.putText(display_frame, status_text, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # 5. Hiển thị cửa sổ Camera OpenCV
        cv2.imshow("OMR Scanner - Realtime Camera", display_frame)

        # 6. Bắt sự kiện bàn phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Phím SPACE
            cv2.imwrite("captured_paper.jpg", frame)
            print("📸 Đã chụp thành công! Ảnh được lưu tại 'captured_paper.jpg'")

            # Hiệu ứng báo chụp thành công trên màn hình
            cv2.putText(display_frame, "DA CHUP ANH!", (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("OMR Scanner - Realtime Camera", display_frame)
            cv2.waitKey(600)
            break
        elif key == ord('q') or key == 27:  # Phím Q hoặc ESC
            print("👋 Đã đóng camera.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()