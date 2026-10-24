import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import các hàm xử lý từ module processor
from processor import (
    scan_and_flatten_paper, 
    extract_answer_region, 
    crop_300_bubbles, 
    build_grid_visualization
)

class OMRScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI - Phần Mềm Scan & Trích Xuất 300 Ô Trắc Nghiệm")
        self.root.geometry("1400x820")
        self.root.configure(bg="#1e1e2e")

        self.scanned_img_cv = None
        self.roi_img_cv = None
        self.grid_img_cv = None

        # --- Thanh Điều Khiển Phía Trên ---
        control_frame = tk.Frame(root, bg="#2b2b3b", height=60, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_select = tk.Button(
            control_frame, 
            text="📁 Chọn Ảnh Bài Thi", 
            command=self.load_image, 
            font=("Arial", 11, "bold"), 
            bg="#2ecc71", 
            fg="white", 
            padx=15, pady=5, 
            relief=tk.FLAT,
            cursor="hand2"
        )
        btn_select.pack(side=tk.LEFT, padx=20)

        btn_save = tk.Button(
            control_frame, 
            text="💾 Lưu Kết Quả Scan & Lưới Ô", 
            command=self.save_results, 
            font=("Arial", 11, "bold"), 
            bg="#3498db", 
            fg="white", 
            padx=15, pady=5, 
            relief=tk.FLAT,
            cursor="hand2"
        )
        btn_save.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(
            control_frame, 
            text="Trạng thái: Sẵn sàng chọn ảnh...", 
            font=("Arial", 10, "italic"), 
            fg="#a6adc8", 
            bg="#2b2b3b"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # --- Khung Hiển Thị Ảnh Kết Quả (3 Cột) ---
        display_frame = tk.Frame(root, bg="#1e1e2e")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Cột 1: Ảnh Tờ Giấy Nắn Phẳng
        panel_left = tk.LabelFrame(display_frame, text=" 1. Tờ Giấy Nắn Phẳng ", font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#181825")
        panel_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

        self.canvas_scanned = tk.Label(panel_left, bg="#11111b")
        self.canvas_scanned.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Cột 2: Vùng Trắc Nghiệm (ROI)
        panel_mid = tk.LabelFrame(display_frame, text=" 2. Vùng Bài Làm (ROI) ", font=("Arial", 10, "bold"), fg="#cdd6f4", bg="#181825")
        panel_mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

        self.canvas_roi = tk.Label(panel_mid, bg="#11111b")
        self.canvas_roi.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Cột 3: Lưới 300 Ô Trắc Nghiệm (Grid 15x20)
        panel_right = tk.LabelFrame(display_frame, text=" 3. Lưới 300 Ô Trắc Nghiệm (32x32 px) ", font=("Arial", 10, "bold"), fg="#f1c40f", bg="#181825")
        panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=3)

        self.canvas_grid = tk.Label(panel_right, bg="#11111b")
        self.canvas_grid.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

    def cv2_to_imagetk(self, cv_img, max_w, max_h):
        """Chuyển đổi ảnh OpenCV sang ImageTk có tự động resize vừa khung hiển thị."""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        w, h = pil_img.size
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.heic")]
        )
        if not file_path:
            return

        try:
            self.lbl_status.config(text="Đang xử lý scan, cắt 300 ô và dựng lưới...")
            self.root.update()

            # 1. Nắn phẳng bài thi
            self.scanned_img_cv = scan_and_flatten_paper(file_path)

            # 2. Trích xuất vùng bài làm 60 câu (ROI)
            self.roi_img_cv = extract_answer_region(self.scanned_img_cv)

            # 3. Cắt 300 ô và dựng Lưới 15x20
            bubbles = crop_300_bubbles(self.roi_img_cv)
            self.grid_img_cv = build_grid_visualization(bubbles)

            # 4. Hiển thị cả 3 ảnh lên giao diện
            max_w_left = self.canvas_scanned.winfo_width() or 400
            max_h_left = self.canvas_scanned.winfo_height() or 650

            max_w_mid = self.canvas_roi.winfo_width() or 400
            max_h_mid = self.canvas_roi.winfo_height() or 650

            max_w_right = self.canvas_grid.winfo_width() or 500
            max_h_right = self.canvas_grid.winfo_height() or 650

            img_tk_scanned = self.cv2_to_imagetk(self.scanned_img_cv, max_w_left, max_h_left)
            img_tk_roi = self.cv2_to_imagetk(self.roi_img_cv, max_w_mid, max_h_mid)
            img_tk_grid = self.cv2_to_imagetk(self.grid_img_cv, max_w_right, max_h_right)

            self.canvas_scanned.config(image=img_tk_scanned)
            self.canvas_scanned.image = img_tk_scanned

            self.canvas_roi.config(image=img_tk_roi)
            self.canvas_roi.image = img_tk_roi

            self.canvas_grid.config(image=img_tk_grid)
            self.canvas_grid.image = img_tk_grid

            filename = os.path.basename(file_path)
            self.lbl_status.config(text=f"🟢 Đã tách thành công 300 ô trắc nghiệm: {filename}")

        except Exception as e:
            messagebox.showerror("Lỗi Xử Lý", f"Không thể xử lý ảnh:\n{str(e)}")
            self.lbl_status.config(text="🔴 Lỗi xử lý ảnh!")

    def save_results(self):
        if self.scanned_img_cv is None or self.roi_img_cv is None or self.grid_img_cv is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả scan để lưu!")
            return

        save_dir = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if save_dir:
            path_scanned = os.path.join(save_dir, "scanned_paper.jpg")
            path_roi = os.path.join(save_dir, "omr_answer_region.jpg")
            path_grid = os.path.join(save_dir, "bubbles_grid_300.jpg")

            cv2.imwrite(path_scanned, self.scanned_img_cv)
            cv2.imwrite(path_roi, self.roi_img_cv)
            cv2.imwrite(path_grid, self.grid_img_cv)

            messagebox.showinfo(
                "Thành công", 
                f"Đã lưu thành công 3 file ảnh:\n- scanned_paper.jpg\n- omr_answer_region.jpg\n- bubbles_grid_300.jpg\n\nTại: {save_dir}"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = OMRScannerApp(root)
    root.mainloop()