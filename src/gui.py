import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from processor import scan_and_flatten_paper, extract_answer_region

class OMRScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI - Phần Mềm Scan Phiếu Trắc Nghiệm")
        self.root.geometry("1280x800")
        self.root.configure(bg="#1e1e2e")

        self.scanned_img_cv = None
        self.roi_img_cv = None

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
            px=15, py=5, 
            relief=tk.FLAT,
            cursor="hand2"
        )
        btn_select.pack(side=tk.LEFT, padx=20)

        btn_save = tk.Button(
            control_frame, 
            text="💾 Lưu Kết Quả Scan", 
            command=self.save_results, 
            font=("Arial", 11, "bold"), 
            bg="#3498db", 
            fg="white", 
            px=15, py=5, 
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

        # --- Khung Hiển Thị Ảnh Kết Quả ---
        display_frame = tk.Frame(root, bg="#1e1e2e")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Cột 1: Ảnh Tờ Giấy Đã Scan & Nắn Phẳng
        panel_left = tk.LabelFrame(display_frame, text=" 1. Tờ Giấy Nắn Phẳng (Full Scan) ", font=("Arial", 11, "bold"), fg="#cdd6f4", bg="#181825")
        panel_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_scanned = tk.Label(panel_left, bg="#11111b")
        self.canvas_scanned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Cột 2: Ảnh Chỉ Vùng Trắc Nghiệm
        panel_right = tk.LabelFrame(display_frame, text=" 2. Vùng Bài Làm Trắc Nghiệm (ROI) ", font=("Arial", 11, "bold"), fg="#cdd6f4", bg="#181825")
        panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_roi = tk.Label(panel_right, bg="#11111b")
        self.canvas_roi.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def cv2_to_imagetk(self, cv_img, max_w, max_h):
        """Chuyển đổi ảnh OpenCV sang ImageTk có tự động resize vừa khung hiển thị."""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Tính tỷ lệ scale vừa khung canvas
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.heic")]
        )
        if not file_path:
            return

        try:
            self.lbl_status.config(text="Đang xử lý scan và nắn phẳng ảnh...")
            self.root.update()

            # 1. Gọi module Scan & Nắn phẳng
            self.scanned_img_cv = scan_and_flatten_paper(file_path)

            # 2. Trích xuất riêng vùng trắc nghiệm
            self.roi_img_cv = extract_answer_region(self.scanned_img_cv)

            # 3. Hiển thị lên giao diện
            max_w = self.canvas_scanned.winfo_width() or 550
            max_h = self.canvas_scanned.winfo_height() or 680

            img_tk_scanned = self.cv2_to_imagetk(self.scanned_img_cv, max_w, max_h)
            img_tk_roi = self.cv2_to_imagetk(self.roi_img_cv, max_w, max_h)

            self.canvas_scanned.config(image=img_tk_scanned)
            self.canvas_scanned.image = img_tk_scanned

            self.canvas_roi.config(image=img_tk_roi)
            self.canvas_roi.image = img_tk_roi

            filename = os.path.basename(file_path)
            self.lbl_status.config(text=f"🟢 Đã xử lý thành công: {filename}")

        except Exception as e:
            messagebox.showerror("Lỗi Xử Lý", f"Không thể xử lý ảnh:\n{str(e)}")
            self.lbl_status.config(text="🔴 Lỗi xử lý ảnh!")

    def save_results(self):
        if self.scanned_img_cv is None or self.roi_img_cv is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả scan để lưu!")
            return

        save_dir = filedialog.askdirectory(title="Chọn thư mục lưu kết quả")
        if save_dir:
            path_scanned = os.path.join(save_dir, "scanned_paper.jpg")
            path_roi = os.path.join(save_dir, "omr_answer_region.jpg")

            cv2.imwrite(path_scanned, self.scanned_img_cv)
            cv2.imwrite(path_roi, self.roi_img_cv)

            messagebox.showinfo("Thành công", f"Đã lưu ảnh scan và vùng trắc nghiệm vào:\n{save_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OMRScannerApp(root)
    root.mainloop()