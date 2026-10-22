import os
import glob
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Thiết lập đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset01")

class YOLO_Labeler_App:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI - Giao Diện Gán Nhãn Phiếu Trắc Nghiệm")
        self.root.geometry("1150x850")

        # 1. Lấy danh sách ảnh trong dataset01
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_paths = [f for f in glob.glob(os.path.join(SOURCE_DIR, "*")) if f.lower().endswith(valid_exts)]
        
        # Sắp xếp danh sách theo số (0.jpg, 1.jpg, 2.jpg...)
        self.image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else x)

        if not self.image_paths:
            messagebox.showerror("Lỗi", f"Không tìm thấy ảnh trong thư mục:\n{SOURCE_DIR}")
            root.destroy()
            return

        self.current_idx = 0
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        self.orig_img = None
        self.disp_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # 2. Thanh điều khiển phía trên
        top_frame = tk.Frame(root, bg="#2c3e50", pady=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = tk.Label(top_frame, text="", font=("Arial", 12, "bold"), fg="white", bg="#2c3e50")
        self.lbl_info.pack(side=tk.LEFT, padx=15)

        btn_prev = tk.Button(top_frame, text="◄ Ảnh trước [A]", command=self.prev_img, font=("Arial", 10, "bold"), bg="#bdc3c7")
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(top_frame, text="Ảnh sau [D] ►", command=self.save_and_next, font=("Arial", 10, "bold"), bg="#2ecc71", fg="white")
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(top_frame, text="Xóa khung [C]", command=self.clear_box, font=("Arial", 10, "bold"), bg="#e74c3c", fg="white")
        btn_clear.pack(side=tk.LEFT, padx=5)

        # 3. Khung Canvas hiển thị ảnh
        self.canvas = tk.Canvas(root, bg="#1e1e1e", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 4. Bắt sự kiện chuột và phím tắt
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        root.bind("<a>", lambda e: self.prev_img())
        root.bind("<d>", lambda e: self.save_and_next())
        root.bind("<Left>", lambda e: self.prev_img())
        root.bind("<Right>", lambda e: self.save_and_next())
        root.bind("<c>", lambda e: self.clear_box())

        # Cập nhật khi resize cửa sổ
        self.canvas.bind("<Configure>", lambda e: self.render_current_image())

        self.load_image_data()

    def get_current_paths(self):
        img_path = self.image_paths[self.current_idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        return img_path, txt_path

    def load_image_data(self):
        img_path, txt_path = self.get_current_paths()
        img_name = os.path.basename(img_path)
        
        # Mở ảnh gốc
        self.orig_img = Image.open(img_path)
        self.current_box = None # (x1, y1, x2, y2) theo tọa độ ảnh gốc

        # Đọc nhãn cũ nếu đã tồn tại
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = list(map(float, line.split()))
                        if len(parts) >= 5:
                            _, xc, yc, nw, nh = parts[:5]
                            w_orig, h_orig = self.orig_img.size
                            x1 = (xc - nw / 2) * w_orig
                            y1 = (yc - nh / 2) * h_orig
                            x2 = (xc + nw / 2) * w_orig
                            y2 = (yc + nh / 2) * h_orig
                            self.current_box = (x1, y1, x2, y2)
            except Exception as e:
                print(f"Lỗi đọc label {txt_path}: {e}")

        # Cập nhật thanh tiêu đề
        status = "🟢 Đã có nhãn" if self.current_box else "🔴 Chưa gán nhãn"
        self.lbl_info.config(text=f"Ảnh: [{self.current_idx + 1}/{len(self.image_paths)}] - {img_name} | Trạng thái: {status}")

        self.render_current_image()

    def render_current_image(self):
        if not self.orig_img:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        w_orig, h_orig = self.orig_img.size
        self.scale = min(canvas_w / w_orig, canvas_h / h_orig)
        disp_w = int(w_orig * self.scale)
        disp_h = int(h_orig * self.scale)

        self.offset_x = (canvas_w - disp_w) // 2
        self.offset_y = (canvas_h - disp_h) // 2

        resized = self.orig_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.disp_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.disp_img)

        # Vẽ lại khung nếu có
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cx1 = x1 * self.scale + self.offset_x
            cy1 = y1 * self.scale + self.offset_y
            cx2 = x2 * self.scale + self.offset_x
            cy2 = y2 * self.scale + self.offset_y
            self.rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00FF00", width=3)

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="#00FF00", width=3)

    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        end_x, end_y = event.x, event.y
        cx1, cx2 = min(self.start_x, end_x), max(self.start_x, end_x)
        cy1, cy2 = min(self.start_y, end_y), max(self.start_y, end_y)

        # Chuyển về tọa độ ảnh gốc
        x1 = (cx1 - self.offset_x) / self.scale
        y1 = (cy1 - self.offset_y) / self.scale
        x2 = (cx2 - self.offset_x) / self.scale
        y2 = (cy2 - self.offset_y) / self.scale

        w_orig, h_orig = self.orig_img.size
        x1, x2 = max(0, x1), min(w_orig, x2)
        y1, y2 = max(0, y1), min(h_orig, y2)

        if (x2 - x1) > 20 and (y2 - y1) > 20:
            self.current_box = (x1, y1, x2, y2)
        else:
            self.clear_box()

    def save_current_label(self):
        img_path, txt_path = self.get_current_paths()
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            w_orig, h_orig = self.orig_img.size
            
            x_center = ((x1 + x2) / 2) / w_orig
            y_center = ((y1 + y2) / 2) / h_orig
            norm_w = (x2 - x1) / w_orig
            norm_h = (y2 - y1) / h_orig

            with open(txt_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        else:
            if os.path.exists(txt_path):
                os.remove(txt_path)

    def save_and_next(self):
        self.save_current_label()
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image_data()
        else:
            messagebox.showinfo("Thành công", "Bạn đã duyệt hết toàn bộ tập ảnh!")

    def prev_img(self):
        self.save_current_label()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image_data()

    def clear_box(self):
        self.current_box = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        img_path, txt_path = self.get_current_paths()
        status = "🔴 Chưa gán nhãn"
        img_name = os.path.basename(img_path)
        self.lbl_info.config(text=f"Ảnh: [{self.current_idx + 1}/{len(self.image_paths)}] - {img_name} | Trạng thái: {status}")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLO_Labeler_App(root)
    root.mainloop()