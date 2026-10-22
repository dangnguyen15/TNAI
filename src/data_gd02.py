import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# -------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN & CÁC LỚP NHÃN (CLASSES)
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

# Danh sách 7 nhãn kèm màu viền đánh dấu
LABEL_CONFIG = {
    '1': ('A', 'empty_a', '#3498db'), # Xanh dương - Ô 'a' trống
    '2': ('B', 'empty_b', '#2ecc71'), # Xanh lá - Ô 'b' trống
    '3': ('C', 'empty_c', '#e67e22'), # Cam - Ô 'c' trống
    '4': ('D', 'empty_d', '#9b59b6'), # Tím - Ô 'd' trống
    '5': ('E', 'empty_e', '#1abc9c'), # Xanh ngọc - Ô 'e' trống
    '6': ('FILLED',  'filled',  '#f1c40f'), # Vàng - Ô đã tô đen
    '7': ('CROSSED', 'crossed', '#e74c3c')  # Đỏ - Ô bị gạch chéo
}

for _, folder_name, _ in LABEL_CONFIG.values():
    os.makedirs(os.path.join(TRAIN_DIR, folder_name), exist_ok=True)


# -------------------------------------------------------------------
# 2. GIAO DIỆN KÉO THẢ, XOAY ẢNH & XÓA LỖI (UNDO)
# -------------------------------------------------------------------
class DragLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI - Công Cụ Kéo Thả Cắt & Gán Nhãn Ô Trắc Nghiệm")
        self.root.geometry("1280x900")
        self.root.configure(bg="#1e1e2e")

        self.image_files = []
        self.current_img_idx = 0
        self.current_img = None  # Ảnh gốc nguyên bản
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0

        self.active_key = '6'  # Mặc định nhãn FILLED
        self.start_x = None
        self.start_y = None
        self.temp_rect = None
        self.saved_count = 0

        # Lịch sử lưu các ô đã cắt của ảnh hiện tại để phục vụ tính năng Xóa/Undo
        self.crop_history = []

        self.setup_ui()
        self.bind_shortcuts()
        self.load_raw_folder()

    def setup_ui(self):
        # Top Bar Controls
        top_bar = tk.Frame(self.root, bg="#2b2b3b", height=50, pady=5)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_open = tk.Button(top_bar, text="📁 Thư Mục Raw", command=self.load_raw_folder, bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=10)
        btn_open.pack(side=tk.LEFT, padx=10)

        self.lbl_file_info = tk.Label(top_bar, text="File: Chưa chọn", bg="#2b2b3b", fg="#cdd6f4", font=("Arial", 11, "bold"))
        self.lbl_file_info.pack(side=tk.LEFT, padx=10)

        # Nút xoay ảnh 90 độ
        btn_rotate = tk.Button(top_bar, text="🔄 Xoay 90° (Phím R)", command=self.rotate_image, bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=10)
        btn_rotate.pack(side=tk.LEFT, padx=10)

        # NÚT XÓA Ô VỪA CẮT (UNDO)
        btn_undo = tk.Button(top_bar, text="↩️ Xóa Ô Vừa Cắt (Phím Z)", command=self.undo_last_crop, bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), padx=10)
        btn_undo.pack(side=tk.LEFT, padx=10)

        btn_prev = tk.Button(top_bar, text="◀ Ảnh Trước", command=self.prev_image, bg="#e67e22", fg="white", font=("Arial", 10, "bold"))
        btn_prev.pack(side=tk.RIGHT, padx=5)

        btn_next = tk.Button(top_bar, text="Ảnh Tiếp ▶", command=self.next_image, bg="#e67e22", fg="white", font=("Arial", 10, "bold"))
        btn_next.pack(side=tk.RIGHT, padx=10)

        # Label Selection Bar
        label_bar = tk.Frame(self.root, bg="#181825", pady=8)
        label_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(label_bar, text="NHÃN ĐANG CHỌN (Phím 1-7):", bg="#181825", fg="#a6adc8", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)

        self.label_buttons = {}
        for key, (class_name, _, color) in LABEL_CONFIG.items():
            btn = tk.Button(
                label_bar, text=f"[{key}] {class_name}", bg=color, fg="black" if key in ['6', '1', '5'] else "white",
                font=("Arial", 9, "bold"), padx=8, pady=2, command=lambda k=key: self.set_active_label(k)
            )
            btn.pack(side=tk.LEFT, padx=4)
            self.label_buttons[key] = btn

        self.lbl_active_status = tk.Label(label_bar, text="", bg="#181825", font=("Arial", 10, "bold"))
        self.lbl_active_status.pack(side=tk.RIGHT, padx=15)
        self.update_active_label_ui()

        # Canvas Hiển Thị Ảnh
        canvas_frame = tk.Frame(self.root, bg="#11111b")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#11111b", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def bind_shortcuts(self):
        # Đăng ký phím 1-7 chọn nhãn
        for key in LABEL_CONFIG.keys():
            self.root.bind(key, lambda e, k=key: self.set_active_label(k))
        
        # Phím R để xoay ảnh
        self.root.bind('r', lambda e: self.rotate_image())
        self.root.bind('R', lambda e: self.rotate_image())

        # Phím Z hoặc Ctrl+Z để Xóa / Undo
        self.root.bind('z', lambda e: self.undo_last_crop())
        self.root.bind('Z', lambda e: self.undo_last_crop())
        self.root.bind('<Control-z>', lambda e: self.undo_last_crop())

    def rotate_image(self):
        """Xoay ảnh 90 độ theo chiều kim đồng hồ"""
        if self.current_img is not None:
            self.current_img = cv2.rotate(self.current_img, cv2.ROTATE_90_CLOCKWISE)
            self.crop_history.clear()  # Xóa vết vẽ cũ khi xoay lại ảnh
            self.display_image()

    def set_active_label(self, key):
        self.active_key = key
        self.update_active_label_ui()

    def update_active_label_ui(self):
        class_name, _, color = LABEL_CONFIG[self.active_key]
        for k, btn in self.label_buttons.items():
            if k == self.active_key:
                btn.config(relief=tk.SUNKEN, bd=4)
            else:
                btn.config(relief=tk.RAISED, bd=1)

        self.lbl_active_status.config(
            text=f"Đang chọn: [{self.active_key}] {class_name}", fg=color
        )

    def load_raw_folder(self):
        if os.path.exists(RAW_DIR):
            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(exts)]
            if self.image_files:
                self.current_img_idx = 0
                self.load_current_image()
            else:
                self.lbl_file_info.config(text="Thư mục 'dataset/raw' đang trống!")

    def load_current_image(self):
        if not self.image_files:
            return

        raw_name = self.image_files[self.current_img_idx]
        self.lbl_file_info.config(text=f"File [{self.current_img_idx + 1}/{len(self.image_files)}]: {raw_name}")

        img_path = os.path.join(RAW_DIR, raw_name)
        self.current_img = cv2.imread(img_path)
        if self.current_img is None:
            return

        self.saved_count = 0
        self.crop_history.clear()
        self.display_image()

    def display_image(self):
        if self.current_img is None:
            return

        h_orig, w_orig = self.current_img.shape[:2]
        
        canvas_w = self.canvas.winfo_width() or 1200
        canvas_h = self.canvas.winfo_height() or 750

        # Giữ nguyên tỷ lệ chiều rộng/chiều cao gốc của ảnh
        scale = min(canvas_w / float(w_orig), canvas_h / float(h_orig))
        disp_w = max(1, int(w_orig * scale))
        disp_h = max(1, int(h_orig * scale))

        self.display_scale_x = w_orig / float(disp_w)
        self.display_scale_y = h_orig / float(disp_h)

        rgb_img = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    # -------------------------------------------------------------------
    # XỬ LÝ SỰ KIỆN KÉO THẢ MOUSE & LƯU VÀO HISTORY
    # -------------------------------------------------------------------
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)

        _, _, color = LABEL_CONFIG[self.active_key]
        self.temp_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline=color, width=2)

    def on_move_press(self, event):
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.temp_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = event.x, event.y

        # Tọa độ thực trên ảnh gốc
        x1 = int(min(self.start_x, end_x) * self.display_scale_x)
        y1 = int(min(self.start_y, end_y) * self.display_scale_y)
        x2 = int(max(self.start_x, end_x) * self.display_scale_x)
        y2 = int(max(self.start_y, end_y) * self.display_scale_y)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            self.canvas.delete(self.temp_rect)
            return

        crop_patch = self.current_img[y1:y2, x1:x2]
        if crop_patch.size == 0:
            return

        # Resize patch về 32x32 px
        patch_32 = cv2.resize(crop_patch, (32, 32), interpolation=cv2.INTER_AREA)

        class_name, folder_name, color = LABEL_CONFIG[self.active_key]
        raw_base_name = os.path.splitext(self.image_files[self.current_img_idx])[0]
        
        self.saved_count += 1
        filename = f"{raw_base_name}_{class_name.lower()}_{self.saved_count:03d}.jpg"
        save_path = os.path.join(TRAIN_DIR, folder_name, filename)

        # Lưu ảnh đĩa cứng
        cv2.imwrite(save_path, patch_32)

        # Vẽ khung cố định đánh dấu trên màn hình
        rect_id = self.canvas.create_rectangle(
            min(self.start_x, end_x), min(self.start_y, end_y),
            max(self.start_x, end_x), max(self.start_y, end_y),
            outline=color, width=2
        )
        text_id = self.canvas.create_text(
            min(self.start_x, end_x) + 15, min(self.start_y, end_y) - 8,
            text=class_name, fill=color, font=("Arial", 8, "bold")
        )

        # Lưu thông tin ô vừa khoanh vào lịch sử để cho phép xóa
        self.crop_history.append({
            'file_path': save_path,
            'rect_id': rect_id,
            'text_id': text_id
        })

    # -------------------------------------------------------------------
    # HÀM XÓA Ô VỪA CẮT (UNDO)
    # -------------------------------------------------------------------
    def undo_last_crop(self):
        """Xóa file đĩa cứng và xóa khung hiển thị của ô vừa khoanh gần nhất"""
        if not self.crop_history:
            return

        last_crop = self.crop_history.pop()
        file_path = last_crop['file_path']
        rect_id = last_crop['rect_id']
        text_id = last_crop['text_id']

        # 1. Xóa file khỏi ổ đĩa
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Lỗi không thể xóa file: {e}")

        # 2. Xóa khung vẽ trên giao diện
        if rect_id:
            self.canvas.delete(rect_id)
        if text_id:
            self.canvas.delete(text_id)

    def prev_image(self):
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_img_idx < len(self.image_files) - 1:
            self.current_img_idx += 1
            self.load_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = DragLabelerApp(root)
    root.mainloop()