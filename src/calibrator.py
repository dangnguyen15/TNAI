import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from processor import scan_and_flatten_paper, extract_exact_table_roi


class OMRCalibratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI OMR - Tool Cân Chỉnh 4 Cột Độc Lập (Per-Column Calibrator)")
        self.root.geometry("1600x900")
        self.root.configure(bg="#1e1e2e")

        self.table_roi = None
        self.canvas_w = 900
        self.canvas_h = 750

        # --- TỌA ĐỘ TRỤC Y CHUNG ---
        self.var_y_top = tk.IntVar(value=22)
        self.var_y_bottom = tk.IntVar(value=38)
        self.var_box_w = tk.IntVar(value=28)
        self.var_box_h = tk.IntVar(value=28)

        # --- TỌA ĐỘ TRỤC X ĐỘC LẬP CHO 4 CỘT ---
        self.col_vars = []
        default_ratios = [
            [0.380, 0.505, 0.630, 0.755, 0.880],  # Cột 1
            [0.395, 0.520, 0.645, 0.770, 0.895],  # Cột 2
            [0.388, 0.513, 0.638, 0.763, 0.888],  # Cột 3
            [0.380, 0.505, 0.630, 0.755, 0.880]   # Cột 4
        ]

        for col in range(4):
            vars_for_col = [tk.DoubleVar(value=default_ratios[col][i]) for i in range(5)]
            self.col_vars.append(vars_for_col)

        self.setup_ui()

    def setup_ui(self):
        # HEADER
        header_frame = tk.Frame(self.root, bg="#2b2b3b", height=50, pady=5, padx=15)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        lbl_title = tk.Label(
            header_frame, text="🛠️ TOOL CÂN CHỈNH TỌA ĐỘ 4 CỘT ĐỘC LẬP", 
            font=("Segoe UI", 12, "bold"), fg="#f39c12", bg="#2b2b3b"
        )
        lbl_title.pack(side=tk.LEFT, padx=10)

        btn_load = tk.Button(
            header_frame, text="📁 Chọn Ảnh Nạp ROI", command=self.load_image,
            font=("Segoe UI", 10, "bold"), bg="#2ecc71", fg="white", padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
        )
        btn_load.pack(side=tk.LEFT, padx=10)

        btn_copy = tk.Button(
            header_frame, text="📋 Export Mã Nguồn Python", command=self.export_code,
            font=("Segoe UI", 10, "bold"), bg="#e67e22", fg="white", padx=10, pady=4, relief=tk.FLAT, cursor="hand2"
        )
        btn_copy.pack(side=tk.LEFT, padx=10)

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT CONTROL PANEL
        ctrl_frame = tk.Frame(main_paned, bg="#181825", width=440)
        main_paned.add(ctrl_frame, weight=1)

        canvas_ctrl = tk.Canvas(ctrl_frame, bg="#181825", highlightthickness=0)
        scrollbar_ctrl = ttk.Scrollbar(ctrl_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
        scrollable_frame = tk.Frame(canvas_ctrl, bg="#181825")

        scrollable_frame.bind("<Configure>", lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all")))
        canvas_ctrl.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_ctrl.configure(yscrollcommand=scrollbar_ctrl.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ctrl.pack(side=tk.RIGHT, fill=tk.Y)

        # Trục Y
        group_y = tk.LabelFrame(scrollable_frame, text=" 📐 Trục Y & Ô Kích Thước ", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825", padx=10, pady=5)
        group_y.pack(fill=tk.X, padx=10, pady=5)

        self.create_slider(group_y, "Lề Trên Top (px):", self.var_y_top, 0, 100, self.update_preview)
        self.create_slider(group_y, "Lề Dưới Bottom (px):", self.var_y_bottom, 0, 100, self.update_preview)
        self.create_slider(group_y, "Chiều Rộng W (px):", self.var_box_w, 15, 45, self.update_preview)
        self.create_slider(group_y, "Chiều Cao H (px):", self.var_box_h, 15, 45, self.update_preview)

        # TAB CONTROL 4 CỘT
        group_x = tk.LabelFrame(scrollable_frame, text=" 📐 Căn Chỉnh Trục X Từng Cột ", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825", padx=5, pady=5)
        group_x.pack(fill=tk.X, padx=10, pady=5)

        nb_cols = ttk.Notebook(group_x)
        nb_cols.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        options_labels = ['a', 'b', 'c', 'd', 'e']
        for col_i in range(4):
            tab = tk.Frame(nb_cols, bg="#181825", padx=5, pady=5)
            nb_cols.add(tab, text=f" Cột {col_i+1} ")

            for opt_i in range(5):
                self.create_slider_double(
                    tab, f"Ô ({options_labels[opt_i]}):", 
                    self.col_vars[col_i][opt_i], 0.200, 0.980, self.update_preview
                )

        # Text Code
        group_code = tk.LabelFrame(scrollable_frame, text=" 💻 Mã Nguồn Xuất Ra ", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825", padx=5, pady=5)
        group_code.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.txt_code = tk.Text(group_code, height=9, font=("Consolas", 8), bg="#11111b", fg="#a6e3a1")
        self.txt_code.pack(fill=tk.BOTH, expand=True)

        # RIGHT CANVAS DISPLAY
        right_frame = tk.Frame(main_paned, bg="#11111b")
        main_paned.add(right_frame, weight=3)

        self.main_canvas = tk.Canvas(right_frame, bg="#11111b", highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<Configure>", self.on_canvas_resize)

    def create_slider(self, parent, label, var, from_val, to_val, callback):
        frame = tk.Frame(parent, bg="#181825")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, width=18, anchor="w", fg="#cdd6f4", bg="#181825", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Scale(frame, from_=from_val, to=to_val, variable=var, command=lambda e: callback()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Label(frame, textvariable=var, width=4, fg="#f39c12", bg="#181825", font=("Segoe UI", 8, "bold")).pack(side=tk.RIGHT)

    def create_slider_double(self, parent, label, var, from_val, to_val, callback):
        frame = tk.Frame(parent, bg="#181825")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, width=14, anchor="w", fg="#cdd6f4", bg="#181825", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Scale(frame, from_=from_val, to=to_val, variable=var, command=lambda e: callback()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        lbl_val = tk.Label(frame, text="", width=6, fg="#f39c12", bg="#181825", font=("Segoe UI", 8, "bold"))
        lbl_val.pack(side=tk.RIGHT)

        def update_lbl(*args):
            lbl_val.config(text=f"{var.get():.3f}")
        var.trace_add("write", update_lbl)
        update_lbl()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.heic")])
        if not file_path:
            return
        try:
            scanned = scan_and_flatten_paper(file_path)
            self.table_roi = extract_exact_table_roi(scanned)
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc ảnh:\n{str(e)}")

    def on_canvas_resize(self, event):
        if event.width > 50 and event.height > 50:
            self.canvas_w = event.width
            self.canvas_h = event.height
            self.update_preview()

    def update_preview(self):
        if self.table_roi is None:
            return

        overlay = self.table_roi.copy()
        roi_h, roi_w = overlay.shape[:2]

        y_top = self.var_y_top.get()
        y_bottom = self.var_y_bottom.get()
        box_w = self.var_box_w.get()
        box_h = self.var_box_h.get()

        group_w = roi_w / 4.0
        active_h = max(10, roi_h - y_top - y_bottom)
        row_h = active_h / 15.0

        col_matrix = []
        for col_i in range(4):
            ratios = [self.col_vars[col_i][opt_i].get() for opt_i in range(5)]
            col_matrix.append(ratios)

            group_x1 = col_i * group_w
            for row_i in range(15):
                cy = int(y_top + (row_i + 0.5) * row_h)
                for opt_i in range(5):
                    cx = int(group_x1 + group_w * ratios[opt_i])
                    bx1 = max(0, cx - box_w // 2)
                    bx2 = min(roi_w, cx + box_w // 2)
                    by1 = max(0, cy - box_h // 2)
                    by2 = min(roi_h, cy + box_h // 2)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        # Render Auto-Fit
        rgb_img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        w_p, h_p = pil_img.size
        scale = min((self.canvas_w - 20) / w_p, (self.canvas_h - 20) / h_p)
        scale = max(0.1, scale)
        new_w, new_h = int(w_p * scale), int(h_p * scale)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized)

        self.main_canvas.delete("all")
        self.main_canvas.create_image(self.canvas_w // 2, self.canvas_h // 2, anchor=tk.CENTER, image=img_tk)
        self.main_canvas.image = img_tk

        # Xuất mã nguồn
        matrix_str = "col_opt_x_ratios = [\n"
        for i, r in enumerate(col_matrix):
            matrix_str += f"    [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}, {r[3]:.3f}, {r[4]:.3f}]" + (",\n" if i < 3 else "\n")
        matrix_str += "]"

        code_str = f"y_margin_top = {y_top}\ny_margin_bottom = {y_bottom}\nbox_w = {box_w}\nbox_h = {box_h}\n{matrix_str}"
        self.txt_code.delete("1.0", tk.END)
        self.txt_code.insert(tk.END, code_str)

    def export_code(self):
        code_content = self.txt_code.get("1.0", tk.END).strip()
        if not code_content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(code_content)
        messagebox.showinfo("Thành công", "Đã copy thông số 4 cột vào Clipboard!\nBạn mở processor.py và dán đè vào.")


if __name__ == "__main__":
    root = tk.Tk()
    app = OMRCalibratorApp(root)
    root.mainloop()