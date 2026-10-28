import os
import csv
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from processor import (
    scan_and_flatten_paper,
    extract_exact_table_roi,
    extract_and_draw_300_bubbles,
    predict_bubbles_with_cnn
)

class ModernOMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI OMR - Hệ Thống Scan & Chấm Thi Trắc Nghiệm Tự Động")
        self.root.geometry("1550x880")
        self.root.configure(bg="#1e1e2e")

        self.scanned_img_cv = None
        self.roi_img_cv = None
        self.grid_img_cv = None
        self.bubbles_list = []
        self.recognition_results = []

        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg="#2b2b3b", height=65, pady=8, padx=15)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        lbl_title = tk.Label(
            header_frame, 
            text="🎯 TNAI OMR SYSTEM", 
            font=("Segoe UI", 14, "bold"), 
            fg="#2ecc71", bg="#2b2b3b"
        )
        lbl_title.pack(side=tk.LEFT, padx=10)

        btn_load = tk.Button(
            header_frame, text="📁 Chọn Ảnh Bài Thi", command=self.load_image,
            font=("Segoe UI", 10, "bold"), bg="#2ecc71", fg="white",
            padx=12, pady=6, relief=tk.FLAT, cursor="hand2"
        )
        btn_load.pack(side=tk.LEFT, padx=10)

        btn_process = tk.Button(
            header_frame, text="🤖 Chấm Thi CNN", command=self.run_recognition,
            font=("Segoe UI", 10, "bold"), bg="#9b59b6", fg="white",
            padx=12, pady=6, relief=tk.FLAT, cursor="hand2"
        )
        btn_process.pack(side=tk.LEFT, padx=5)

        btn_export = tk.Button(
            header_frame, text="📊 Xuất CSV Chi Tiết", command=self.export_csv,
            font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white",
            padx=12, pady=6, relief=tk.FLAT, cursor="hand2"
        )
        btn_export.pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(
            header_frame, text="Trạng thái: Sẵn sàng...",
            font=("Segoe UI", 10, "italic"), fg="#a6adc8", bg="#2b2b3b"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=15)

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL
        left_frame = tk.Frame(main_paned, bg="#1e1e2e")
        main_paned.add(left_frame, weight=3)

        nb_images = ttk.Notebook(left_frame)
        nb_images.pack(fill=tk.BOTH, expand=True)

        self.tab_grid = tk.Frame(nb_images, bg="#181825")
        nb_images.add(self.tab_grid, text=" 🟢 Khung 300 Ô Trắc Nghiệm ")
        self.lbl_canvas_grid = tk.Label(self.tab_grid, bg="#11111b")
        self.lbl_canvas_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_roi = tk.Frame(nb_images, bg="#181825")
        nb_images.add(self.tab_roi, text=" 📐 Bảng ROI Nắn Phẳng ")
        self.lbl_canvas_roi = tk.Label(self.tab_roi, bg="#11111b")
        self.lbl_canvas_roi.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_scanned = tk.Frame(nb_images, bg="#181825")
        nb_images.add(self.tab_scanned, text=" 📄 Tờ Giấy Scan ")
        self.lbl_canvas_scanned = tk.Label(self.tab_scanned, bg="#11111b")
        self.lbl_canvas_scanned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # RIGHT PANEL: DISPLAY DETAILED RECOGNITION RESULTS
        right_frame = tk.Frame(main_paned, bg="#181825", width=520)
        main_paned.add(right_frame, weight=2)

        summary_frame = tk.LabelFrame(
            right_frame, text=" 📈 Thống Kê Bài Làm ",
            font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825", pady=5, padx=10
        )
        summary_frame.pack(fill=tk.X, padx=8, pady=5)

        self.lbl_stat_done = tk.Label(summary_frame, text="Đã trả lời: 0/60", font=("Segoe UI", 9), fg="#2ecc71", bg="#181825")
        self.lbl_stat_done.pack(anchor="w")

        self.lbl_stat_empty = tk.Label(summary_frame, text="Bỏ trống: 0", font=("Segoe UI", 9), fg="#f1c40f", bg="#181825")
        self.lbl_stat_empty.pack(anchor="w")

        self.lbl_stat_invalid = tk.Label(summary_frame, text="Tô phạm quy (>1 ô): 0", font=("Segoe UI", 9), fg="#e74c3c", bg="#181825")
        self.lbl_stat_invalid.pack(anchor="w")

        table_frame = tk.LabelFrame(
            right_frame, text=" 📋 Kết Quả Nhận Dạng 300 Ô (60 Câu) ",
            font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825"
        )
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        # Cột Treeview bổ sung 'detail' hiển thị rõ nhận diện 5 ô A, B, C, D, E
        columns = ("question", "detail", "answer", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        self.tree.heading("question", text="Câu")
        self.tree.heading("detail", text="Chi Tiết Nhận Dạng 5 Ô (A B C D E)")
        self.tree.heading("answer", text="Đ.Án Chốt")
        self.tree.heading("status", text="Trạng Thái")

        self.tree.column("question", width=60, anchor="center")
        self.tree.column("detail", width=270, anchor="w")
        self.tree.column("answer", width=75, anchor="center")
        self.tree.column("status", width=105, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def cv2_to_imagetk(self, cv_img, max_w, max_h):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        w, h = pil_img.size
        scale = min(max_w / w, max_h / h) if w > 0 and h > 0 else 1.0
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
            self.lbl_status.config(text="Đang xử lý nắn phẳng và định vị 300 ô...")
            self.root.update()

            self.scanned_img_cv = scan_and_flatten_paper(file_path)
            self.roi_img_cv = extract_exact_table_roi(self.scanned_img_cv)
            self.bubbles_list, self.grid_img_cv = extract_and_draw_300_bubbles(self.roi_img_cv)

            img_tk_grid = self.cv2_to_imagetk(self.grid_img_cv, 800, 650)
            img_tk_roi = self.cv2_to_imagetk(self.roi_img_cv, 800, 650)
            img_tk_scanned = self.cv2_to_imagetk(self.scanned_img_cv, 800, 650)

            self.lbl_canvas_grid.config(image=img_tk_grid)
            self.lbl_canvas_grid.image = img_tk_grid

            self.lbl_canvas_roi.config(image=img_tk_roi)
            self.lbl_canvas_roi.image = img_tk_roi

            self.lbl_canvas_scanned.config(image=img_tk_scanned)
            self.lbl_canvas_scanned.image = img_tk_scanned

            filename = os.path.basename(file_path)
            self.lbl_status.config(text=f"🟢 Đã định vị 300 ô: {filename}")

            self.run_recognition()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh bài thi:\n{str(e)}")
            self.lbl_status.config(text="🔴 Lỗi xử lý!")

    def run_recognition(self):
        if not self.bubbles_list:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh bài thi trước!")
            return

        self.lbl_status.config(text="Đang chạy AI suy luận kết quả...")
        self.root.update()

        self.recognition_results = predict_bubbles_with_cnn(self.bubbles_list)

        for row in self.tree.get_children():
            self.tree.delete(row)

        cnt_done = 0
        cnt_empty = 0
        cnt_invalid = 0

        for res in self.recognition_results:
            q = res['question']
            detail_str = res['detail_str']
            ans = res['answer']
            st = res['status']

            if st == "Hợp lệ":
                cnt_done += 1
            elif st == "Bỏ câu":
                cnt_empty += 1
            else:
                cnt_invalid += 1

            self.tree.insert("", tk.END, values=(f"Câu {q}", detail_str, ans, st))

        self.lbl_stat_done.config(text=f"Đã trả lời: {cnt_done}/60")
        self.lbl_stat_empty.config(text=f"Bỏ trống: {cnt_empty}")
        self.lbl_stat_invalid.config(text=f"Tô phạm quy (>1 ô): {cnt_invalid}")

        self.lbl_status.config(text="🟢 Hoàn tất chấm thi 60 câu!")

    def export_csv(self):
        if not self.recognition_results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất CSV!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")]
        )
        if save_path:
            with open(save_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["Câu Hỏi", "Nhóm Cột", "Ô A", "Ô B", "Ô C", "Ô D", "Ô E", "Đáp Án Chốt", "Trạng Thái"])
                for item in self.recognition_results:
                    d = item['details']
                    writer.writerow([
                        item['question'],
                        item['col'],
                        d.get('A', ''),
                        d.get('B', ''),
                        d.get('C', ''),
                        d.get('D', ''),
                        d.get('E', ''),
                        item['answer'],
                        item['status']
                    ])

            messagebox.showinfo("Thành công", f"Đã xuất báo cáo ra file:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernOMRApp(root)
    root.mainloop()