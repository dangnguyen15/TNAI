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
    predict_bubbles_with_cnn,
    extract_student_information,
)


class ModernOMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TNAI OMR - Hệ Thống Scan & Chấm Thi Trắc Nghiệm Tự Động")
        self.root.geometry("1600x920")
        self.root.minsize(1200, 720)
        self.root.configure(bg="#1e1e2e")

        self.scanned_img_cv = None
        self.roi_img_cv = None
        self.grid_img_cv = None
        self.bubbles_list = []
        self.recognition_results = []
        self.student_info = {}
        self.info_crops = {}
        self._photo_refs = []

        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#2b2b3b", height=65, pady=8, padx=15)
        header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            header, text="🎯 TNAI OMR SYSTEM",
            font=("Segoe UI", 14, "bold"),
            fg="#2ecc71", bg="#2b2b3b"
        ).pack(side=tk.LEFT, padx=10)

        self._button(header, "📁 Chọn Ảnh Bài Thi", self.load_image,
                     "#2ecc71").pack(side=tk.LEFT, padx=6)
        self._button(header, "🤖 Chấm Thi CNN", self.run_recognition,
                     "#9b59b6").pack(side=tk.LEFT, padx=6)
        self._button(header, "📊 Xuất CSV", self.export_csv,
                     "#3498db").pack(side=tk.LEFT, padx=6)

        self.lbl_status = tk.Label(
            header, text="Trạng thái: Sẵn sàng...",
            font=("Segoe UI", 10, "italic"),
            fg="#a6adc8", bg="#2b2b3b"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=12)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg="#1e1e2e")
        right = tk.Frame(main, bg="#181825", width=570)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.nb_images = ttk.Notebook(left)
        self.nb_images.pack(fill=tk.BOTH, expand=True)

        self.tab_grid = tk.Frame(self.nb_images, bg="#181825")
        self.tab_roi = tk.Frame(self.nb_images, bg="#181825")
        self.tab_scanned = tk.Frame(self.nb_images, bg="#181825")
        self.tab_info = tk.Frame(self.nb_images, bg="#181825")

        self.nb_images.add(self.tab_grid, text=" 🟢 Khung 300 Ô ")
        self.nb_images.add(self.tab_roi, text=" 📐 Bảng trắc nghiệm ")
        self.nb_images.add(self.tab_scanned, text=" 📄 Tờ giấy scan ")
        self.nb_images.add(self.tab_info, text=" 🪪 Vùng thông tin OCR ")

        self.lbl_canvas_grid = self._image_label(self.tab_grid)
        self.lbl_canvas_roi = self._image_label(self.tab_roi)
        self.lbl_canvas_scanned = self._image_label(self.tab_scanned)

        self.info_text = tk.Text(
            self.tab_info, wrap=tk.WORD, font=("Segoe UI", 11),
            bg="#11111b", fg="#cdd6f4", insertbackground="white",
            relief=tk.FLAT, padx=15, pady=12
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.info_text.insert(
            "1.0",
            "Thông tin sinh viên và bài thi sẽ hiển thị tại đây.\n\n"
            "Lưu ý: OCR chữ viết tay có thể cần kiểm tra thủ công."
        )
        self.info_text.configure(state=tk.DISABLED)

        # RIGHT: thống kê
        summary = tk.LabelFrame(
            right, text=" 📈 Thống Kê Bài Làm ",
            font=("Segoe UI", 10, "bold"),
            fg="#cdd6f4", bg="#181825", pady=5, padx=10
        )
        summary.pack(fill=tk.X, padx=8, pady=5)

        self.lbl_stat_done = self._stat_label(summary, "Đã trả lời: 0/60", "#2ecc71")
        self.lbl_stat_empty = self._stat_label(summary, "Bỏ trống: 0", "#f1c40f")
        self.lbl_stat_invalid = self._stat_label(summary, "Tô phạm quy (>1 ô): 0", "#e74c3c")

        table = tk.LabelFrame(
            right, text=" 📋 Kết Quả Nhận Dạng 300 Ô (60 Câu) ",
            font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825"
        )
        table.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        columns = ("question", "detail", "answer", "status")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=20)
        for key, title in [
            ("question", "Câu"),
            ("detail", "Chi tiết A B C D E"),
            ("answer", "Đáp án"),
            ("status", "Trạng thái")
        ]:
            self.tree.heading(key, text=title)

        self.tree.column("question", width=55, anchor="center")
        self.tree.column("detail", width=275, anchor="w")
        self.tree.column("answer", width=65, anchor="center")
        self.tree.column("status", width=130, anchor="center")

        scroll = ttk.Scrollbar(table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    @staticmethod
    def _button(parent, text, command, color):
        return tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 10, "bold"),
            bg=color, fg="white", padx=12, pady=6,
            relief=tk.FLAT, cursor="hand2"
        )

    @staticmethod
    def _image_label(parent):
        label = tk.Label(parent, bg="#11111b")
        label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return label

    @staticmethod
    def _stat_label(parent, text, color):
        label = tk.Label(
            parent, text=text, font=("Segoe UI", 9),
            fg=color, bg="#181825"
        )
        label.pack(anchor="w")
        return label

    def cv2_to_imagetk(self, cv_img, max_w=900, max_h=760):
        if cv_img is None:
            return None
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        resized = pil.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS
        )
        return ImageTk.PhotoImage(resized)

    def _show_image(self, label, image):
        photo = self.cv2_to_imagetk(image)
        if photo is not None:
            label.configure(image=photo)
            label.image = photo
            self._photo_refs.append(photo)

    def _show_info(self, values, ocr_ready):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "THÔNG TIN NHẬN DIỆN\n")
        self.info_text.insert(tk.END, "=" * 55 + "\n\n")

        for field, value in values.items():
            shown = value if value else "(Chưa đọc được / để trống)"
            self.info_text.insert(tk.END, f"{field:<20}: {shown}\n")

        self.info_text.insert(tk.END, "\n" + "=" * 55 + "\n")
        if ocr_ready:
            self.info_text.insert(
                tk.END,
                "OCR đã chạy. Hãy đối chiếu kết quả với ảnh gốc, đặc biệt "
                "là họ tên, MSSV, mã đề và chữ viết tay.\n"
            )
        else:
            self.info_text.insert(
                tk.END,
                "Chưa có OCR hoặc Tesseract chưa được cấu hình. "
                "Các vùng thông tin vẫn được cắt để kiểm tra.\n"
            )
        self.info_text.configure(state=tk.DISABLED)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh phiếu thi",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        try:
            self.lbl_status.config(text="Đang hiệu chỉnh trang giấy...")
            self.root.update_idletasks()

            self.scanned_img_cv = scan_and_flatten_paper(path)
            self.roi_img_cv = extract_exact_table_roi(self.scanned_img_cv)
            self.bubbles_list, self.grid_img_cv = extract_and_draw_300_bubbles(
                self.roi_img_cv
            )

            self.student_info, self.info_crops, ocr_ready = (
                extract_student_information(self.scanned_img_cv)
            )

            self._show_image(self.lbl_canvas_grid, self.grid_img_cv)
            self._show_image(self.lbl_canvas_roi, self.roi_img_cv)
            self._show_image(self.lbl_canvas_scanned, self.scanned_img_cv)
            self._show_info(self.student_info, ocr_ready)

            self.lbl_status.config(
                text=f"🟢 Đã định vị {len(self.bubbles_list)}/300 ô"
            )
            self.nb_images.select(self.tab_grid)
            self.run_recognition()
        except Exception as exc:
            messagebox.showerror(
                "Lỗi xử lý ảnh",
                f"Không thể xử lý ảnh bài thi:\n{exc}"
            )
            self.lbl_status.config(text="🔴 Lỗi xử lý!")

    def run_recognition(self):
        if not self.bubbles_list:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh bài thi trước!")
            return

        try:
            self.lbl_status.config(text="Đang chạy CNN nhận diện 300 ô...")
            self.root.update_idletasks()

            self.recognition_results = predict_bubbles_with_cnn(self.bubbles_list)

            for row in self.tree.get_children():
                self.tree.delete(row)

            done = empty = invalid = 0
            for result in self.recognition_results:
                status = result["status"]
                if status == "Hợp lệ":
                    done += 1
                elif status == "Bỏ câu":
                    empty += 1
                else:
                    invalid += 1

                self.tree.insert(
                    "", tk.END,
                    values=(
                        f"Câu {result['question']}",
                        result["detail_str"],
                        result["answer"],
                        status
                    )
                )

            self.lbl_stat_done.config(text=f"Đã trả lời: {done}/60")
            self.lbl_stat_empty.config(text=f"Bỏ trống: {empty}")
            self.lbl_stat_invalid.config(text=f"Tô phạm quy (>1 ô): {invalid}")
            self.lbl_status.config(text="🟢 Hoàn tất nhận diện 60 câu!")
        except Exception as exc:
            messagebox.showerror("Lỗi CNN", str(exc))
            self.lbl_status.config(text="🔴 Lỗi nhận diện!")

    def export_csv(self):
        if not self.recognition_results:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất CSV!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not save_path:
            return

        info_path = os.path.splitext(save_path)[0] + "_thong_tin.csv"
        try:
            # CSV đáp án: giữ cấu trúc cũ và thêm trạng thái từng ô.
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Câu hỏi", "Nhóm cột", "Ô A", "Ô B", "Ô C", "Ô D", "Ô E",
                    "Đáp án chốt", "Trạng thái"
                ])
                for item in self.recognition_results:
                    d = item["details"]
                    writer.writerow([
                        item["question"], item["col"],
                        d.get("A", "Trắng"), d.get("B", "Trắng"),
                        d.get("C", "Trắng"), d.get("D", "Trắng"),
                        d.get("E", "Trắng"), item["answer"], item["status"]
                    ])

            # CSV thông tin: một dòng, thuận tiện nhập vào hệ thống quản lý.
            with open(info_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["Trường thông tin", "Giá trị OCR"])
                for field, value in self.student_info.items():
                    writer.writerow([field, value])

            messagebox.showinfo(
                "Xuất thành công",
                f"Đã xuất kết quả:\n{save_path}\n\nThông tin OCR:\n{info_path}"
            )
        except Exception as exc:
            messagebox.showerror("Lỗi xuất CSV", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernOMRApp(root)
    root.mainloop()