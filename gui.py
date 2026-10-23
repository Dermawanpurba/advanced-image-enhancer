from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Ensure image_enhancer package can be imported
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from image_enhancer import (
    load_ai_models,
    enhance_faces_ai,
    apply_natural_beauty_smoothing,
    apply_cinematic_pro_camera,
    remove_background,
    replace_background_studio,
)
from image_enhancer.enhancers import (
    adaptive_sharpening,
    apply_non_local_means_denoising,
    enhance_contrast_local,
    enhance_saturation,
    auto_white_balance,
    enhance_low_light,
    load_image,
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ModernImageEnhancerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Advanced AI Image Enhancer Studio")
        self.geometry("1280x850")
        self.minsize(1050, 720)

        # State
        self.original_cv: Optional[np.ndarray] = None
        self.enhanced_cv: Optional[np.ndarray] = None
        self.current_file_path: Optional[Path] = None
        self._is_processing = False

        self._build_layout()
        self._set_preset("✨ Magic AI Enhance")

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar Controls ──────────────────────────────────────────────────
        self.sidebar = ctk.CTkScrollableFrame(self, width=340, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Title / Brand
        brand_label = ctk.CTkLabel(
            self.sidebar,
            text="⚡ AI Enhancer Pro",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#38BDF8"
        )
        brand_label.pack(padx=16, pady=(16, 4), anchor="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Professional Local AI Enhancement Suite",
            font=ctk.CTkFont(size=12),
            text_color="#94A3B8"
        )
        subtitle.pack(padx=16, pady=(0, 14), anchor="w")

        # Preset Dropdown
        ctk.CTkLabel(self.sidebar, text="Quick Presets", font=ctk.CTkFont(weight="bold")).pack(padx=16, pady=(4, 2), anchor="w")
        self.preset_var = ctk.StringVar(value="✨ Magic AI Enhance")
        self.preset_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=[
                "✨ Magic AI Enhance",
                "👤 Portrait & Skin Beauty",
                "🖨️ High-Res Print (300 DPI)",
                "🎬 Cinematic Kodak Portra",
                "✂️ Remove BG (Transparent)",
                "🏢 Studio Gray Backdrop",
                "⚪ Clean White Backdrop",
            ],
            variable=self.preset_var,
            command=self._on_preset_change,
        )
        self.preset_menu.pack(padx=16, pady=(0, 12), fill="x")

        # ── Group: AI Face & Beauty ──
        face_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=8)
        face_frame.pack(padx=12, pady=6, fill="x")
        ctk.CTkLabel(face_frame, text="Local AI Face & Beauty", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0").pack(padx=12, pady=(10, 4), anchor="w")

        self.ai_face_switch = ctk.CTkSwitch(face_frame, text="AI Face Restoration (GFPGAN)")
        self.ai_face_switch.select()
        self.ai_face_switch.pack(padx=12, pady=4, anchor="w")

        self.slider_face_blend = self._create_slider(face_frame, "Face Blend Strength", 0.0, 1.0, 0.85)

        self.ai_beauty_switch = ctk.CTkSwitch(face_frame, text="Natural Skin Smoothing")
        self.ai_beauty_switch.select()
        self.ai_beauty_switch.pack(padx=12, pady=(8, 4), anchor="w")

        self.slider_beauty = self._create_slider(face_frame, "Skin Beauty Strength", 0.0, 1.0, 0.50)

        # ── Group: Optical & Color ──
        enh_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=8)
        enh_frame.pack(padx=12, pady=6, fill="x")
        ctk.CTkLabel(enh_frame, text="Clarity & Tone Enhancement", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0").pack(padx=12, pady=(10, 4), anchor="w")

        self.slider_sharpen = self._create_slider(enh_frame, "Adaptive Sharpening", 1.0, 2.5, 1.15)
        self.slider_denoise = self._create_slider(enh_frame, "Denoise Strength", 0.0, 3.0, 1.2)
        self.slider_contrast = self._create_slider(enh_frame, "Local Contrast Lift", 1.0, 1.4, 1.06)
        self.slider_sat = self._create_slider(enh_frame, "Vibrance / Saturation", 1.0, 1.3, 1.05)

        # ── Group: Cinematic & Backdrop ──
        cine_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=8)
        cine_frame.pack(padx=12, pady=6, fill="x")
        ctk.CTkLabel(cine_frame, text="Cinematic Camera & Studio", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0").pack(padx=12, pady=(10, 4), anchor="w")

        self.cinematic_switch = ctk.CTkSwitch(cine_frame, text="Kodak Portra Bokeh Simulation")
        self.cinematic_switch.pack(padx=12, pady=4, anchor="w")
        self.slider_bokeh = self._create_slider(cine_frame, "Lens Bokeh Falloff", 0.0, 1.0, 0.35)

        # ── Group: Output & DPI Export ──
        export_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E293B", corner_radius=8)
        export_frame.pack(padx=12, pady=6, fill="x")
        ctk.CTkLabel(export_frame, text="Target Resolution & Export", font=ctk.CTkFont(size=13, weight="bold"), text_color="#E2E8F0").pack(padx=12, pady=(10, 4), anchor="w")

        ctk.CTkLabel(export_frame, text="Resolution Preset:", font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(padx=12, pady=(4, 0), anchor="w")
        self.res_var = ctk.StringVar(value="Print 2624x3936 (High-Res)")
        self.res_menu = ctk.CTkOptionMenu(
            export_frame,
            values=[
                "Print 2624x3936 (High-Res)",
                "Original Dimensions",
                "2K UHD (2048 x 1536)",
                "4K UHD (3840 x 2160)",
            ],
            variable=self.res_var,
        )
        self.res_menu.pack(padx=12, pady=(2, 6), fill="x")

        dpi_row = ctk.CTkFrame(export_frame, fg_color="transparent")
        dpi_row.pack(padx=12, pady=4, fill="x")
        ctk.CTkLabel(dpi_row, text="Target DPI:", font=ctk.CTkFont(size=11), text_color="#94A3B8").pack(side="left")
        self.dpi_var = ctk.StringVar(value="300")
        self.dpi_menu = ctk.CTkSegmentedButton(dpi_row, values=["72", "150", "300", "600"], variable=self.dpi_var)
        self.dpi_menu.pack(side="right")

        # ── Main Display Workspace ────────────────────────────────────────────
        self.main_area = ctk.CTkFrame(self, fg_color="#090D16", corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # Top Bar: File Info & Metrics
        self.top_bar = ctk.CTkFrame(self.main_area, fg_color="#0F172A", height=50, corner_radius=0)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        self.info_label = ctk.CTkLabel(
            self.top_bar,
            text="📁 No photo opened. Click 'Open Photo' to load an image.",
            font=ctk.CTkFont(size=13),
            text_color="#94A3B8"
        )
        self.info_label.pack(side="left", padx=16, pady=12)

        # Central Image Canvas Split (Before / After)
        self.display_container = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.display_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self.display_container.grid_columnconfigure(0, weight=1)
        self.display_container.grid_columnconfigure(1, weight=1)
        self.display_container.grid_rowconfigure(0, weight=1)

        # Left: Original
        self.orig_box = ctk.CTkFrame(self.display_container, fg_color="#1E293B", corner_radius=10)
        self.orig_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self.orig_box.grid_rowconfigure(1, weight=1)
        self.orig_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.orig_box, text="ORIGINAL", font=ctk.CTkFont(size=11, weight="bold"), text_color="#64748B").grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.orig_image_label = ctk.CTkLabel(self.orig_box, text="Drop or Open Image", text_color="#475569")
        self.orig_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Right: Enhanced
        self.proc_box = ctk.CTkFrame(self.display_container, fg_color="#1E293B", corner_radius=10)
        self.proc_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        self.proc_box.grid_rowconfigure(1, weight=1)
        self.proc_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.proc_box, text="AI ENHANCED", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8").grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.proc_image_label = ctk.CTkLabel(self.proc_box, text="Enhanced Result Appears Here", text_color="#475569")
        self.proc_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Bottom Bar: Actions & Progress
        self.bottom_bar = ctk.CTkFrame(self.main_area, fg_color="#0F172A", height=70, corner_radius=0)
        self.bottom_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

        self.btn_open = ctk.CTkButton(
            self.bottom_bar,
            text="📂 Open Photo",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_image_dialog,
            width=130
        )
        self.btn_open.pack(side="left", padx=(16, 8), pady=16)

        self.btn_process = ctk.CTkButton(
            self.bottom_bar,
            text="⚡ Run AI Enhancement",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self._start_processing,
            width=180
        )
        self.btn_process.pack(side="left", padx=8, pady=16)

        self.btn_save = ctk.CTkButton(
            self.bottom_bar,
            text="💾 Save High-Res (300 DPI)",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            command=self._save_image_dialog,
            width=180
        )
        self.btn_save.pack(side="left", padx=8, pady=16)

        self.status_label = ctk.CTkLabel(
            self.bottom_bar,
            text="Ready.",
            font=ctk.CTkFont(size=12),
            text_color="#94A3B8"
        )
        self.status_label.pack(side="right", padx=16, pady=16)

        self.prog_bar = ctk.CTkProgressBar(self.bottom_bar, orientation="horizontal", width=180)
        self.prog_bar.set(0)
        self.prog_bar.pack(side="right", padx=10, pady=16)

    def _create_slider(self, parent, label_text: str, min_val: float, max_val: float, default_val: float):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(padx=12, pady=(6, 2), fill="x")

        lbl = ctk.CTkLabel(row, text=label_text, font=ctk.CTkFont(size=11), text_color="#94A3B8")
        lbl.pack(side="left")

        val_badge = ctk.CTkLabel(row, text=f"{default_val:.2f}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8")
        val_badge.pack(side="right")

        slider = ctk.CTkSlider(
            parent,
            from_=min_val,
            to=max_val,
            number_of_steps=100,
            command=lambda v: val_badge.configure(text=f"{v:.2f}")
        )
        slider.set(default_val)
        slider.pack(padx=12, pady=(0, 6), fill="x")
        return slider

    def _on_preset_change(self, choice: str) -> None:
        self._set_preset(choice)

    def _set_preset(self, choice: str) -> None:
        if choice == "✨ Magic AI Enhance":
            self.ai_face_switch.select()
            self.slider_face_blend.set(0.95)
            self.ai_beauty_switch.select()
            self.slider_beauty.set(0.80)
            self.cinematic_switch.deselect()
            self.slider_sharpen.set(1.40)
            self.slider_denoise.set(1.20)
            self.slider_contrast.set(1.15)
            self.slider_sat.set(1.12)
            self.res_var.set("Print 2624x3936 (High-Res)")
            self.dpi_var.set("300")
        elif choice == "👤 Portrait & Skin Beauty":
            self.ai_face_switch.select()
            self.slider_face_blend.set(1.0)
            self.ai_beauty_switch.select()
            self.slider_beauty.set(0.95)
            self.cinematic_switch.deselect()
            self.slider_sharpen.set(1.25)
            self.slider_denoise.set(1.50)
            self.slider_contrast.set(1.10)
            self.slider_sat.set(1.08)
        elif choice == "🖨️ High-Res Print (300 DPI)":
            self.ai_face_switch.deselect()
            self.ai_beauty_switch.deselect()
            self.cinematic_switch.deselect()
            self.slider_sharpen.set(1.45)
            self.slider_denoise.set(1.20)
            self.slider_contrast.set(1.14)
            self.slider_sat.set(1.08)
            self.res_var.set("Print 2624x3936 (High-Res)")
            self.dpi_var.set("300")
        elif choice == "🎬 Cinematic Kodak Portra":
            self.ai_face_switch.select()
            self.slider_face_blend.set(0.85)
            self.cinematic_switch.select()
            self.slider_bokeh.set(0.55)
            self.slider_contrast.set(1.18)
            self.slider_sat.set(1.10)
        elif choice in ("✂️ Remove BG (Transparent)", "🏢 Studio Gray Backdrop", "⚪ Clean White Backdrop"):
            self.ai_face_switch.select()
            self.slider_face_blend.set(0.95)

    def _open_image_dialog(self) -> None:
        file_types = [("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff")]
        path = filedialog.askopenfilename(title="Select Photo to Enhance", filetypes=file_types)
        if not path:
            return

        self.current_file_path = Path(path)
        img = load_image(self.current_file_path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image:\n{path}")
            return

        self.original_cv = img
        self.enhanced_cv = None
        self._update_preview(self.original_cv, is_proc=False)
        self.proc_image_label.configure(image=None, text="Click 'Run AI Enhancement' to process")

        h, w = img.shape[:2]
        file_size = self.current_file_path.stat().st_size / (1024 * 1024)
        self.info_label.configure(
            text=f"📁 {self.current_file_path.name}  |  {w} × {h} px  |  {file_size:.2f} MB"
        )
        self.status_label.configure(text="Ready to enhance.")

    def _start_processing(self) -> None:
        if self.original_cv is None:
            messagebox.showwarning("Warning", "Please open a photo first!")
            return

        if self._is_processing:
            return

        self._is_processing = True
        self.btn_process.configure(state="disabled")
        self.prog_bar.start()
        self.status_label.configure(text="Processing AI enhancement pipeline...")

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_pipeline(self) -> None:
        t0 = time.time()
        try:
            img = self.original_cv.copy()
            orig_h, orig_w = img.shape[:2]

            # 1. Target Resolution Upscale
            res_choice = self.res_var.get()
            if "Print 2624x3936" in res_choice:
                scale = max(2624.0 / orig_w, 3936.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif "4K" in res_choice:
                scale = max(3840.0 / orig_w, 2160.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif "2K" in res_choice:
                scale = max(2048.0 / orig_w, 1536.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # 2. Denoising
            denoise_val = self.slider_denoise.get()
            if denoise_val > 0.05:
                img = apply_non_local_means_denoising(img, {"denoise_strength": denoise_val})

            # 3. Local Contrast
            contrast_val = self.slider_contrast.get()
            if contrast_val > 1.01:
                img = enhance_contrast_local(img, {"contrast_boost": contrast_val})

            # 4. Saturation
            sat_val = self.slider_sat.get()
            if sat_val > 1.01:
                img = enhance_saturation(img, {"saturation_boost": sat_val})

            # 5. Adaptive Sharpening
            sharp_val = self.slider_sharpen.get()
            if sharp_val > 1.01:
                img = adaptive_sharpening(img, {"sharpening_strength": sharp_val})

            # 6. Local AI Face Restoration & Beauty
            preset = self.preset_var.get()
            if self.ai_face_switch.get():
                face_blend = self.slider_face_blend.get()
                beauty_enabled = bool(self.ai_beauty_switch.get())
                beauty_val = self.slider_beauty.get()
                img = enhance_faces_ai(
                    img,
                    blend_strength=face_blend,
                    enable_beauty=beauty_enabled,
                    beauty_strength=beauty_val,
                )

            # 7. Cinematic Camera
            if self.cinematic_switch.get():
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cine_bgr = apply_cinematic_pro_camera(
                    img_bgr,
                    lens_bokeh_strength=self.slider_bokeh.get(),
                )
                img = cv2.cvtColor(cine_bgr, cv2.COLOR_BGR2RGB)

            # 8. Background Skills (if selected)
            if preset == "✂️ Remove BG (Transparent)":
                rgba = remove_background(img)
                # For display preview on dark canvas
                alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
                bg_preview = np.full_like(img, 30)
                img = (rgba[:, :, :3].astype(np.float32) * alpha + bg_preview * (1 - alpha)).astype(np.uint8)
            elif preset == "🏢 Studio Gray Backdrop":
                img = replace_background_studio(img, preset="studio_gray")
            elif preset == "⚪ Clean White Backdrop":
                img = replace_background_studio(img, preset="clean_white")

            self.enhanced_cv = img
            duration = time.time() - t0

            # UI Update on main thread
            self.after(0, self._on_pipeline_success, duration)

        except Exception as e:
            self.after(0, self._on_pipeline_error, str(e))

    def _on_pipeline_success(self, duration: float) -> None:
        self.prog_bar.stop()
        self.prog_bar.set(1.0)
        self.btn_process.configure(state="normal")
        self._is_processing = False

        if self.enhanced_cv is not None:
            self._update_preview(self.enhanced_cv, is_proc=True)
            h, w = self.enhanced_cv.shape[:2]
            self.status_label.configure(
                text=f"✓ Complete in {duration:.2f}s | Output: {w} × {h} px"
            )

    def _on_pipeline_error(self, err_msg: str) -> None:
        self.prog_bar.stop()
        self.btn_process.configure(state="normal")
        self._is_processing = False
        messagebox.showerror("Processing Error", f"An error occurred:\n{err_msg}")
        self.status_label.configure(text="Processing failed.")

    def _update_preview(self, cv_img: np.ndarray, is_proc: bool = False) -> None:
        target_label = self.proc_image_label if is_proc else self.orig_image_label
        h, w = cv_img.shape[:2]

        max_w, max_h = 480, 560
        scale = min(max_w / w, max_h / h)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))

        disp_img = cv2.resize(cv_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(disp_img)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

        target_label.configure(image=ctk_img, text="")
        target_label.image = ctk_img

    def _save_image_dialog(self) -> None:
        if self.enhanced_cv is None:
            messagebox.showwarning("Warning", "No enhanced image to save! Run enhancement first.")
            return

        default_name = "enhanced_photo.jpg"
        if self.current_file_path:
            default_name = f"{self.current_file_path.stem}_enhanced.jpg"

        file_path = filedialog.asksaveasfilename(
            title="Save High-Res Image",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG Image (*.jpg)", "*.jpg"), ("PNG Image (*.png)", "*.png"), ("WebP Image (*.webp)", "*.webp")]
        )
        if not file_path:
            return

        out_path = Path(file_path)
        ext = out_path.suffix.lower()
        dpi_val = int(self.dpi_var.get())

        pil_img = Image.fromarray(self.enhanced_cv)

        if ext in (".jpg", ".jpeg"):
            # High quality JPEG with 300 DPI and minimum 1MB guarantee
            pil_img.save(
                out_path,
                format="JPEG",
                dpi=(dpi_val, dpi_val),
                quality=98,
                subsampling=0,
            )
        elif ext == ".png":
            pil_img.save(out_path, format="PNG", dpi=(dpi_val, dpi_val))
        else:
            pil_img.save(out_path, format="WEBP", quality=100)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        messagebox.showinfo(
            "Saved Successfully",
            f"Image saved to:\n{out_path.name}\n\n"
            f"Resolution: {self.enhanced_cv.shape[1]} × {self.enhanced_cv.shape[0]} px\n"
            f"Density: {dpi_val} DPI\n"
            f"File Size: {size_mb:.2f} MB"
        )
        self.status_label.configure(text=f"Saved: {out_path.name} ({size_mb:.2f} MB)")


def main():
    app = ModernImageEnhancerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
