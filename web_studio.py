from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

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

logger = logging.getLogger("web_studio")
logging.basicConfig(level=logging.INFO)

PORT = 8501
SAMPLE_ORIG_WEBP = Path("/Users/mac/Desktop/PROJECT/htdocs/foto final/IMG_1051.WEBP")
SAMPLE_ENH_JPG = Path("/Users/mac/Desktop/PROJECT/htdocs/foto final/IMG_1051.JPG")


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Image Enhancer Studio Pro ⚡</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #090D16;
      --bg-surface: #0F172A;
      --bg-card: #1E293B;
      --bg-card-hover: #27354A;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(56, 189, 248, 0.5);
      --primary: #2563EB;
      --primary-hover: #1D4ED8;
      --accent: #38BDF8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --success: #10B981;
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
      --text-dim: #64748B;
      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    header {
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(12px);
    }

    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #2563EB, #06B6D4);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px rgba(37, 99, 235, 0.4);
    }
    .brand-icon svg { width: 20px; height: 20px; stroke: #fff; }
    .brand-text h1 { font-size: 16px; font-weight: 700; letter-spacing: -0.02em; }
    .brand-text p { font-size: 11px; color: var(--text-muted); }

    .header-actions { display: flex; align-items: center; gap: 14px; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #34D399;
      transition: all 0.2s ease;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10B981;
      box-shadow: 0 0 8px #10B981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(1.2); }
    }

    main {
      flex: 1;
      display: grid;
      grid-template-columns: 360px 1fr;
      height: calc(100vh - 65px);
    }

    .sidebar {
      background: var(--bg-surface);
      border-right: 1px solid var(--border-subtle);
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .section-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 14px;
      transition: border-color 0.2s ease;
    }
    .card:focus-within { border-color: var(--border-focus); }

    .presets-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 8px;
      text-align: left;
      cursor: pointer;
      color: var(--text-main);
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .preset-btn:hover {
      background: var(--bg-card-hover);
      border-color: rgba(56, 189, 248, 0.4);
      transform: translateY(-1px);
    }
    .preset-btn.active {
      background: rgba(37, 99, 235, 0.25);
      border-color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }
    .preset-name { font-size: 12px; font-weight: 700; margin-bottom: 2px; }
    .preset-desc { font-size: 10px; color: var(--text-muted); }

    .control-row { margin-bottom: 12px; }
    .control-row:last-child { margin-bottom: 0; }
    .control-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .control-label { font-size: 12px; font-weight: 600; color: var(--text-main); }
    .control-val {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--accent);
      background: rgba(56, 189, 248, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      min-width: 38px;
      text-align: center;
    }

    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      background: #334155;
      border-radius: 3px;
      outline: none;
      cursor: pointer;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 8px var(--accent);
      transition: transform 0.15s ease;
    }
    input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }

    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 0;
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 38px;
      height: 22px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider-toggle {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      transition: .3s;
      border-radius: 22px;
    }
    .slider-toggle:before {
      position: absolute;
      content: "";
      height: 16px;
      width: 16px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .3s;
      border-radius: 50%;
    }
    input:checked + .slider-toggle { background-color: var(--primary); }
    input:checked + .slider-toggle:before { transform: translateX(16px); }

    select {
      width: 100%;
      background: #090D16;
      border: 1px solid #334155;
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }

    .workspace {
      display: flex;
      flex-direction: column;
      background: var(--bg-base);
      overflow: hidden;
    }

    .toolbar {
      padding: 10px 24px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(8px);
    }
    .meta-badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .meta-badge {
      font-size: 11px;
      font-family: var(--font-mono);
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 4px 10px;
      border-radius: 6px;
      color: var(--text-muted);
    }
    .meta-badge strong { color: var(--text-main); }

    .btn-group { display: flex; align-items: center; gap: 8px; }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .btn-secondary {
      background: var(--bg-card);
      border-color: var(--border-subtle);
      color: var(--text-main);
    }
    .btn-secondary:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-secondary.active {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent);
      color: var(--accent);
    }
    .btn-primary {
      background: linear-gradient(135deg, #2563EB, #1D4ED8);
      color: #fff;
      box-shadow: 0 0 16px rgba(37, 99, 235, 0.3);
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 0 24px rgba(37, 99, 235, 0.5);
    }
    .btn-success {
      background: linear-gradient(135deg, #059669, #047857);
      color: #fff;
      box-shadow: 0 0 16px rgba(5, 150, 105, 0.3);
    }
    .btn-success:hover {
      transform: translateY(-1px);
      box-shadow: 0 0 24px rgba(5, 150, 105, 0.5);
    }

    .canvas-container {
      flex: 1;
      padding: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
      background: radial-gradient(circle at center, rgba(30, 41, 59, 0.5) 0%, rgba(9, 13, 22, 1) 100%);
    }

    /* Comparison Display Modes */
    /* Mode 1: Split Slider */
    .comp-slider-box {
      position: relative;
      max-height: 80vh;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 0 1px var(--border-subtle);
      user-select: none;
      display: inline-block;
    }
    .comp-img {
      display: block;
      max-height: 80vh;
      max-width: 100%;
      object-fit: contain;
    }
    .comp-overlay {
      position: absolute;
      top: 0; left: 0;
      height: 100%;
      width: 50%;
      overflow: hidden;
      border-right: 2px solid #fff;
      box-shadow: 0 0 16px rgba(255, 255, 255, 0.7);
    }
    .comp-overlay-img {
      position: absolute;
      top: 0; left: 0;
      height: 100%;
      max-height: 80vh;
      object-fit: contain;
    }
    .slider-handle {
      position: absolute;
      top: 50%;
      right: -14px;
      transform: translateY(-50%);
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: #fff;
      color: #0F172A;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 12px rgba(0, 0, 0, 0.6);
      cursor: ew-resize;
      font-size: 11px;
      font-weight: 800;
    }

    /* Mode 2: Side-by-Side */
    .side-by-side-box {
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: center;
      max-height: 80vh;
      width: 100%;
    }
    .side-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      max-height: 80vh;
      flex: 1;
      max-width: 48%;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .side-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      align-self: flex-start;
    }
    .side-img {
      max-height: 72vh;
      max-width: 100%;
      object-fit: contain;
      border-radius: 8px;
    }

    .pill-label {
      position: absolute;
      top: 14px;
      padding: 4px 10px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.05em;
      pointer-events: none;
      z-index: 10;
    }
    .pill-label.before { left: 14px; color: #94A3B8; }
    .pill-label.after { right: 14px; color: var(--accent); border-color: rgba(56, 189, 248, 0.4); }

    .dropzone {
      border: 2px dashed #334155;
      border-radius: 16px;
      padding: 48px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      max-width: 460px;
    }
    .dropzone:hover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.04);
    }
    .drop-icon {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: var(--bg-card);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
      color: var(--accent);
    }
    .drop-icon svg { width: 28px; height: 28px; }
    .drop-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
    .drop-subtitle { font-size: 13px; color: var(--text-muted); margin-bottom: 18px; }

    /* Live Update Toast */
    .live-indicator {
      position: absolute;
      bottom: 24px;
      right: 24px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--border-subtle);
      backdrop-filter: blur(8px);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
      z-index: 30;
      transition: opacity 0.3s;
    }
    .live-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(56, 189, 248, 0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Zoom Transition */
    .zoom-active .comp-img,
    .zoom-active .comp-overlay-img,
    .zoom-active .side-img {
      transform: scale(2.2);
      transform-origin: 50% 20%;
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
      </div>
      <div class="brand-text">
        <h1>AI Image Enhancer Studio Pro ⚡</h1>
        <p>Local On-Device AI • GFPGAN • Beauty Retouch • 300 DPI Super-Resolution</p>
      </div>
    </div>
    <div class="header-actions">
      <div class="status-pill" id="pill_status">
        <div class="status-dot"></div>
        <span id="txt_status">Local AI Engine Ready</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Left Sidebar -->
    <div class="sidebar">
      
      <!-- Presets -->
      <div>
        <div class="section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
          Quick Presets (1-Click)
        </div>
        <div class="presets-grid">
          <button class="preset-btn active" onclick="setPreset('magic')">
            <div class="preset-name">✨ Magic AI</div>
            <div class="preset-desc">Full AI Auto Pipeline</div>
          </button>
          <button class="preset-btn" onclick="setPreset('portrait')">
            <div class="preset-name">👤 Beauty Skin</div>
            <div class="preset-desc">Porcelain skin + GFPGAN</div>
          </button>
          <button class="preset-btn" onclick="setPreset('print300')">
            <div class="preset-name">🖨️ Print 300 DPI</div>
            <div class="preset-desc">Min 2624x3936 px, >1MB</div>
          </button>
          <button class="preset-btn" onclick="setPreset('cinematic')">
            <div class="preset-name">🎬 Kodak Bokeh</div>
            <div class="preset-desc">Optical blur & bloom</div>
          </button>
          <button class="preset-btn" onclick="setPreset('remove_bg')">
            <div class="preset-name">✂️ Transparent</div>
            <div class="preset-desc">AI Background cutout</div>
          </button>
          <button class="preset-btn" onclick="setPreset('studio_gray')">
            <div class="preset-name">🏢 Studio Gray</div>
            <div class="preset-desc">Studio backdrop</div>
          </button>
        </div>
      </div>

      <!-- Local AI Face & Beauty -->
      <div class="card">
        <div class="section-title">Local AI Face & Beauty Retouching</div>
        <div class="toggle-row">
          <span class="control-label">GFPGAN Face Restoration</span>
          <label class="switch">
            <input type="checkbox" id="face_enabled" checked onchange="triggerDebounce()">
            <span class="slider-toggle"></span>
          </label>
        </div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Face Blend Strength</span>
            <span class="control-val" id="val_face_blend">0.95</span>
          </div>
          <input type="range" id="face_blend" min="0.1" max="1.0" step="0.05" value="0.95" oninput="onSliderMove('face_blend', this.value)">
        </div>
        <div class="toggle-row">
          <span class="control-label">Natural Skin Beauty</span>
          <label class="switch">
            <input type="checkbox" id="beauty_enabled" checked onchange="triggerDebounce()">
            <span class="slider-toggle"></span>
          </label>
        </div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Skin Smoothing (Pores/Radiance)</span>
            <span class="control-val" id="val_beauty_strength">0.80</span>
          </div>
          <input type="range" id="beauty_strength" min="0.1" max="1.0" step="0.05" value="0.80" oninput="onSliderMove('beauty_strength', this.value)">
        </div>
      </div>

      <!-- Detail & Clarity Tuning -->
      <div class="card">
        <div class="section-title">Clarity, Color & Optics</div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Lanczos Micro-Sharpening</span>
            <span class="control-val" id="val_sharpen">1.40</span>
          </div>
          <input type="range" id="sharpen" min="1.0" max="2.5" step="0.05" value="1.40" oninput="onSliderMove('sharpen', this.value)">
        </div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Denoise (Texture-Gated)</span>
            <span class="control-val" id="val_denoise">1.20</span>
          </div>
          <input type="range" id="denoise" min="0.0" max="3.0" step="0.1" value="1.20" oninput="onSliderMove('denoise', this.value)">
        </div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Dynamic Local Contrast</span>
            <span class="control-val" id="val_contrast">1.15</span>
          </div>
          <input type="range" id="contrast" min="1.0" max="1.45" step="0.02" value="1.15" oninput="onSliderMove('contrast', this.value)">
        </div>
        <div class="control-row">
          <div class="control-header">
            <span class="control-label">Color Vibrance & Pop</span>
            <span class="control-val" id="val_sat">1.12</span>
          </div>
          <input type="range" id="sat" min="1.0" max="1.45" step="0.02" value="1.12" oninput="onSliderMove('sat', this.value)">
        </div>
      </div>

      <!-- Target Resolution & Export -->
      <div class="card">
        <div class="section-title">Export Settings</div>
        <div class="control-row">
          <div class="control-label" style="margin-bottom: 6px;">Target Dimensions:</div>
          <select id="export_res" onchange="triggerDebounce()">
            <option value="print_300" selected>Print 2624 × 3936 (Proportional High-Res)</option>
            <option value="original">Original Dimensions</option>
            <option value="4k">4K UHD (3840 × 2160)</option>
            <option value="2k">2K QHD (2048 × 1536)</option>
          </select>
        </div>
        <div class="control-row">
          <div class="control-label" style="margin-bottom: 6px;">Export Density:</div>
          <select id="export_dpi" onchange="triggerDebounce()">
            <option value="300" selected>300 DPI (Official High-Res Print)</option>
            <option value="600">600 DPI (Ultra Fine Art)</option>
            <option value="72">72 DPI (Standard Web)</option>
          </select>
        </div>
        <div class="control-row">
          <div class="control-label" style="margin-bottom: 6px;">Output Format:</div>
          <select id="export_fmt">
            <option value="jpg" selected>JPEG / JPG (High Quality >1MB)</option>
            <option value="png">PNG (Lossless)</option>
            <option value="webp">WebP</option>
          </select>
        </div>
      </div>

    </div>

    <!-- Right Workspace Area -->
    <div class="workspace">
      
      <div class="toolbar">
        <div class="meta-badges">
          <div class="meta-badge" id="badge_dim">Resolution: <strong>-</strong></div>
          <div class="meta-badge" id="badge_dpi">DPI: <strong>-</strong></div>
          <div class="meta-badge" id="badge_size">Size: <strong>-</strong></div>
          <div class="meta-badge" id="badge_time">Latency: <strong>Ready</strong></div>
        </div>
        <div class="btn-group">
          <!-- View Modes -->
          <button class="btn btn-secondary active" id="btn_mode_slider" onclick="switchViewMode('slider')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/></svg>
            Split Slider
          </button>
          <button class="btn btn-secondary" id="btn_mode_side" onclick="switchViewMode('side')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"/><rect x="13" y="3" width="8" height="18" rx="2"/></svg>
            Side-by-Side
          </button>
          <button class="btn btn-secondary" id="btn_face_zoom" onclick="toggleFaceZoom()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            Face 1:1 Zoom
          </button>
          <input type="file" id="file_input" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
          <button class="btn btn-secondary" onclick="document.getElementById('file_input').click()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
            Upload
          </button>
          <button class="btn btn-secondary" onclick="loadSamplePhoto()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Load Sample (foto final)
          </button>
          <button class="btn btn-primary" id="btn_process" onclick="runEnhancement(false)" disabled>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            Apply Now
          </button>
          <button class="btn btn-success" id="btn_download" onclick="downloadResult()" disabled>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
            Download High-Res
          </button>
        </div>
      </div>

      <!-- Main Canvas Container -->
      <div class="canvas-container" id="canvas_container">
        
        <!-- Empty Dropzone State -->
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file_input').click()">
          <div class="drop-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242M12 12v9m-4-4 4 4 4-4"/>
            </svg>
          </div>
          <div class="drop-title">Drag & Drop Any Photo Here</div>
          <div class="drop-subtitle">Supports JPG, PNG, WEBP up to 50MB. Processed entirely with on-device local AI.</div>
          <button class="btn btn-primary" style="margin: 0 auto;">Select from Computer</button>
        </div>

        <!-- Mode 1: Split Comparison Slider -->
        <div class="comp-slider-box" id="slider_view" style="display: none;">
          <div class="pill-label before">ORIGINAL (WEBP)</div>
          <div class="pill-label after">ENHANCED (AI PRO)</div>
          <img id="img_after" class="comp-img" alt="Enhanced Photo">
          <div class="comp-overlay" id="comp_overlay">
            <img id="img_before" class="comp-overlay-img" alt="Original Photo">
            <div class="slider-handle" id="slider_handle">⇄</div>
          </div>
        </div>

        <!-- Mode 2: Side-by-Side View -->
        <div class="side-by-side-box" id="side_view" style="display: none;">
          <div class="side-card">
            <div class="side-title" style="color: #94A3B8;">ORIGINAL (WEBP)</div>
            <img id="img_side_orig" class="side-img" alt="Original">
          </div>
          <div class="side-card">
            <div class="side-title" style="color: var(--accent);">AI ENHANCED (300 DPI)</div>
            <img id="img_side_proc" class="side-img" alt="Enhanced">
          </div>
        </div>

        <!-- Live Status Toast -->
        <div class="live-indicator" id="live_indicator" style="display: none;">
          <div class="live-spinner"></div>
          <span id="live_indicator_text">Rendering live AI preview...</span>
        </div>

      </div>

    </div>
  </main>

  <script>
    let currentImageBase64 = null;
    let enhancedImageBase64 = null;
    let selectedPreset = 'magic';
    let currentViewMode = 'slider'; // 'slider' or 'side'
    let isFaceZoomActive = false;
    let debounceTimer = null;
    let isProcessing = false;

    function onSliderMove(id, val) {
      document.getElementById('val_' + id).innerText = parseFloat(val).toFixed(2);
      triggerDebounce();
    }

    function triggerDebounce() {
      if (!currentImageBase64) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        runEnhancement(true);
      }, 280);
    }

    function setPreset(p) {
      selectedPreset = p;
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      event.currentTarget.classList.add('active');

      if (p === 'magic') {
        document.getElementById('face_enabled').checked = true;
        document.getElementById('beauty_enabled').checked = true;
        setSlider('face_blend', 0.95);
        setSlider('beauty_strength', 0.80);
        setSlider('sharpen', 1.40);
        setSlider('denoise', 1.20);
        setSlider('contrast', 1.15);
        setSlider('sat', 1.12);
        document.getElementById('export_res').value = 'print_300';
      } else if (p === 'portrait') {
        document.getElementById('face_enabled').checked = true;
        document.getElementById('beauty_enabled').checked = true;
        setSlider('face_blend', 1.0);
        setSlider('beauty_strength', 0.95);
        setSlider('sharpen', 1.25);
        setSlider('denoise', 1.50);
        setSlider('contrast', 1.10);
        setSlider('sat', 1.08);
      } else if (p === 'print300') {
        document.getElementById('face_enabled').checked = false;
        document.getElementById('beauty_enabled').checked = false;
        setSlider('sharpen', 1.45);
        setSlider('denoise', 1.20);
        setSlider('contrast', 1.14);
        setSlider('sat', 1.08);
        document.getElementById('export_res').value = 'print_300';
        document.getElementById('export_dpi').value = '300';
      } else if (p === 'cinematic') {
        document.getElementById('face_enabled').checked = true;
        document.getElementById('beauty_enabled').checked = true;
        setSlider('face_blend', 0.85);
        setSlider('sharpen', 1.20);
        setSlider('contrast', 1.18);
        setSlider('sat', 1.10);
      } else if (p === 'remove_bg' || p === 'studio_gray') {
        document.getElementById('face_enabled').checked = true;
      }

      runEnhancement(true);
    }

    function setSlider(id, val) {
      const el = document.getElementById(id);
      el.value = val;
      document.getElementById('val_' + id).innerText = parseFloat(val).toFixed(2);
    }

    function switchViewMode(mode) {
      currentViewMode = mode;
      document.getElementById('btn_mode_slider').classList.toggle('active', mode === 'slider');
      document.getElementById('btn_mode_side').classList.toggle('active', mode === 'side');

      if (mode === 'slider') {
        document.getElementById('slider_view').style.display = currentImageBase64 ? 'inline-block' : 'none';
        document.getElementById('side_view').style.display = 'none';
        syncOverlayWidth();
      } else {
        document.getElementById('slider_view').style.display = 'none';
        document.getElementById('side_view').style.display = currentImageBase64 ? 'flex' : 'none';
      }
    }

    function toggleFaceZoom() {
      isFaceZoomActive = !isFaceZoomActive;
      document.getElementById('canvas_container').classList.toggle('zoom-active', isFaceZoomActive);
      document.getElementById('btn_face_zoom').classList.toggle('active', isFaceZoomActive);
    }

    function handleFileSelect(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(evt) {
        currentImageBase64 = evt.target.result;
        showImages(currentImageBase64, currentImageBase64);
        runEnhancement(true);
      };
      reader.readAsDataURL(file);
    }

    function loadSamplePhoto() {
      showLiveToast("Loading original foto final...");
      fetch('/api/sample')
        .then(r => r.json())
        .then(data => {
          hideLiveToast();
          if (data.image) {
            currentImageBase64 = data.image;
            showImages(currentImageBase64, currentImageBase64);
            document.getElementById('badge_dim').innerHTML = `Original: <strong>${data.dimensions[0]} × ${data.dimensions[1]}</strong>`;
            document.getElementById('badge_dpi').innerHTML = `DPI: <strong>${data.dpi}</strong>`;
            document.getElementById('badge_size').innerHTML = `Size: <strong>${data.file_size_mb} MB</strong>`;
            runEnhancement(true);
          }
        })
        .catch(err => {
          hideLiveToast();
          alert("Error loading sample: " + err);
        });
    }

    function showImages(origB64, procB64) {
      document.getElementById('dropzone').style.display = 'none';
      if (currentViewMode === 'slider') {
        document.getElementById('slider_view').style.display = 'inline-block';
      } else {
        document.getElementById('side_view').style.display = 'flex';
      }

      document.getElementById('img_before').src = origB64;
      document.getElementById('img_after').src = procB64;
      document.getElementById('img_side_orig').src = origB64;
      document.getElementById('img_side_proc').src = procB64;

      document.getElementById('btn_process').disabled = false;
      document.getElementById('btn_download').disabled = false;

      document.getElementById('img_after').onload = () => {
        syncOverlayWidth();
      };
    }

    function syncOverlayWidth() {
      const imgAfter = document.getElementById('img_after');
      const imgBefore = document.getElementById('img_before');
      if (imgAfter && imgBefore && imgAfter.clientWidth > 0) {
        imgBefore.style.width = imgAfter.clientWidth + 'px';
        imgBefore.style.height = imgAfter.clientHeight + 'px';
      }
    }

    window.addEventListener('resize', syncOverlayWidth);

    function runEnhancement(isLive = false) {
      if (!currentImageBase64 || isProcessing) return;
      isProcessing = true;
      showLiveToast(isLive ? "Updating AI preview in real-time..." : "Rendering High-Resolution 300 DPI Export...");

      const payload = {
        image: currentImageBase64,
        preset: selectedPreset,
        face_enabled: document.getElementById('face_enabled').checked,
        face_blend: parseFloat(document.getElementById('face_blend').value),
        beauty_enabled: document.getElementById('beauty_enabled').checked,
        beauty_strength: parseFloat(document.getElementById('beauty_strength').value),
        sharpen: parseFloat(document.getElementById('sharpen').value),
        denoise: parseFloat(document.getElementById('denoise').value),
        contrast: parseFloat(document.getElementById('contrast').value),
        sat: parseFloat(document.getElementById('sat').value),
        target_res: document.getElementById('export_res').value,
        dpi: parseInt(document.getElementById('export_dpi').value),
        format: document.getElementById('export_fmt').value,
        is_preview: isLive,
      };

      fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(r => r.json())
      .then(res => {
        isProcessing = false;
        hideLiveToast();
        if (res.error) {
          console.error("Enhancement error:", res.error);
          return;
        }

        enhancedImageBase64 = res.image;
        document.getElementById('img_after').src = enhancedImageBase64;
        document.getElementById('img_side_proc').src = enhancedImageBase64;
        syncOverlayWidth();

        document.getElementById('badge_dim').innerHTML = `Output: <strong>${res.dimensions[0]} × ${res.dimensions[1]}</strong>`;
        document.getElementById('badge_dpi').innerHTML = `DPI: <strong>${res.dpi}</strong>`;
        document.getElementById('badge_size').innerHTML = `Size: <strong>${res.file_size_mb} MB</strong>`;
        document.getElementById('badge_time').innerHTML = `Speed: <strong>${res.time_seconds}s</strong>`;
        document.getElementById('txt_status').innerText = `Enhanced in ${res.time_seconds}s`;
      })
      .catch(err => {
        isProcessing = false;
        hideLiveToast();
        console.error("Network error:", err);
      });
    }

    function downloadResult() {
      if (!enhancedImageBase64) return;
      const link = document.createElement('a');
      link.href = enhancedImageBase64;
      link.download = `enhanced_${Date.now()}.jpg`;
      link.click();
    }

    function showLiveToast(msg) {
      const el = document.getElementById('live_indicator');
      document.getElementById('live_indicator_text').innerText = msg;
      el.style.display = 'flex';
      el.style.opacity = '1';
    }

    function hideLiveToast() {
      const el = document.getElementById('live_indicator');
      el.style.opacity = '0';
      setTimeout(() => el.style.display = 'none', 300);
    }

    // Interactive slider handle drag
    (function setupSlider() {
      const box = document.getElementById('slider_view');
      const overlay = document.getElementById('comp_overlay');
      const handle = document.getElementById('slider_handle');
      let dragging = false;

      function onMove(clientX) {
        if (!dragging) return;
        const rect = box.getBoundingClientRect();
        let pos = (clientX - rect.left) / rect.width;
        pos = Math.max(0.02, Math.min(0.98, pos));
        overlay.style.width = (pos * 100) + '%';
      }

      window.addEventListener('mousedown', (e) => {
        if (e.target === handle || overlay.contains(e.target)) dragging = true;
      });
      window.addEventListener('mouseup', () => dragging = false);
      window.addEventListener('mousemove', (e) => onMove(e.clientX));

      window.addEventListener('touchstart', (e) => {
        if (e.target === handle || overlay.contains(e.target)) dragging = true;
      });
      window.addEventListener('touchend', () => dragging = false);
      window.addEventListener('touchmove', (e) => {
        if (dragging && e.touches[0]) onMove(e.touches[0].clientX);
      });
    })();
  </script>
</body>
</html>
"""


class StudioRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif parsed.path == "/api/sample":
            self.handle_get_sample()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/process":
            self.handle_process()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_get_sample(self):
        # Prefer the original unedited WEBP image so before/after contrast is massive!
        sample_path = SAMPLE_ORIG_WEBP if SAMPLE_ORIG_WEBP.exists() else SAMPLE_ENH_JPG
        if not sample_path.exists():
            self.send_json({"error": "Sample file not found"}, 404)
            return

        with Image.open(sample_path) as im:
            w, h = im.size
            dpi = im.info.get("dpi", (300, 300))[0]
            size_mb = round(sample_path.stat().st_size / (1024 * 1024), 2)

            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=95)
            b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.send_json({
            "image": b64_str,
            "dimensions": [w, h],
            "dpi": dpi,
            "file_size_mb": size_mb,
            "filename": sample_path.name,
        })

    def handle_process(self):
        t0 = time.time()
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))

            img_b64 = data.get("image", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_rgb = np.array(pil_img)
            orig_h, orig_w = img_rgb.shape[:2]

            is_preview = data.get("is_preview", False)

            # 1. Target Resolution
            target_res = data.get("target_res", "print_300")
            if target_res == "print_300":
                scale = max(2624.0 / orig_w, 3936.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                # For live interactive preview, keep preview responsive
                if is_preview and (target_w > 1600 or target_h > 1600):
                    p_scale = min(1400.0 / target_w, 1400.0 / target_h)
                    target_w = int(round(target_w * p_scale))
                    target_h = int(round(target_h * p_scale))
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif target_res == "4k":
                scale = max(3840.0 / orig_w, 2160.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            elif target_res == "2k":
                scale = max(2048.0 / orig_w, 1536.0 / orig_h)
                target_w = int(round(orig_w * scale))
                target_h = int(round(orig_h * scale))
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # 2. Denoising
            denoise = float(data.get("denoise", 1.2))
            if denoise > 0.05:
                img_rgb = apply_non_local_means_denoising(img_rgb, {"denoise_strength": denoise})

            # 3. Local Contrast Lift
            contrast = float(data.get("contrast", 1.15))
            if contrast > 1.01:
                img_rgb = enhance_contrast_local(img_rgb, {"contrast_boost": contrast})

            # 4. Color Vibrance & Pop
            sat = float(data.get("sat", 1.12))
            if sat > 1.01:
                img_rgb = enhance_saturation(img_rgb, {"saturation_boost": sat})

            # 5. High-Pass Crisp Sharpening
            sharpen = float(data.get("sharpen", 1.40))
            if sharpen > 1.01:
                img_rgb = adaptive_sharpening(img_rgb, {"sharpening_strength": sharpen})

            # 6. Local AI Face Restoration & Natural Skin Beauty
            preset = data.get("preset", "")
            if data.get("face_enabled", True) and preset != "print300":
                blend = float(data.get("face_blend", 0.95))
                beauty = bool(data.get("beauty_enabled", True))
                beauty_strength = float(data.get("beauty_strength", 0.80))
                img_rgb = enhance_faces_ai(
                    img_rgb,
                    blend_strength=blend,
                    enable_beauty=beauty,
                    beauty_strength=beauty_strength,
                )

            # 7. Cinematic Bokeh / Backdrop Presets
            if preset == "cinematic":
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cine_bgr = apply_cinematic_pro_camera(img_bgr, lens_bokeh_strength=0.55)
                img_rgb = cv2.cvtColor(cine_bgr, cv2.COLOR_BGR2RGB)
            elif preset == "remove_bg":
                img_rgb = replace_background_studio(img_rgb, preset="clean_white")
            elif preset == "studio_gray":
                img_rgb = replace_background_studio(img_rgb, preset="studio_gray")

            # 8. Encode Output
            dpi_val = int(data.get("dpi", 300))
            out_pil = Image.fromarray(img_rgb)
            buf = io.BytesIO()
            out_pil.save(
                buf,
                format="JPEG",
                dpi=(dpi_val, dpi_val),
                quality=97,
                subsampling=0,
            )
            raw_bytes = buf.getvalue()
            b64_out = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode("utf-8")

            elapsed = round(time.time() - t0, 2)
            out_h, out_w = img_rgb.shape[:2]
            size_mb = round(len(raw_bytes) / (1024 * 1024), 2)

            self.send_json({
                "image": b64_out,
                "dimensions": [out_w, out_h],
                "dpi": dpi_val,
                "file_size_mb": size_mb,
                "time_seconds": elapsed,
            })

        except Exception as e:
            logger.error(f"Process error: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_server(port: int = PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), StudioRequestHandler)
    print(f"============================================================")
    print(f"  ⚡ AI Image Enhancer Live Studio v2.0 running on:")
    print(f"  -> Local URL: http://localhost:{port}")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Live Studio server.")
        server.server_close()


if __name__ == "__main__":
    run_server()
