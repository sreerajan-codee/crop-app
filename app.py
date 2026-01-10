import cv2
import numpy as np
import base64
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ---------- THE PERFECT UI ----------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Accurate ID Printer (Horizontal)</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; text-align: center; padding: 20px; color: #333; }
        .container { background: white; max-width: 900px; margin: auto; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        h1 { color: #007bff; margin-bottom: 10px; }
        .status-msg { color: #666; margin-bottom: 30px; }
        .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 30px; }
        .upload-box { border: 2px dashed #007bff; padding: 30px; border-radius: 12px; background: #f8fbff; transition: all 0.3s ease; }
        .upload-box:hover { background: #edf5ff; border-color: #0056b3; }
        .upload-box strong { display: block; margin-bottom: 15px; color: #0056b3; }
        .btn { background: #007bff; color: white; border: none; padding: 18px 30px; border-radius: 8px; cursor: pointer; font-size: 18px; width: 100%; font-weight: bold; transition: background 0.2s; }
        .btn:hover { background: #0056b3; }
        .a4-container { margin-top: 40px; padding-top: 30px; border-top: 2px solid #eee; }
        .preview-img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); margin-bottom: 20px; }
        .btn-download { background: #28a745; text-decoration: none; display: inline-block; padding: 15px 40px; color: white; border-radius: 8px; font-weight: bold; font-size: 16px; }
        .btn-download:hover { background: #218838; }
        input[type="file"] { width: 100%; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ID Card Print Generator</h1>
        <p class="status-msg">Upload Front and Back sides for a perfectly cropped, side-by-side A4 layout.</p>
        
        <form method="POST" enctype="multipart/form-data">
            <div class="upload-grid">
                <div class="upload-box">
                    <strong>FRONT SIDE</strong>
                    <input type="file" name="front_file" accept="image/*" required>
                </div>
                <div class="upload-box">
                    <strong>BACK SIDE</strong>
                    <input type="file" name="back_file" accept="image/*" required>
                </div>
            </div>
            <button type="submit" class="btn">GENERATE PRINT-READY A4 SHEET</button>
        </form>

        {% if a4_result %}
        <div class="a4-container">
            <h3>Final Print Preview (Centered Horizontal)</h3>
            <img src="data:image/jpeg;base64,{{ a4_result }}" class="preview-img" style="max-width: 450px;">
            <br>
            <a href="data:image/jpeg;base64,{{ a4_result }}" download="ID_Print_Ready_Horizontal.jpg" class="btn-download">
                DOWNLOAD A4 SHEET
            </a>
            <p><small style="color: #888; display: block; mt-10">Tip: Print at 100% scale for exact 3.375" x 2.125" dimensions.</small></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ---------- ACCURATE IMAGE PROCESSING LOGIC ----------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # Top-Left
    rect[2] = pts[np.argmax(s)]    # Bottom-Right
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def get_perfect_crop(img_bgr):
    """Accurate 3-stage detection from your most successful logic."""
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    img_small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    
    # 1. Image Enhancement
    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    
    # 2. Edge Detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 150)
    edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    target_w, target_h = 1012, 638 # Proper ID Ratio (approx 300DPI)

    # STAGE 1: Geometric Contour Detection
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32") / scale
            rect = order_points(pts)
            dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h))
            # Clean edges (shave 1.5%)
            sh, sw = int(target_h * 0.015), int(target_w * 0.015)
            return warped[sh:target_h-sh, sw:target_w-sw]

    # STAGE 2: GrabCut Fallback (Background Removal)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    rect_gc = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect_gc, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        cnts_gc, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_gc:
            c = max(cnts_gc, key=cv2.contourArea)
            rect_min = cv2.minAreaRect(c)
            pts = cv2.boxPoints(rect_min).astype("float32")
            rect = order_points(pts)
            dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img_bgr, M, (target_w, target_h))
    except: pass

    # STAGE 3: Smart Center Fallback
    ratio = 3.375 / 2.125
    if w/h > ratio:
        new_w = int(h * ratio); start = (w - new_w) // 2
        crop = img_bgr[:, start:start+new_w]
    else:
        new_h = int(w / ratio); start = (h - new_h) // 2
        crop = img_bgr[start:start+new_h, :]
    return cv2.resize(crop, (target_w, target_h))

def create_a4_horizontal(front, back):
    """Places cards side-by-side in the center of an A4 canvas."""
    # A4 dimensions at 300 DPI
    a4_w, a4_h = 2480, 3508
    canvas = np.ones((a4_h, a4_w, 3), dtype="uint8") * 255
    
    fh, fw = front.shape[:2]
    bh, bw = back.shape[:2]
    gap = 120 # Space between cards
    
    # Calculate centering
    cx, cy = a4_w // 2, a4_h // 2
    total_width = fw + bw + gap
    
    start_x = cx - (total_width // 2)
    start_y = cy - (fh // 2)
    
    # Place Front (Left)
    canvas[start_y:start_y+fh, start_x:start_x+fw] = front
    
    # Place Back (Right)
    canvas[start_y:start_y+bh, start_x+fw+gap : start_x+fw+gap+bw] = back
    
    return canvas

def to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf).decode("utf-8")

# ---------- FLASK APP ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    a4_b64 = None
    if request.method == "POST":
        f_file = request.files.get("front_file")
        b_file = request.files.get("back_file")
        
        if f_file and b_file:
            img_f = cv2.imdecode(np.frombuffer(f_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img_b = cv2.imdecode(np.frombuffer(b_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            if img_f is not None and img_b is not None:
                # Accurate Cropping
                crop_f = get_perfect_crop(img_f)
                crop_b = get_perfect_crop(img_b)
                
                # Horizontal A4 Layout
                a4_img = create_a4_horizontal(crop_f, crop_b)
                a4_b64 = to_base64(a4_img)
                
    return render_template_string(HTML_PAGE, a4_result=a4_b64)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

