import cv2
import numpy as np
import base64
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ✅ LIMIT UPLOAD SIZE (VERY IMPORTANT)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4MB

# ---------- UI ----------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Accurate ID Printer (Horizontal)</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background:#f0f2f5; text-align:center; padding:20px; }
.container { background:white; max-width:900px; margin:auto; padding:40px; border-radius:20px; }
.upload-grid { display:grid; grid-template-columns:1fr 1fr; gap:25px; }
.upload-box { border:2px dashed #007bff; padding:30px; border-radius:12px; }
.btn { background:#007bff; color:white; padding:18px; border:none; width:100%; font-size:18px; }
.preview-img { max-width:450px; border-radius:8px; margin-top:20px; }
.btn-download { background:#28a745; color:white; padding:15px 40px; border-radius:8px; display:inline-block; }
</style>
</head>
<body>
<div class="container">
<h1>ID Card Print Generator</h1>

<form method="POST" enctype="multipart/form-data">
<div class="upload-grid">
<div class="upload-box">
<strong>FRONT</strong>
<input type="file" name="front_file" accept="image/*" required>
</div>
<div class="upload-box">
<strong>BACK</strong>
<input type="file" name="back_file" accept="image/*" required>
</div>
</div>
<button class="btn">GENERATE A4</button>
</form>

{% if a4_result %}
<img src="data:image/jpeg;base64,{{ a4_result }}" class="preview-img">
<br>
<a href="data:image/jpeg;base64,{{ a4_result }}" download="ID_Print.jpg" class="btn-download">
DOWNLOAD
</a>
{% endif %}
</div>
</body>
</html>
"""

# ---------- IMAGE PROCESSING ----------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_perfect_crop(img):
    h, w = img.shape[:2]

    # ✅ REDUCED SCALE (FAST & SAFE)
    scale = 600 / max(h, w)
    img_small = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Enhance edges lightly
    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 40, 120)
    edged = cv2.dilate(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # ✅ SAFE ID SIZE
    target_w, target_h = 800, 504

    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype("float32") / scale
            rect = order_points(pts)
            dst = np.array([
                [0, 0],
                [target_w, 0],
                [target_w, target_h],
                [0, target_h]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img, M, (target_w, target_h))

    # ✅ SAFE FALLBACK (NO GRABCUT)
    return cv2.resize(img, (target_w, target_h))

def create_a4_horizontal(front, back):
    # ✅ SAFE A4 SIZE (150 DPI)
    a4_w, a4_h = 1240, 1754
    canvas = np.ones((a4_h, a4_w, 3), dtype="uint8") * 255

    fh, fw = front.shape[:2]
    gap = 100

    cx, cy = a4_w // 2, a4_h // 2
    start_x = cx - (fw * 2 + gap) // 2
    start_y = cy - fh // 2

    canvas[start_y:start_y+fh, start_x:start_x+fw] = front
    canvas[start_y:start_y+fh, start_x+fw+gap:start_x+fw*2+gap] = back

    return canvas

def to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()

# ---------- ROUTE ----------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        f = request.files.get("front_file")
        b = request.files.get("back_file")

        if f and b:
            img_f = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            img_b = cv2.imdecode(np.frombuffer(b.read(), np.uint8), cv2.IMREAD_COLOR)

            if img_f is not None and img_b is not None:
                front = get_perfect_crop(img_f)
                back = get_perfect_crop(img_b)
                a4 = create_a4_horizontal(front, back)
                result = to_base64(a4)

    return render_template_string(HTML_PAGE, a4_result=result)

# ---------- LOCAL RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
