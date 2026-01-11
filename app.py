import cv2
import numpy as np
import base64
import os
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# 

# ---------- UI REMAINS THE SAME (STABLE) ----------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accurate Fast ID Printer</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; text-align: center; padding: 20px; }
        .container { background: white; max-width: 800px; margin: auto; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .upload-box { border: 2px dashed #007bff; padding: 20px; border-radius: 12px; background: #f8fbff; }
        .preview-area { height: 150px; background: #eee; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 8px; }
        .preview-area img { max-height: 90%; max-width: 90%; transition: transform 0.3s; }
        .btn { background: #007bff; color: white; border: none; padding: 15px; border-radius: 8px; cursor: pointer; font-size: 18px; width: 100%; font-weight: bold; }
        .rot-btn { background: #ffc107; color: black; border: none; padding: 8px; width: 100%; cursor: pointer; margin-top: 5px; border-radius: 5px; font-weight: bold; }
        #prog-wrap { display: none; margin: 20px 0; }
        .meter { height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; position: relative; }
        #fill { width: 0%; height: 100%; background: #28a745; transition: width 0.2s; }
        #pct { position: absolute; width: 100%; top:0; left:0; font-size: 12px; line-height: 20px; font-weight: bold; }
        #result-img { max-width: 100%; margin-top: 20px; border: 1px solid #ddd; display: none; border-radius: 10px; }
        .btn-dl { display: none; background: #28a745; color: white; padding: 15px; text-decoration: none; border-radius: 8px; margin-top: 10px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ID Card Print Generator</h1>
        <div class="upload-grid">
            <div class="upload-box">
                <strong>FRONT SIDE</strong>
                <div class="preview-area"><img id="pf"></div>
                <input type="file" id="f_file" accept="image/*">
                <button type="button" class="rot-btn" onclick="rotate('pf', 'f_rot')">ROTATE 90° ↻</button>
                <input type="hidden" id="f_rot" value="0">
            </div>
            <div class="upload-box">
                <strong>BACK SIDE</strong>
                <div class="preview-area"><img id="pb"></div>
                <input type="file" id="b_file" accept="image/*">
                <button type="button" class="rot-btn" onclick="rotate('pb', 'b_rot')">ROTATE 90° ↻</button>
                <input type="hidden" id="b_rot" value="0">
            </div>
        </div>
        <button class="btn" onclick="upload()">GENERATE PRINT-READY A4</button>
        <div id="prog-wrap">
            <div class="meter"><div id="fill"></div><div id="pct">0%</div></div>
            <p id="msg" style="color: #007bff; font-weight: bold; margin-top: 10px;">Processing...</p>
        </div>
        <img id="result-img">
        <br><br>
        <a id="dl-btn" href="#" download="ID_Print_Ready.jpg" class="btn-dl">DOWNLOAD A4 SHEET</a>
    </div>

    <script>
        document.getElementById('f_file').onchange = e => { preview(e, 'pf'); };
        document.getElementById('b_file').onchange = e => { preview(e, 'pb'); };
        function preview(e, id) {
            let reader = new FileReader();
            reader.onload = () => { document.getElementById(id).src = reader.result; };
            reader.readAsDataURL(e.target.files[0]);
        }
        function rotate(imgId, inputId) {
            let img = document.getElementById(imgId);
            let input = document.getElementById(inputId);
            let r = (parseInt(input.value) + 90) % 360;
            input.value = r;
            img.style.transform = `rotate(${r}deg)`;
        }
        function upload() {
            let f = document.getElementById('f_file').files[0];
            let b = document.getElementById('b_file').files[0];
            if(!f || !b) return alert("Select both images!");
            let fd = new FormData();
            fd.append('front_file', f);
            fd.append('back_file', b);
            fd.append('front_rot', document.getElementById('f_rot').value);
            fd.append('back_rot', document.getElementById('b_rot').value);
            let xhr = new XMLHttpRequest();
            document.getElementById('prog-wrap').style.display = 'block';
            xhr.upload.onprogress = e => {
                let p = Math.round((e.loaded / e.total) * 100);
                document.getElementById('fill').style.width = p + '%';
                document.getElementById('pct').innerText = p + '%';
            };
            xhr.onload = function() {
                let res = JSON.parse(xhr.responseText);
                document.getElementById('prog-wrap').style.display = 'none';
                let img = document.getElementById('result-img');
                img.src = "data:image/jpeg;base64," + res.image;
                img.style.display = 'block';
                let dl = document.getElementById('dl-btn');
                dl.href = img.src;
                dl.style.display = 'inline-block';
            };
            xhr.open("POST", "/generate");
            xhr.send(fd);
        }
    </script>
</body>
</html>
"""

# ---------- ACCURATE BACKEND PROCESSING ----------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # Top-Left
    rect[2] = pts[np.argmax(s)]    # Bottom-Right
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def get_fast_crop(img_bgr):
    """Accurate but fast cropping using Contrast Enhancement."""
    h, w = img_bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    
    # IMPROVEMENT 1: Increase Contrast (CLAHE)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    
    # IMPROVEMENT 2: Better Edge Detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    # IMPROVEMENT 3: Dilation to close gaps in ID border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    tw, th = 1012, 638 
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # We look for 4 points and a minimum size (10% of image)
        if len(approx) == 4 and cv2.contourArea(c) > (small.shape[0]*small.shape[1]*0.1):
            pts = approx.reshape(4, 2).astype("float32") / scale
            M = cv2.getPerspectiveTransform(order_points(pts), np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]], dtype="float32"))
            return cv2.warpPerspective(img_bgr, M, (tw, th))

    # Fallback to Smart Center Crop
    ratio = 3.375 / 2.125
    if w/h > ratio:
        nw = int(h * ratio); s = (w - nw) // 2
        crop = img_bgr[:, s:s+nw]
    else:
        nh = int(w / ratio); s = (h - nh) // 2
        crop = img_bgr[s:s+nh, :]
    return cv2.resize(crop, (tw, th))

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/generate", methods=["POST"])
def generate():
    f_file = request.files['front_file']
    b_file = request.files['back_file']
    f_rot = int(request.form.get('front_rot', 0))
    b_rot = int(request.form.get('back_rot', 0))

    img_f = cv2.imdecode(np.frombuffer(f_file.read(), np.uint8), 1)
    img_b = cv2.imdecode(np.frombuffer(b_file.read(), np.uint8), 1)

    def rot(i, a):
        if a == 90: return cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)
        if a == 180: return cv2.rotate(i, cv2.ROTATE_180)
        if a == 270: return cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return i

    # Process rotated images
    cf = get_fast_crop(rot(img_f, f_rot))
    cb = get_fast_crop(rot(img_b, b_rot))

    # A4 Canvas
    canvas = np.ones((3508, 2480, 3), dtype="uint8") * 255
    gap = 120
    sx = (2480 - (1012*2 + gap)) // 2
    sy = (3508 - 638) // 2
    canvas[sy:sy+638, sx:sx+1012] = cf
    canvas[sy:sy+638, sx+1012+gap : sx+2024+gap] = cb

    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({"image": base64.b64encode(buf).decode("utf-8")})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
