import argparse
import io
from io import BytesIO
import os
from PIL import Image
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)


model = torch.hub.load(
        "ultralytics/yolov5", "custom", path = "model/tugaktenun.pt", force_reload=True
        )

model.eval()
model.conf = 0.6  
model.iou = 0.45  

def gen():
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            results.print()  
            img = np.squeeze(results.render())
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')
    
@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 


# from re import DEBUG, sub
# from flask import Flask, render_template, request, redirect, send_file, url_for
# from werkzeug.utils import secure_filename, send_from_directory
# import os
# import subprocess

# app = Flask(__name__)


# uploads_dir = os.path.join(app.instance_path, 'uploads')

# os.makedirs(uploads_dir, exist_ok=True)

# @app.route("/")
# def hello_world():
#     return render_template('index.html')


# @app.route("/detect", methods=['POST'])
# def detect():
#     if not request.method == "POST":
#         return
#     video = request.files['video']
#     video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
#     print(video)
#     subprocess.run("ls")
#     subprocess.run(['python3', 'detect.py', '--source', os.path.join(uploads_dir, secure_filename(video.filename))])

#     # return os.path.join(uploads_dir, secure_filename(video.filename))
#     obj = secure_filename(video.filename)
#     return obj

# @app.route("/opencam", methods=['GET'])
# def opencam():
#     print("here")
#     subprocess.run(['python3', 'detect.py', '--source', '0'])
#     return "done"
    

# @app.route('/return-files', methods=['GET'])
# def return_file():
#     obj = request.args.get('obj')
#     loc = os.path.join("runs/detect", obj)
#     print(loc)
#     try:
#         return send_file(os.path.join("runs/detect", obj), attachment_filename=obj)
#         # return send_from_directory(loc, obj)
#     except Exception as e:
#         return str(e)

# # @app.route('/display/<filename>')
# # def display_video(filename):
# # 	#print('display_video filename: ' + filename)
# # 	return redirect(url_for('static/video_1.mp4', code=200))