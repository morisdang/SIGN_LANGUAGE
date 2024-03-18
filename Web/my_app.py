from flask import Flask, render_template,Response, jsonify, request
import cv2
import os
import sys
sys.path.insert(0, "D:/AI/DPLS2L/S2L/Project")
import web_detect

app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route("/")
def demo():
    return render_template("demo.html")

def generate_results(file):
    for result in web_detect.predict_video('upload/' + file.filename):
        yield f"{result}\n\n"

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Không có phần tệp nào được chọn."

    file = request.files['file']

    if file.filename == '':
        return "Chưa chọn file."

    if file:
        file.save('upload/' + file.filename)
        return Response(generate_results(file), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run( debug=False)