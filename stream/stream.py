import cv2
from flask import Flask, Response

app = Flask(__name__)

def video_stream():
    # Capturar video desde la cámara
    cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Codificar el frame como JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Crear un flujo continuo para el navegador
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #url: http://172.20.16.1:5000/video_feed
    app.run(host='0.0.0.0', port=5000, debug=True)
