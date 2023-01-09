from flask import Flask,request, jsonify
import numpy as np
import cv2
from fer import FER

app = Flask(__name__)

@app.route('/analyze',methods=['POST'])
def index():   
    image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)    
    emotion_detector = FER(mtcnn=True)

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    
    grayscaled_img = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2GRAY)
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    totalFaces = len(face_coordinates)
    if(totalFaces > 1):            
        return jsonify({'result': "There are multiple faces in the image"})
    else:
        analysis = emotion_detector.detect_emotions(image_sharp)        
        dominant_emotion, emotion_score = emotion_detector.top_emotion(image_sharp)            
        if(dominant_emotion is None):
            return jsonify({'result': "No Face Found"})
        else:
            return jsonify({'result': dominant_emotion})
    
if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    
    