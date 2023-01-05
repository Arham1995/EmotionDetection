from flask import Flask,request, jsonify
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)

@app.route('/analyze',methods=['POST'])
def index():   
    image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    try:
        grayscaled_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        totalFaces = len(face_coordinates)
        if(totalFaces > 1):            
            return jsonify({'result': "There are multiple faces in the image"})
        else:
            analyze = DeepFace.analyze(image,actions=['emotion']) 
            return jsonify({'result': analyze['dominant_emotion']})
             
    except:
        return jsonify({'result': "No Face Found"})

    
if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    
    