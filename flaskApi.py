from flask import Flask,request, jsonify
import numpy as np
import cv2
from fer import FER

app = Flask(__name__)

@app.route('/analyze',methods=['POST'])
def index():   
    image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)    
    emotion_detector = FER(mtcnn=True)

    kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)    
    analysis = emotion_detector.detect_emotions(image_sharp)  
    
    if(len(analysis) > 1):            
        return jsonify({'result': "There are multiple faces in the image"})
    elif (len(analysis) == 1):
        dominant_emotion, emotion_score = emotion_detector.top_emotion(image_sharp) 
        return jsonify({'result': dominant_emotion})                    
    else:
        return jsonify({'result': "No Face Found"})
    
if __name__ == '__main__':
    app.run(host="localhost", port=8080)
