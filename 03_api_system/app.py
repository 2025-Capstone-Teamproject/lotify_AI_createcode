from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from detector import IllegalParkingDetector

app = Flask(__name__)
CORS(app)

detector = IllegalParkingDetector()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "불법주차 탐지 API 서버 정상 작동"})

@app.route('/detect', methods=['POST'])
def detect_illegal_parking():
    try:
        if 'image' not in request.json:
            return jsonify({"success": False, "error": "이미지 데이터 없음"}), 400
        
        # Base64 디코딩
        image_data = base64.b64decode(request.json['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "error": "이미지 디코딩 실패"}), 400
        
        result = detector.detect_illegal_parking(image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        "model_version": "v2.0",
        "classes": detector.violation_classes,
        "vehicle_classes": detector.vehicle_classes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
