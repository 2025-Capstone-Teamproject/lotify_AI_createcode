from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime
import os

class IllegalParkingDetector:
    def __init__(self, model_path="../01_trained_models/integrated_model/integrated_3class_model_v2/weights/best.pt"):
        # 모델 로드
        if os.path.exists(model_path):
            self.violation_zone_model = YOLO(model_path)
        else:
            self.violation_zone_model = None
        
        self.vehicle_model = YOLO('yolov8n.pt')
        
        # 클래스 정의
        self.violation_classes = {
            0: {"name": "hydrant", "korean": "소화전", "penalty": 120000},
            1: {"name": "disabled_parking", "korean": "장애인주차구역", "penalty": 100000},
            2: {"name": "school_zone", "korean": "어린이보호구역", "penalty": 80000}
        }
        
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect_illegal_parking(self, image_path):
        """불법주차 탐지 메인 함수"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            return {"success": False, "error": "이미지 로드 실패"}

        # 특수구역 탐지
        violation_zones = []
        if self.violation_zone_model:
            results = self.violation_zone_model.predict(image, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    if hasattr(box, 'xyxy'):
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if class_id in self.violation_classes:
                            violation_zones.append({
                                'bbox': [int(x) for x in coords],
                                'confidence': round(conf, 3),
                                'class_id': class_id,
                                'type': self.violation_classes[class_id]['name'],
                                'korean_name': self.violation_classes[class_id]['korean'],
                                'penalty': self.violation_classes[class_id]['penalty']
                            })

        # 차량 탐지
        vehicles = []
        results = self.vehicle_model.predict(image, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                if hasattr(box, 'cls'):
                    class_id = int(box.cls[0])
                    if class_id in self.vehicle_classes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        vehicles.append({
                            'bbox': [int(x) for x in coords],
                            'confidence': round(conf, 3),
                            'vehicle_type': self.vehicle_classes[class_id]
                        })

        # 불법주차 판단 (거리 기반)
        h, w = image.shape[:2]
        distance_threshold = ((w**2 + h**2) ** 0.5) * 0.4
        
        violations = []
        for vehicle in vehicles:
            for zone in violation_zones:
                # 중심점 거리 계산
                v_center = [(vehicle['bbox'][0] + vehicle['bbox'][2])/2, 
                           (vehicle['bbox'][1] + vehicle['bbox'][3])/2]
                z_center = [(zone['bbox'][0] + zone['bbox'][2])/2, 
                           (zone['bbox'][1] + zone['bbox'][3])/2]
                
                distance = ((v_center[0] - z_center[0])**2 + (v_center[1] - z_center[1])**2)**0.5
                
                if distance < distance_threshold:
                    violations.append({
                        'vehicle': vehicle,
                        'violation_zone': zone,
                        'distance': round(distance, 1),
                        'penalty': zone['penalty']
                    })
                    break

        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'detection_summary': {
                'total_violation_zones': len(violation_zones),
                'total_vehicles': len(vehicles),
                'total_violations': len(violations),
                'total_penalty': sum(v['penalty'] for v in violations)
            },
            'detections': {
                'violation_zones': violation_zones,
                'vehicles': vehicles,
                'violations': violations
            }
        }

# 사용 예시
if __name__ == "__main__":
    detector = IllegalParkingDetector()
    result = detector.detect_illegal_parking("test_image.jpg")
    print(json.dumps(result, indent=2, ensure_ascii=False))
