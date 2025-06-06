import requests
import base64
import json
import os
from detector import IllegalParkingDetector

def test_detector():
    """탐지기 직접 테스트"""
    print("🧪 탐지기 테스트...")
    detector = IllegalParkingDetector()
    
    # 샘플 이미지가 있다면 테스트
    sample_dir = "../02_datasets/samples"
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.endswith(('.jpg', '.png')):
                result = detector.detect_illegal_parking(os.path.join(sample_dir, file))
                print(f"✓ {file}: {result['detection_summary']}")
                break

def test_api():
    """API 서버 테스트"""
    print("🌐 API 테스트...")
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✓ 서버 상태 정상")
        else:
            print("❌ 서버 연결 실패")
    except:
        print("❌ 서버가 실행되지 않았습니다. python app.py로 서버를 먼저 실행하세요.")

if __name__ == "__main__":
    test_detector()
    test_api()
