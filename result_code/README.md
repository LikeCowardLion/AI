# 스마트 디바이스 위치 예측 AI 모델

## 프로젝트 개요

이 프로젝트는 스마트 디바이스를 발로 차는 동작에서 발생하는 가속도 및 자이로스코프 데이터를 기반으로 스마트 디바이스의 위치를 예측하는 인공지능 모델입니다.

## 주요 기능

- **실시간 센서 데이터 처리**: UDP를 통한 실시간 센서 데이터 수신
- **AI 모델 기반 위치 예측**: 딥러닝 모델을 사용한 정확한 위치 예측
- **Unity 연동**: 예측 결과를 Unity로 실시간 전송
- **데이터 검증 및 전처리**: 센서 데이터의 유효성 검증 및 정규화
- **성능 모니터링**: 실시간 처리 통계 및 성능 지표 제공

## 시스템 요구사항

- Python 3.8 이상
- TensorFlow 2.10.0 이상
- 필요한 패키지: `requirements.txt` 참조

## 설치 방법

1. 저장소 클론
```bash
git clone <repository-url>
cd puck
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 기본 실행

최적화된 UDP 서버 실행:
```bash
python optimized_test.py
```

### 2. 설정 변경

`optimized_test.py`에서 다음 설정을 변경할 수 있습니다:

```python
server = PuckUDPServer(
    listen_ip="127.0.0.1",      # 수신 IP
    listen_port=8080,           # 수신 포트
    unity_ip="127.0.0.1",       # Unity IP
    unity_port=8082,            # Unity 포트
    buffer_size=4096            # 버퍼 크기
)
```

### 3. 데이터 형식

센서 데이터는 다음 형식으로 전송되어야 합니다:
```
Acc_X,Acc_Y,Acc_Z,Gyr_X,Gyr_Y,Gyr_Z
```

- **Acc_X, Acc_Y, Acc_Z**: 가속도 센서 데이터 (X, Y, Z축)
- **Gyr_X, Gyr_Y, Gyr_Z**: 자이로스코프 데이터 (X, Y, Z축)

### 4. 예측 결과

Unity로 전송되는 예측 결과 형식:
```
|x좌표|y좌표|
```

## 프로젝트 구조

```
puck/
├── optimized_model_predict.py  # 최적화된 모델 예측 클래스
├── optimized_test.py          # 최적화된 UDP 서버
├── data_processor.py          # 데이터 전처리 클래스
├── requirements.txt           # 의존성 패키지 목록
├── README.md                 # 프로젝트 설명서
├── model123.h5              # 학습된 AI 모델
├── scaler.pkl               # 데이터 스케일러
└── 기존 파일들...
```

## 주요 최적화 사항

### 1. 성능 최적화
- **모델 로딩 최적화**: 싱글톤 패턴으로 모델을 한 번만 로드
- **메모리 효율성**: 불필요한 데이터 복사 최소화
- **비동기 처리**: 멀티스레딩을 통한 동시 처리

### 2. 안정성 향상
- **강화된 에러 처리**: 상세한 예외 처리 및 로깅
- **데이터 검증**: 입력 데이터 유효성 검증
- **타임아웃 설정**: 네트워크 통신 타임아웃 관리

### 3. 모니터링 및 디버깅
- **실시간 통계**: 처리 성공률, 평균 처리 시간 등
- **상세한 로깅**: 각 단계별 로그 출력
- **성능 지표**: 메모리 사용량, 처리 시간 모니터링

## API 문서

### PuckPredictor 클래스

```python
from optimized_model_predict import PuckPredictor

# 예측기 초기화
predictor = PuckPredictor()

# 위치 예측
predictions = predictor.predict(sensor_data)
```

### SensorDataProcessor 클래스

```python
from data_processor import SensorDataProcessor

# 데이터 프로세서 초기화
processor = SensorDataProcessor()

# 데이터 처리
success, processed_data, message = processor.process_sensor_data(raw_data)
```

## 문제 해결

### 일반적인 문제들

1. **모델 로딩 실패**
   - `model123.h5` 파일이 올바른 경로에 있는지 확인
   - TensorFlow 버전 호환성 확인

2. **UDP 연결 실패**
   - 포트가 사용 중인지 확인
   - 방화벽 설정 확인

3. **데이터 형식 오류**
   - 센서 데이터가 올바른 형식인지 확인
   - 10행의 6열 데이터인지 확인

## 기여 방법

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요. 