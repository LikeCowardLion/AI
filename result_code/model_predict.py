import tensorflow as tf
import keras
import numpy as np
import pickle
import pandas as pd
from typing import List, Tuple, Optional
import logging
from data_processor import get_data_processor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PuckPredictor:
    """스마트 디바이스 위치 예측을 위한 최적화된 예측 클래스"""
    
    def __init__(self, model_path: str = 'model123.h5', scaler_path: str = 'scaler.pkl'):
        """
        Args:
            model_path: 모델 파일 경로
            scaler_path: 스케일러 파일 경로
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.data_processor = get_data_processor()  # 데이터 프로세서 사용
        self._load_model_and_scaler()
        
    def _load_model_and_scaler(self):
        """모델과 스케일러를 한 번만 로드"""
        try:
            # 모델 로드
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
            logger.info("모델 로드 완료")
            
            # 스케일러 로드
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("스케일러 로드 완료")
            
        except Exception as e:
            logger.error(f"모델/스케일러 로드 실패: {e}")
            raise
    
    def predict(self, input_data: List[List[float]]) -> np.ndarray:
        """
        위치 예측 수행
        
        Args:
            input_data: 입력 데이터 (10행 x 6열)
            
        Returns:
            예측된 위치 좌표
        """
        try:
            if len(input_data) != 10:
                raise ValueError(f"입력 데이터는 10행이어야 합니다. 현재: {len(input_data)}행")
            
            # 데이터 프로세서를 사용하여 데이터 처리
            success, processed_data, message = self.data_processor.process_sensor_data(input_data)
            
            if not success:
                raise ValueError(f"데이터 처리 실패: {message}")
            
            # 배치 차원 추가
            model_input = processed_data.reshape(1, *processed_data.shape)
            
            # 예측 수행
            prediction = self.model.predict(model_input, verbose=0)
            
            # 스케일러를 사용하여 역변환
            result = self.scaler.inverse_transform(prediction)
            
            logger.info(f"예측 완료: {result.flatten()}")
            return result
            
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise

# 전역 예측기 인스턴스 (한 번만 생성)
_predictor = None

def get_predictor() -> PuckPredictor:
    """싱글톤 패턴으로 예측기 인스턴스 반환"""
    global _predictor
    if _predictor is None:
        _predictor = PuckPredictor()
    return _predictor

def puck_predict(input_data: List[List[float]]) -> np.ndarray:
    """
    기존 인터페이스와 호환되는 예측 함수
    
    Args:
        input_data: 입력 데이터
        
    Returns:
        예측 결과
    """
    predictor = get_predictor()
    return predictor.predict(input_data) 