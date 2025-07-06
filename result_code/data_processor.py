import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class SensorDataProcessor:
    """센서 데이터 전처리 및 검증을 위한 최적화된 클래스"""
    
    def __init__(self):
        """센서 데이터 프로세서 초기화"""
        # 센서 데이터 컬럼 정의
        self.sensor_columns = ['Acc_X', 'Acc_Y', 'Acc_Z', ' Gyr_X', ' Gyr_Y', ' Gyr_Z']
        
        # 사용할 특성 컬럼 (Acc_X, Acc_Y, Gyr_Z만 사용)
        self.feature_columns = ['Acc_X', 'Acc_Y', ' Gyr_Z']
        
        # 각 특성별 정규화 범위 (학습 시 사용된 범위)
        self.normalization_ranges = {
            'Acc_X': (-3.99, 4.02),
            'Acc_Y': (-4.019, 2.918),
            'Gyr_Z': (-2297.985, 2289.371)
        }
        
        # 데이터 검증 기준
        self.validation_rules = {
            'min_rows': 10,
            'max_rows': 10,
            'required_columns': 6,
            'numeric_check': True
        }
        
        # 스케일러 캐시 (성능 최적화)
        self._scalers = {}
        
    def validate_sensor_data(self, data: List[List[float]]) -> Tuple[bool, str]:
        """
        센서 데이터 유효성 검증
        
        Args:
            data: 검증할 센서 데이터
            
        Returns:
            (유효성 여부, 오류 메시지)
        """
        try:
            # 행 수 검증
            if len(data) < self.validation_rules['min_rows']:
                return False, f"데이터 행 수 부족: {len(data)} < {self.validation_rules['min_rows']}"
            
            if len(data) > self.validation_rules['max_rows']:
                return False, f"데이터 행 수 초과: {len(data)} > {self.validation_rules['max_rows']}"
            
            # 각 행 검증
            for i, row in enumerate(data):
                if len(row) != self.validation_rules['required_columns']:
                    return False, f"행 {i}: 컬럼 수 불일치 ({len(row)} != {self.validation_rules['required_columns']})"
                
                # 숫자형 데이터 검증
                if self.validation_rules['numeric_check']:
                    for j, val in enumerate(row):
                        try:
                            float(val)
                        except (ValueError, TypeError):
                            return False, f"행 {i}, 컬럼 {j}: 숫자가 아닌 값 ({val})"
            
            return True, "데이터 유효성 검증 통과"
            
        except Exception as e:
            return False, f"데이터 검증 중 오류: {str(e)}"
    
    def extract_features(self, data: List[List[float]]) -> np.ndarray:
        """
        센서 데이터에서 필요한 특성 추출
        
        Args:
            data: 원본 센서 데이터
            
        Returns:
            추출된 특성 배열 (Acc_X, Acc_Y, Gyr_Z)
        """
        try:
            # DataFrame 생성
            df = pd.DataFrame(data, columns=self.sensor_columns)
            
            # 필요한 특성만 선택
            features_df = df[self.feature_columns].astype(float)
            
            return features_df.values
            
        except Exception as e:
            logger.error(f"특성 추출 실패: {e}")
            raise
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        특성 정규화 (Min-Max 스케일링)
        
        Args:
            features: 정규화할 특성 배열
            
        Returns:
            정규화된 특성 배열
        """
        try:
            normalized_features = features.copy()
            
            for col_idx, feature_name in enumerate(self.feature_columns):
                min_val, max_val = self.normalization_ranges[feature_name]
                
                # 기존 방식: 데이터에 최소/최대값을 추가하여 MinMaxScaler 적용
                data_to_standardize = features[:, col_idx].reshape(-1, 1)
                
                # 최소값과 최대값을 데이터에 추가
                data_with_bounds = np.vstack([
                    data_to_standardize,
                    [[min_val]],
                    [[max_val]]
                ])
                
                # MinMaxScaler 적용
                scaler = MinMaxScaler()
                standardized_data = scaler.fit_transform(data_with_bounds)
                
                # 원본 데이터 길이만큼만 반환 (마지막 2개는 최소/최대값이므로 제외)
                normalized_features[:, col_idx] = standardized_data[:len(features), 0]
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"특성 정규화 실패: {e}")
            raise
    
    def process_sensor_data(self, data: List[List[float]]) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        센서 데이터 전체 처리 파이프라인
        
        Args:
            data: 원본 센서 데이터
            
        Returns:
            (성공 여부, 처리된 데이터, 메시지)
        """
        try:
            # 1. 데이터 검증
            is_valid, validation_msg = self.validate_sensor_data(data)
            if not is_valid:
                return False, None, validation_msg
            
            # 2. 특성 추출
            features = self.extract_features(data)
            
            # 3. 특성 정규화
            normalized_features = self.normalize_features(features)
            
            return True, normalized_features, "데이터 처리 완료"
            
        except Exception as e:
            error_msg = f"데이터 처리 실패: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def get_data_statistics(self, data: List[List[float]]) -> Dict[str, Any]:
        """
        센서 데이터 통계 정보 반환
        
        Args:
            data: 센서 데이터
            
        Returns:
            통계 정보 딕셔너리
        """
        try:
            df = pd.DataFrame(data, columns=self.sensor_columns)
            
            stats = {
                'data_shape': df.shape,
                'feature_stats': {},
                'data_quality': {
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicate_rows': df.duplicated().sum()
                }
            }
            
            # 각 특성별 통계
            for col in self.feature_columns:
                if col in df.columns:
                    stats['feature_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'range': float(df[col].max() - df[col].min())
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 계산 실패: {e}")
            return {'error': str(e)}
    
    def save_processed_data(self, data: np.ndarray, filepath: str) -> bool:
        """
        처리된 데이터를 파일로 저장
        
        Args:
            data: 처리된 데이터
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            np.save(filepath, data)
            logger.info(f"처리된 데이터 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
            return False
    
    def load_processed_data(self, filepath: str) -> Optional[np.ndarray]:
        """
        처리된 데이터를 파일에서 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 데이터 또는 None
        """
        try:
            data = np.load(filepath)
            logger.info(f"처리된 데이터 로드 완료: {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return None

# 전역 프로세서 인스턴스 (싱글톤)
_data_processor = None

def get_data_processor() -> SensorDataProcessor:
    """싱글톤 패턴으로 데이터 프로세서 인스턴스 반환"""
    global _data_processor
    if _data_processor is None:
        _data_processor = SensorDataProcessor()
    return _data_processor 