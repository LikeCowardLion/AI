import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """시스템 성능 모니터링 및 최적화를 위한 클래스"""
    
    def __init__(self, max_history: int = 1000):
        """
        Args:
            max_history: 저장할 최대 히스토리 개수
        """
        self.max_history = max_history
        
        # 성능 지표 저장소
        self.metrics = {
            'cpu_usage': deque(maxlen=max_history),
            'memory_usage': deque(maxlen=max_history),
            'processing_times': deque(maxlen=max_history),
            'prediction_accuracy': deque(maxlen=max_history),
            'network_latency': deque(maxlen=max_history)
        }
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 콜백 함수들
        self.callbacks = {
            'high_cpu': [],
            'high_memory': [],
            'slow_processing': [],
            'low_accuracy': []
        }
        
        # 임계값 설정
        self.thresholds = {
            'cpu_usage': 80.0,      # CPU 사용률 80% 이상
            'memory_usage': 85.0,   # 메모리 사용률 85% 이상
            'processing_time': 100.0, # 처리 시간 100ms 이상
            'prediction_accuracy': 0.7  # 예측 정확도 70% 이하
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """
        성능 모니터링 시작
        
        Args:
            interval: 모니터링 간격 (초)
        """
        if self.is_monitoring:
            logger.warning("모니터링이 이미 실행 중입니다.")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"성능 모니터링 시작 (간격: {interval}초)")
    
    def stop_monitoring(self):
        """성능 모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("성능 모니터링 중지")
    
    def _monitor_loop(self, interval: float):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 시스템 리소스 모니터링
                self._collect_system_metrics()
                
                # 임계값 체크 및 알림
                self._check_thresholds()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics['cpu_usage'].append({
                'timestamp': time.time(),
                'value': cpu_percent
            })
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics['memory_usage'].append({
                'timestamp': time.time(),
                'value': memory_percent
            })
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
    
    def _check_thresholds(self):
        """임계값 체크 및 알림"""
        try:
            # CPU 사용률 체크
            if self.metrics['cpu_usage']:
                current_cpu = self.metrics['cpu_usage'][-1]['value']
                if current_cpu > self.thresholds['cpu_usage']:
                    self._trigger_callbacks('high_cpu', current_cpu)
            
            # 메모리 사용률 체크
            if self.metrics['memory_usage']:
                current_memory = self.metrics['memory_usage'][-1]['value']
                if current_memory > self.thresholds['memory_usage']:
                    self._trigger_callbacks('high_memory', current_memory)
            
            # 처리 시간 체크
            if self.metrics['processing_times']:
                avg_processing_time = np.mean([m['value'] for m in self.metrics['processing_times']])
                if avg_processing_time > self.thresholds['processing_time']:
                    self._trigger_callbacks('slow_processing', avg_processing_time)
            
            # 예측 정확도 체크
            if self.metrics['prediction_accuracy']:
                avg_accuracy = np.mean([m['value'] for m in self.metrics['prediction_accuracy']])
                if avg_accuracy < self.thresholds['prediction_accuracy']:
                    self._trigger_callbacks('low_accuracy', avg_accuracy)
                    
        except Exception as e:
            logger.error(f"임계값 체크 실패: {e}")
    
    def _trigger_callbacks(self, event_type: str, value: float):
        """콜백 함수 실행"""
        for callback in self.callbacks[event_type]:
            try:
                callback(event_type, value)
            except Exception as e:
                logger.error(f"콜백 실행 실패: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        콜백 함수 추가
        
        Args:
            event_type: 이벤트 타입 ('high_cpu', 'high_memory', 'slow_processing', 'low_accuracy')
            callback: 실행할 콜백 함수
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"콜백 추가: {event_type}")
        else:
            logger.error(f"알 수 없는 이벤트 타입: {event_type}")
    
    def record_processing_time(self, processing_time: float):
        """
        처리 시간 기록
        
        Args:
            processing_time: 처리 시간 (밀리초)
        """
        self.metrics['processing_times'].append({
            'timestamp': time.time(),
            'value': processing_time
        })
    
    def record_prediction_accuracy(self, accuracy: float):
        """
        예측 정확도 기록
        
        Args:
            accuracy: 예측 정확도 (0.0 ~ 1.0)
        """
        self.metrics['prediction_accuracy'].append({
            'timestamp': time.time(),
            'value': accuracy
        })
    
    def record_network_latency(self, latency: float):
        """
        네트워크 지연시간 기록
        
        Args:
            latency: 네트워크 지연시간 (밀리초)
        """
        self.metrics['network_latency'].append({
            'timestamp': time.time(),
            'value': latency
        })
    
    def get_performance_summary(self) -> Dict:
        """
        성능 요약 정보 반환
        
        Returns:
            성능 요약 딕셔너리
        """
        try:
            summary = {}
            
            # CPU 사용률 통계
            if self.metrics['cpu_usage']:
                cpu_values = [m['value'] for m in self.metrics['cpu_usage']]
                summary['cpu'] = {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': np.mean(cpu_values),
                    'max': np.max(cpu_values),
                    'min': np.min(cpu_values)
                }
            
            # 메모리 사용률 통계
            if self.metrics['memory_usage']:
                memory_values = [m['value'] for m in self.metrics['memory_usage']]
                summary['memory'] = {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': np.mean(memory_values),
                    'max': np.max(memory_values),
                    'min': np.min(memory_values)
                }
            
            # 처리 시간 통계
            if self.metrics['processing_times']:
                processing_values = [m['value'] for m in self.metrics['processing_times']]
                summary['processing_time'] = {
                    'current': processing_values[-1] if processing_values else 0,
                    'average': np.mean(processing_values),
                    'max': np.max(processing_values),
                    'min': np.min(processing_values)
                }
            
            # 예측 정확도 통계
            if self.metrics['prediction_accuracy']:
                accuracy_values = [m['value'] for m in self.metrics['prediction_accuracy']]
                summary['prediction_accuracy'] = {
                    'current': accuracy_values[-1] if accuracy_values else 0,
                    'average': np.mean(accuracy_values),
                    'max': np.max(accuracy_values),
                    'min': np.min(accuracy_values)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"성능 요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def get_recommendations(self) -> List[str]:
        """
        성능 최적화 권장사항 반환
        
        Returns:
            권장사항 리스트
        """
        recommendations = []
        summary = self.get_performance_summary()
        
        try:
            # CPU 사용률 권장사항
            if 'cpu' in summary and summary['cpu']['average'] > 70:
                recommendations.append("CPU 사용률이 높습니다. 더 효율적인 알고리즘 사용을 고려하세요.")
            
            # 메모리 사용률 권장사항
            if 'memory' in summary and summary['memory']['average'] > 80:
                recommendations.append("메모리 사용률이 높습니다. 메모리 누수 확인 및 최적화가 필요합니다.")
            
            # 처리 시간 권장사항
            if 'processing_time' in summary and summary['processing_time']['average'] > 50:
                recommendations.append("처리 시간이 느립니다. 알고리즘 최적화 또는 병렬 처리 도입을 고려하세요.")
            
            # 예측 정확도 권장사항
            if 'prediction_accuracy' in summary and summary['prediction_accuracy']['average'] < 0.8:
                recommendations.append("예측 정확도가 낮습니다. 모델 재학습 또는 데이터 품질 개선이 필요합니다.")
            
            if not recommendations:
                recommendations.append("현재 성능이 양호합니다.")
                
        except Exception as e:
            logger.error(f"권장사항 생성 실패: {e}")
            recommendations.append("권장사항 생성 중 오류가 발생했습니다.")
        
        return recommendations
    
    def export_metrics(self, filepath: str) -> bool:
        """
        메트릭을 파일로 내보내기
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            import json
            
            export_data = {
                'timestamp': time.time(),
                'metrics': {k: list(v) for k, v in self.metrics.items()},
                'summary': self.get_performance_summary(),
                'recommendations': self.get_recommendations()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"메트릭 내보내기 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"메트릭 내보내기 실패: {e}")
            return False

# 전역 모니터 인스턴스 (싱글톤)
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """싱글톤 패턴으로 성능 모니터 인스턴스 반환"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor 