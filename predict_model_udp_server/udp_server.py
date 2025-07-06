import socket
import numpy as np
import time
import threading
from typing import List, Optional, Tuple
import logging
from model_predict import get_predictor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PuckUDPServer:
    """스마트 디바이스 위치 예측을 위한 최적화된 UDP 서버"""
    
    def __init__(self, 
                 listen_ip: str = "127.0.0.1",
                 listen_port: int = 8080,
                 unity_ip: str = "127.0.0.1", 
                 unity_port: int = 8082,
                 buffer_size: int = 1024):
        """
        Args:
            listen_ip: 수신할 IP 주소
            listen_port: 수신할 포트
            unity_ip: Unity로 전송할 IP 주소
            unity_port: Unity로 전송할 포트
            buffer_size: UDP 버퍼 크기
        """
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.unity_ip = unity_ip
        self.unity_port = unity_port
        self.buffer_size = buffer_size
        
        # 예측기 초기화
        self.predictor = get_predictor()
        
        # 통계 정보
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_processing_time': 0.0
        }
        
        # UDP 소켓
        self.socket = None
        self.running = False
        
    def _parse_sensor_data(self, data: str) -> Optional[List[List[float]]]:
        """
        센서 데이터 파싱 및 검증
        
        Args:
            data: 수신된 데이터 문자열
            
        Returns:
            파싱된 센서 데이터 또는 None
        """
        try:
            lines = data.strip().split('\n')
            valid_lines = []
            
            for line in lines:
                values = line.strip().split(',')
                if len(values) == 6:  # Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z
                    try:
                        # 모든 값을 float로 변환
                        float_values = [float(val) for val in values]
                        valid_lines.append(float_values)
                    except ValueError:
                        logger.warning(f"잘못된 데이터 형식: {line}")
                        continue
            
            if len(valid_lines) == 10:
                return valid_lines
            else:
                logger.warning(f"유효한 데이터 행 수 부족: {len(valid_lines)}/10")
                return None
                
        except Exception as e:
            logger.error(f"데이터 파싱 실패: {e}")
            return None
    
    def _send_to_unity(self, predictions: np.ndarray) -> bool:
        """
        Unity로 예측 결과 전송
        
        Args:
            predictions: 예측 결과
            
        Returns:
            전송 성공 여부
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)  # 1초 타임아웃
            
            # 예측 결과를 리스트로 변환
            if isinstance(predictions, np.ndarray):
                pred_list = predictions.flatten().tolist()
            else:
                pred_list = predictions
            
            # 메시지 형식: |x|y|
            message = f"|{pred_list[0]:.3f}|{pred_list[1]:.3f}|"
            
            sock.sendto(message.encode('utf-8'), (self.unity_ip, self.unity_port))
            sock.close()
            
            logger.info(f"Unity로 전송: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Unity 전송 실패: {e}")
            return False
    
    def _process_data(self, data: str, client_addr: Tuple[str, int]) -> None:
        """
        데이터 처리 및 예측 수행
        
        Args:
            data: 수신된 데이터
            client_addr: 클라이언트 주소
        """
        start_time = time.time()
        
        try:
            logger.info(f"클라이언트 {client_addr}로부터 데이터 수신")
            
            # 데이터 파싱
            sensor_data = self._parse_sensor_data(data)
            if sensor_data is None:
                logger.warning("유효하지 않은 데이터 - 처리 건너뜀")
                return
            
            # 예측 수행
            predictions = self.predictor.predict(sensor_data)
            
            # Unity로 전송
            if self._send_to_unity(predictions):
                self.stats['successful_predictions'] += 1
                logger.info(f"예측 성공: {predictions.flatten()}")
            else:
                self.stats['failed_predictions'] += 1
                logger.error("Unity 전송 실패")
            
            # 통계 업데이트
            self.stats['total_predictions'] += 1
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_predictions'] - 1) + processing_time) 
                / self.stats['total_predictions']
            )
            
        except Exception as e:
            self.stats['failed_predictions'] += 1
            logger.error(f"데이터 처리 실패: {e}")
    
    def _print_stats(self) -> None:
        """통계 정보 출력"""
        logger.info("=== 서버 통계 ===")
        logger.info(f"총 예측 시도: {self.stats['total_predictions']}")
        logger.info(f"성공: {self.stats['successful_predictions']}")
        logger.info(f"실패: {self.stats['failed_predictions']}")
        if self.stats['total_predictions'] > 0:
            success_rate = (self.stats['successful_predictions'] / self.stats['total_predictions']) * 100
            logger.info(f"성공률: {success_rate:.1f}%")
            logger.info(f"평균 처리 시간: {self.stats['avg_processing_time']*1000:.1f}ms")
        logger.info("================")
    
    def start(self) -> None:
        """UDP 서버 시작"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.listen_ip, self.listen_port))
            self.socket.settimeout(1.0)  # 1초 타임아웃
            
            self.running = True
            logger.info(f"UDP 서버 시작 - {self.listen_ip}:{self.listen_port}")
            logger.info(f"Unity 전송 주소 - {self.unity_ip}:{self.unity_port}")
            
            # 통계 출력 스레드 시작
            stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            stats_thread.start()
            
            while self.running:
                try:
                    data, addr = self.socket.recvfrom(self.buffer_size)
                    decoded_data = data.decode('utf-8')
                    
                    # 별도 스레드에서 데이터 처리 (비동기)
                    process_thread = threading.Thread(
                        target=self._process_data, 
                        args=(decoded_data, addr),
                        daemon=True
                    )
                    process_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"데이터 수신 오류: {e}")
                    
        except Exception as e:
            logger.error(f"서버 시작 실패: {e}")
        finally:
            self.stop()
    
    def _stats_loop(self) -> None:
        """주기적으로 통계 정보 출력"""
        while self.running:
            time.sleep(30)  # 30초마다 통계 출력
            self._print_stats()
    
    def stop(self) -> None:
        """서버 중지"""
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info("UDP 서버 중지")

def main():
    """메인 함수"""
    try:
        # 서버 설정
        server = PuckUDPServer(
            listen_ip="127.0.0.1",
            listen_port=8080,
            unity_ip="127.0.0.1",
            unity_port=8082,
            buffer_size=1024
        )
        
        # 서버 시작
        server.start()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 서버 중지")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")

if __name__ == "__main__":
    main() 