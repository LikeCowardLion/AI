#!/usr/bin/env python3
"""
최적화된 스마트 디바이스 위치 예측 코드 테스트 스크립트
"""

import numpy as np
import time
import logging
from optimized_model_predict import get_predictor
from data_processor import get_data_processor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """테스트용 센서 데이터 생성"""
    # 10행 x 6열의 랜덤 센서 데이터 생성
    np.random.seed(10)  # 재현 가능한 결과를 위해 시드 설정
    
    test_data = []
    for i in range(10):
        # 가속도 데이터 (Acc_X, Acc_Y, Acc_Z)
        acc_x = np.random.uniform(-4, 4)
        acc_y = np.random.uniform(-4, 3)
        acc_z = np.random.uniform(-4, 4)
        
        # 자이로스코프 데이터 (Gyr_X, Gyr_Y, Gyr_Z)
        gyr_x = np.random.uniform(-2000, 2000)
        gyr_y = np.random.uniform(-2000, 2000)
        gyr_z = np.random.uniform(-2300, 2300)
        
        test_data.append([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z])
    
    return test_data

def test_data_processor():
    """데이터 프로세서 테스트"""
    logger.info("=== 데이터 프로세서 테스트 시작 ===")
    
    try:
        # 데이터 프로세서 초기화
        processor = get_data_processor()
        logger.info("✅ 데이터 프로세서 초기화 성공")
        
        # 테스트 데이터 생성
        test_data = create_test_data()
        logger.info(f"✅ 테스트 데이터 생성 완료: {len(test_data)}행 x {len(test_data[0])}열")
        
        # 데이터 검증
        is_valid, validation_msg = processor.validate_sensor_data(test_data)
        logger.info(f"✅ 데이터 검증: {validation_msg}")
        
        # 데이터 통계
        stats = processor.get_data_statistics(test_data)
        logger.info(f"✅ 데이터 통계: {stats['data_shape']}")
        
        # 데이터 처리
        success, processed_data, message = processor.process_sensor_data(test_data)
        if success:
            logger.info(f"✅ 데이터 처리 성공: {processed_data.shape}")
            logger.info(f"처리된 데이터 샘플:\n{processed_data[:3]}")
        else:
            logger.error(f"❌ 데이터 처리 실패: {message}")
            return False
        
        logger.info("=== 데이터 프로세서 테스트 완료 ===\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터 프로세서 테스트 실패: {e}")
        return False

def test_model_predictor():
    """모델 예측기 테스트"""
    logger.info("=== 모델 예측기 테스트 시작 ===")
    
    try:
        # 예측기 초기화
        predictor = get_predictor()
        logger.info("✅ 모델 예측기 초기화 성공")
        
        # 테스트 데이터 생성
        test_data = create_test_data()
        logger.info(f"✅ 테스트 데이터 생성 완료: {len(test_data)}행 x {len(test_data[0])}열")
        
        # 예측 수행
        start_time = time.time()
        predictions = predictor.predict(test_data)
        processing_time = (time.time() - start_time) * 1000  # 밀리초 단위
        
        logger.info(f"✅ 예측 성공: {predictions.flatten()}")
        logger.info(f"✅ 처리 시간: {processing_time:.2f}ms")
        
        # 예측 결과 검증
        if predictions.shape == (1, 2):  # (배치, 좌표) 형태
            logger.info("✅ 예측 결과 형태 정상")
        else:
            logger.warning(f"⚠️ 예측 결과 형태 이상: {predictions.shape}")
        
        logger.info("=== 모델 예측기 테스트 완료 ===\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 예측기 테스트 실패: {e}")
        return False

def test_performance():
    """성능 테스트"""
    logger.info("=== 성능 테스트 시작 ===")
    
    try:
        predictor = get_predictor()
        test_data = create_test_data()
        
        # 여러 번 예측하여 평균 처리 시간 측정
        times = []
        for i in range(10):
            start_time = time.time()
            predictions = predictor.predict(test_data)
            processing_time = (time.time() - start_time) * 1000
            times.append(processing_time)
            
            if i < 3:  # 처음 3개 결과만 출력
                logger.info(f"예측 {i+1}: {predictions.flatten()} ({processing_time:.2f}ms)")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"✅ 평균 처리 시간: {avg_time:.2f}ms ± {std_time:.2f}ms")
        logger.info(f"✅ 최소 처리 시간: {np.min(times):.2f}ms")
        logger.info(f"✅ 최대 처리 시간: {np.max(times):.2f}ms")
        
        # 성능 기준 체크
        if avg_time < 100:  # 100ms 이하
            logger.info("✅ 성능 기준 통과 (100ms 이하)")
        else:
            logger.warning(f"⚠️ 성능 기준 미달: {avg_time:.2f}ms")
        
        logger.info("=== 성능 테스트 완료 ===\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ 성능 테스트 실패: {e}")
        return False

def test_error_handling():
    """에러 처리 테스트"""
    logger.info("=== 에러 처리 테스트 시작 ===")
    
    try:
        predictor = get_predictor()
        
        # 잘못된 데이터로 테스트
        invalid_data = [
            [1, 2, 3, 4, 5],  # 컬럼 수 부족
        ]
        
        try:
            predictor.predict(invalid_data)
            logger.error("❌ 잘못된 데이터에 대한 예외 처리가 작동하지 않음")
            return False
        except Exception as e:
            logger.info(f"✅ 잘못된 데이터 예외 처리 정상: {type(e).__name__}")
        
        # 빈 데이터로 테스트
        empty_data = []
        
        try:
            predictor.predict(empty_data)
            logger.error("❌ 빈 데이터에 대한 예외 처리가 작동하지 않음")
            return False
        except Exception as e:
            logger.info(f"✅ 빈 데이터 예외 처리 정상: {type(e).__name__}")
        
        logger.info("=== 에러 처리 테스트 완료 ===\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ 에러 처리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 최적화된 스마트 디바이스 위치 예측 코드 테스트 시작")
    
    test_results = []
    
    # 각 테스트 실행
    test_results.append(("데이터 프로세서", test_data_processor()))
    test_results.append(("모델 예측기", test_model_predictor()))
    test_results.append(("성능 테스트", test_performance()))
    test_results.append(("에러 처리", test_error_handling()))
    
    # 결과 요약
    logger.info("=== 테스트 결과 요약 ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 모든 테스트 통과! 코드가 정상적으로 작동합니다.")
    else:
        logger.warning("⚠️ 일부 테스트 실패. 코드를 확인해주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 