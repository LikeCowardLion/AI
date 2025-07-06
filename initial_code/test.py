import socket
import numpy as np
from model_predict import puck_predict

# 기본 설정
UDP_IP = "127.0.0.1" # Local Server
UDP_PORT = 8080
UDP_RECEIVE_PORT = 8082
BUFFER_SIZE = 1024
CSV_PATH = "temp_input.csv"

# Unity로 예측 결과 전송 (UDP)
def sendToUnity(predictions):
    # UDP 선언
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if isinstance(predictions, np.ndarray):
        predictions = predictions.flatten().tolist()

    # 전송할 메시지 내용
    message = f"|{predictions[0]:.3f}|{predictions[1]:.3f}|" # 여기에 고유 번호를 추가해야할까?
    sock.sendto(message.encode(), (UDP_IP, UDP_RECEIVE_PORT))
    print(f"Sent to Unity: {message}")
    sock.close()
    
# 이후 추가적인 데이터 수집을 위한 CSV 저장 함수
def save_to_csv(data_lines):
    with open(CSV_PATH, "w") as f:
        f.write("Acc_X,Acc_Y,Acc_Z,Gyr_X,Gyr_Y,Gyr_Z\n")
        for line in data_lines:
            values = line.strip().split(",")
            if len(values) >= 6:
                acc_x = values[0]
                acc_y = values[1]
                acc_z = values[2]
                gyr_x = values[3]
                gyr_y = values[4]
                gyr_z = values[5]
                f.write(f"{acc_x},{acc_y},{acc_z},{gyr_x},{gyr_y},{gyr_z}\n")
    print(f"Saved input data to {CSV_PATH}")

# 메인 루프
def main():
    print(f"Listening for UDP on {UDP_IP}:{UDP_PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print("Waiting for data...")

    while True:
        
        try : 
            data, addr = sock.recvfrom(BUFFER_SIZE)
            print(f"Received from {addr}: {data.decode()}")
            decoded = data.decode("utf-8").strip()

            # 받은 데이터가 여러 줄인지 확인
            lines = decoded.split("\n")
            valid_lines = [line for line in lines if len(line.split(',')) == 6]
            
            print(f"Received {len(valid_lines)} valid lines")
        
            if len(valid_lines) != 10:
                print("❌ Not enough valid data rows. Skipping...")
                continue
                
            save_to_csv(valid_lines)  # CSV 저장
            predictions = puck_predict(CSV_PATH)

            # 결과가 numpy array일 경우 → 리스트 변환
            if isinstance(predictions, np.ndarray):
                predictions = predictions.flatten().tolist()

            sendToUnity(predictions)
            
        except Exception as e :
            print(f"Error: {e}")

if __name__ == "__main__":
    main()