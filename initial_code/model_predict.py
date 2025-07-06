from extract import extract_data
from cut_length_data import cut_length
import keras
import tensorflow as tf
import numpy as np
import pickle

def load_puck_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def puck_predict(input_file):
    model = load_puck_model('model123.h5')
    model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
    # 예측 시 스케일러 불러오기
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # 데이터 전처리
    data = []
    
    file = extract_data(input_file)
    insert_data = cut_length(file)
    data.append(insert_data)
    data = np.array(data)
    
    score = model.predict(data)
    result = scaler.inverse_transform(score)

    return result