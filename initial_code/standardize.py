from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
# MinMax 정규화

def standardize_data(df, col, min_value, max_value):
    #global scaler
    scaler = MinMaxScaler()
    
    # 표준화할 데이터 준비 (예시로 특정 열만 사용)
    df.reset_index(drop=True, inplace=True)
    data_to_standardize = df.loc[:,[col]]

    # 최대값, 최솟값 삽입 (Series를 DataFrame에 추가)
    data_to_standardize.loc[len(data_to_standardize)] = min_value
    data_to_standardize.loc[len(data_to_standardize)] = max_value
    
    # 데이터를 표준화
    standardized_data = scaler.fit_transform(data_to_standardize)

    # 표준화된 데이터를 DataFrame으로 변환
    standardized_df = pd.DataFrame(standardized_data, columns=[col])

    # 원래 데이터에 표준화된 데이터로 업데이트
    df.loc[:, [col]] = standardized_df[[col]].astype(float)

    return df[[col]]