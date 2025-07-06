import pandas as pd

# Acc_X, Acc_Y, Gyr_Z 추출하는 함수
def extract_data(input_file):
    file = pd.read_csv(input_file)

    # 삭제할 컬럼
    drop_cols = ['Time', 'Acc_Z', 'Gyr_X', 'Gyr_Y']
    existing_cols = [col for col in drop_cols if col in file.columns]
    file.drop(existing_cols, axis=1, inplace=True)
    file.reset_index(drop=True, inplace=True)
    file = file.astype(float)
    return file