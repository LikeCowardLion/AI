from standardize import standardize_data

# 구간을 자르는 함수
def cut_length(input_file):
    normal_data = input_file.copy()

    for col in normal_data.columns:
        if col == "Acc_X":
            normal_data[col] = standardize_data(normal_data, col, -3.99, 4.02)
        elif col == "Acc_Y":
            normal_data[col] = standardize_data(normal_data, col, -4.019, 2.918)
        else:
            normal_data[col] = standardize_data(normal_data, col, -2297.985, 2289.371)
        
    return normal_data.values