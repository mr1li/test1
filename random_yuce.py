import pandas as pd
import numpy as np
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def demand_validation(customer_name, customer_part_no, supplier_name, supplier_part_no, manufacture_name, site_db,
                      filename1, filename2, filename3, filename4, filename5, filename6,
                      pre_length, con_length,datetime):
    # 提取料五项数据
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df3 = pd.read_csv(filename3)
    df4 = pd.read_csv(filename4)
    df5 = pd.read_csv(filename5)
    df6 = pd.read_csv(filename6)
    df1['eta'] = pd.to_datetime(df1['eta']).dt.strftime('%Y-%m-%d')
    df2['eta'] = pd.to_datetime(df2['eta']).dt.strftime('%Y-%m-%d')
    df3['eta'] = pd.to_datetime(df3['eta']).dt.strftime('%Y-%m-%d')
    df4['eta'] = pd.to_datetime(df4['eta']).dt.strftime('%Y-%m-%d')
    df5['eta'] = pd.to_datetime(df5['eta']).dt.strftime('%Y-%m-%d')
    df6['eta'] = pd.to_datetime(df6['eta']).dt.strftime('%Y-%m-%d')
    filtered_data = df1[(df1['customer_name'] == customer_name) & (df1['customer_part_no'] == customer_part_no)]
    df2_data = df2[(df2['customer_name'] == customer_name) & (df2['customer_part_no'] == customer_part_no)]
    df3_data = df3[(df3['customer_name'] == customer_name) & (df3['customer_part_no'] == customer_part_no)]
    df4_data = df4[(df4['customer_name'] == customer_name) & (df4['customer_part_no'] == customer_part_no)]
    df5_data = df5[(df5['customer_name'] == customer_name) & (df5['customer_part_no'] == customer_part_no)]
    df6_data = df6[(df6['customer_name'] == customer_name) & (df6['customer_part_no'] == customer_part_no)]
    filtered_data = pd.concat([df2_data, df3_data, df4_data, df5_data, df6_data, filtered_data], axis=0)
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['eta'] = pd.to_datetime(filtered_data['eta'])
    filtered_data = filtered_data.groupby(['eta'], as_index=False)['quantity'].sum()
    filtered_data = filtered_data.sort_values(by='eta', ascending=True)
    df = pd.DataFrame(filtered_data)  # df为该料五项数据

    date_range = pd.date_range(df['eta'].min(), df['eta'].max(), freq='D')
    # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
    df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})

    # 特征工程：
    df1['2_mean'] = df1['quantity'].rolling(window=2, min_periods=1).mean()
    df1['3_mean'] = df1['quantity'].rolling(window=3, min_periods=1).mean()
    df1['4_mean'] = df1['quantity'].rolling(window=4, min_periods=1).mean()
    df1['7_mean'] = df1['quantity'].rolling(window=7, min_periods=1).mean()
    df1['14_mean'] = df1['quantity'].rolling(window=14, min_periods=1).mean()
    df1['28_mean'] = df1['quantity'].rolling(window=28, min_periods=1).mean()
    df1['3_max'] = df1['quantity'].rolling(window=3, min_periods=1).max()
    df1['3_min'] = df1['quantity'].rolling(window=3, min_periods=1).min()
    df1['4_max'] = df1['quantity'].rolling(window=4, min_periods=1).max()
    df1['4_min'] = df1['quantity'].rolling(window=4, min_periods=1).min()
    df1['5_max'] = df1['quantity'].rolling(window=5, min_periods=1).max()
    df1['5_min'] = df1['quantity'].rolling(window=5, min_periods=1).min()
    df1['lag'] = df1['quantity'].shift().fillna(0)
    start_time = time.time()
    X = []
    Y = []

    for i in range(len(df) - con_length - pre_length + 1):
        X.append(df.iloc[i:i + con_length, 1:])
        Y.append(df.iloc[i + con_length:i + con_length + pre_length, 1])
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    X_test=df.iloc[-con_length:, 1:]
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test=X_test.reshape(1, -1)
    X_train=X
    Y_train=Y
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # #
    # Fit Randomforest
    forest = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
    forest.fit(X_train, Y_train)
    y_test_pred = forest.predict(X_test)
    ytrue = [sublist2 for sublist1 in y_test_pred for sublist2 in sublist1]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"第二个代码段执行时间为: {execution_time} 秒")
    out = pd.DataFrame({ 'pre': ytrue})

    out.to_csv('suijisenlin5.csv')




customer_name1 = '7e0d847c78df7e95d8e8779df7557fbc'
customer_part_no1 = 'f9743fbed9a869c263409a35b1e8d928'
#
supplier_name = 'a'
supplier_part_no = 'a'
manufacture_name = 'a'
site_db = 'a'
filename1 = r'E:\data1\yuanshi3\split_v_islm_outbound_2021\v_islm_outbound_2021_1_1000001.csv'  # outbound文件
filename2 = r'E:\data1\yuanshi3\split_v_islm_outbound_2021\v_islm_outbound_2021_1000001_1048576.csv'  # outbound文件
filename3 = r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_2000001_2449400.csv'  # outbound文件
filename4 = r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_1_1000001.csv'  # outbound文件
filename5 = r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_1000001_2000001.csv'  # outbound文件
filename6 = r'E:\JusLink＆浙大数据20230523\v_islm_outbound_2023.csv'  # outbound文件
pre_length = 7
con_length = 7
for i in range(1, 2):
    print(i)
    customer_name = globals()['customer_name' + str(i)]
    customer_part_no = globals()['customer_part_no' + str(i)]
    print(customer_name, customer_part_no)
    supplier_name = supplier_name
    supplier_part_no = supplier_part_no
    manufacture_name = manufacture_name
    site_db = 'a'
    demand_validation(customer_name, customer_part_no, supplier_name, supplier_part_no, manufacture_name, site_db,
                      filename1, filename2, filename3, filename4, filename5, filename6,
                      pre_length, con_length)

