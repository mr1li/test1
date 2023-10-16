import pandas as pd
import pandas as pd
import os
import numpy as np
from datetime import datetime,timedelta
def buhuo(file1,file2,file3,date,initial_inventory,a1,a2,a3,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db):
    df_asn = pd.read_csv(file1, encoding='latin1')
    # 读取第二个CSV文件
    df_inbound = pd.read_csv(file2, encoding='latin1')


    name = customer_name + '_' + customer_part_no + '_' + supplier_name + '_' + supplier_part_no + '_' + manufacture_name + '_' + site_db
    base_name = file3 + '//' + name
    forecast_name = base_name + '//' + 'forecast.csv'
    error_name = base_name + '//' + 'error' + '.npy'
    demand_data = pd.read_csv(forecast_name)
    demand_data = np.array((demand_data['mean']))


    df_demand = pd.read_csv(forecast_name, encoding='latin1')
    first_date = df_demand['date'].iloc[0]
    first_date = pd.to_datetime(first_date)

    df_asn = df_asn[(df_asn['customer_name'] == customer_name) & (df_asn['customer_part_no'] == customer_part_no)]
    df_inbound = df_inbound[
        (df_inbound['customer_name'] == customer_name) & (df_inbound['customer_part_no'] == customer_part_no)]
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime']).dt.date
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta']).dt.date
    # 检查 'asn_create_datetime' 列的数据类型
    merged_df = pd.merge(df_asn, df_inbound, on='asn_no')
    merged_df['datetime_diff'] = (merged_df['eta'] - merged_df['asn_create_datetime']).dt.days
    lead_time = round(merged_df['datetime_diff'].mean())
    three_months_ago = first_date - pd.DateOffset(months=3)
    # 仅保留在最近三个月内的数据
    three_months_ago = pd.to_datetime(three_months_ago)
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
    selected_rows = df_asn[df_asn['asn_create_datetime'].between(three_months_ago, first_date)]
    grouped = selected_rows.groupby('asn_create_datetime')
    group_count = len(grouped)
    # 计算频率
    freq = round((pd.to_datetime(first_date) - pd.to_datetime(three_months_ago)).days / group_count)
    # 计算每一行的'eta'列与'asn_create_datetime'之差的平均值
    print(freq, lead_time)



    error=np.load(error_name)
    demand_data2=error[int(a3*len(error))]*demand_data
    #先算出交付期内的数据
    daohuo=np.zeros(len(demand_data)+lead_time+freq)
    xiancun=np.zeros(len(demand_data)+lead_time+freq+1)
    buchong=np.zeros(len(demand_data)+lead_time+freq)
    avg_value = np.mean(demand_data2)
    for k in range(demand_data.shape[0], daohuo.shape[0]):
        demand_data2[k] = avg_value

    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])

    today = pd.to_datetime(date)
    for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
        days_diff = (today - row['asn_create_datetime']).days
        if days_diff <= lead_time and days_diff>0:
            daohuo[days_diff] = row['quantity']

    xiancun[0]=initial_inventory
    for i in range(lead_time):
            xiancun[i+1]=xiancun[i]+daohuo[i]-demand_data2[i]
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])

    # 按日期排序
    df_inbound = df_inbound.sort_values(by='eta')

    # 计算最近一年的开始日期
    one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)

    # 选择最近一年的数据
    recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]

    # 计算 'quantity' 列的 0.95 分位点
    quantile_95 = recent_data['quantity'].quantile(0.95)
    secure_level= quantile_95
    #开始补货
    for i in range(lead_time,len(demand_data)-freq):
        xiancun[i+1]=xiancun[i]-demand_data2[i]
        if xiancun[i+1]<=secure_level:
            q = 0
            chu = xiancun[i + 1]
            for n in range(i, i + freq):
                q = q + demand_data2[n]
            t = q - chu + secure_level
            t = max(t, zuixiaofahuo) + (zuixiaobaozhuang - (max(t, zuixiaofahuo) % zuixiaobaozhuang))
            buchong[i - lead_time] = t
            daohuo[i] = t
            xiancun[i+1] = t + chu





    #保存文件
    date_list = [today + timedelta(days=i) for i in range(len(demand_data))]
    date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
    out= {'date':date_strings,'buhuo': buchong}
    print(out)
    out=pd.DataFrame(out)
    out_name=base_name+'buhuo_'+name+'.csv'
    print(out_name)
    out.to_csv(out_name,index=False,line_terminator='\n')


    # filtered_data.to_csv('a.csv')
    # print(filtered_data)
if __name__ == "__main__":
    file1=r'E:\JusLink＆浙大数据20230523\v_islm_asn_1.csv'
    file2=r'E:\JusLink＆浙大数据20230523\v_islm_inbound.csv'
    file3='E://mxnet//tmp'
    date='2022-08-06'
    initial_inventory=10000
    zuixiaobaozhuang=100
    zuixiaofahuo=100
    xuqiumanzu=0.8
    customer_name='7e0d847c78df7e95d8e8779df7557fbc'
    customer_part_no='2fa8c3e5b021c333cff52f4280171a7b'
    supplier_name='a'
    supplier_part_no='a'
    manufacture_name='a'
    site_db='a'
    buhuo(file1,file2,file3,date,initial_inventory,zuixiaobaozhuang,zuixiaofahuo,xuqiumanzu,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db)
