import pandas as pd
import pandas as pd
import os
import numpy as np
from datetime import datetime,timedelta
def buhuoyanzheng(customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,
                  asn_name,inbound_name,demand_name,
                  inventory,baozhuang,fahuo,validation_length,pre_length):
    df_asn = pd.read_csv(asn_name, encoding='latin1')
    # 读取第二个CSV文件
    df_inbound = pd.read_csv(inbound_name, encoding='latin1')
    df_demand = pd.read_csv(demand_name, encoding='latin1')
    first_date=df_demand['date'].iloc[0]
    first_date = pd.to_datetime(first_date)
    df_asn = df_asn[(df_asn['customer_name'] == customer_name) & (df_asn['customer_part_no'] == customer_part_no)]
    df_inbound=df_inbound[(df_inbound['customer_name'] == customer_name) & (df_inbound['customer_part_no'] == customer_part_no)]
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime']).dt.date
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta']).dt.date
    # 检查 'asn_create_datetime' 列的数据类型
    merged_df = pd.merge(df_asn, df_inbound, on='asn_no')
    merged_df['datetime_diff'] =( merged_df['eta']-merged_df['asn_create_datetime'] ).dt.days
    lead_time = round(merged_df['datetime_diff'].mean())
    three_months_ago =  first_date - pd.DateOffset(months=3)
    # 仅保留在最近三个月内的数据
    three_months_ago=pd.to_datetime(three_months_ago)
    df_asn['asn_create_datetime'] = pd.to_datetime(df_asn['asn_create_datetime'])
    selected_rows = df_asn[df_asn['asn_create_datetime'].between(three_months_ago, first_date)]
    grouped = selected_rows.groupby('asn_create_datetime')
    group_count = len(grouped)
    print(group_count)
    # 计算频率
    freq =round( (pd.to_datetime(first_date )-  pd.to_datetime(three_months_ago)).days/(group_count+1))
    # 计算每一行的'eta'列与'asn_create_datetime'之差的平均值
    print(freq,lead_time)

    inbound = df_inbound
    inbound_grouped = inbound.groupby('eta')['quantity'].sum().reset_index()
    print(inbound_grouped)
    min_date = inbound_grouped['eta'].min()
    max_date = inbound_grouped['eta'].max()
    df_demand['date']=pd.to_datetime(df_demand['date'])
    date2 = df_demand['date'].max()
    print(df_demand)
    print(date2)
    date_range = pd.date_range(min_date, date2, freq='D')
    new_df = pd.DataFrame({'eta': date_range})
    new_df['eta'] = pd.to_datetime(new_df['eta'])
    print(new_df)
    result_df = result_df = pd.concat([new_df.set_index('eta'), inbound_grouped.set_index('eta')], axis=1,
                                      sort=True).fillna(0)
    inbound = result_df.reset_index()
    inbound = inbound[(inbound['eta'] >= first_date) & (inbound['eta'] <= date2)]
    inbound = np.array(inbound['quantity'])
    print(len(inbound))
    #补货验证：
    #1：首先创建好数据表格：
    T = validation_length // pre_length
    demand_yuanshi = np.array((df_demand['mean']))
    inbound_yuanshi=inbound
    outbound_yuanshi=np.array((df_demand['real']))
    xiancun_data=np.zeros((pre_length+lead_time+freq+1,T))
    buchong_data=np.zeros((pre_length+lead_time+freq,T))
    daohuo_data=np.zeros((pre_length+lead_time+freq,T))
    demand_data1=np.zeros((pre_length,T))
    demand_data2 = np.zeros((pre_length + lead_time + freq, T))
    outbound_data=np.zeros((pre_length,T))
    inbound_data=np.zeros((pre_length,T))
    demand_data1=demand_yuanshi.reshape(demand_data1.shape,order='F')
    outbound_data = outbound_yuanshi.reshape(outbound_data.shape, order='F')
    inbound_data = inbound.reshape(inbound_data.shape, order='F')
    tjyz=1.5
    initial=inventory
    df_inbound['eta'] = pd.to_datetime(df_inbound['eta'])

    # 按日期排序
    df_inbound = df_inbound.sort_values(by='eta')

    # 计算最近一年的开始日期
    one_year_ago = df_inbound['eta'].max() - pd.DateOffset(years=1)

    # 选择最近一年的数据
    recent_data = df_inbound[df_inbound['eta'] >= one_year_ago]

    # 计算 'quantity' 列的 0.95 分位点
    quantile_95 = recent_data['quantity'].quantile(0.97)

    anquanshuiwei = quantile_95
    for i in range(T):
        demand_data2[:demand_data1.shape[0],i]=demand_data1[:,i]*tjyz
        avg_value = np.mean(demand_data1[:, i])*tjyz
        for k in range(demand_data1.shape[0], demand_data2.shape[0]):
            demand_data2[k, i] = avg_value
        print(demand_data2[:, i])
        demand_data2[:, i]=np.sort(demand_data2[:, i])[::-1]
        print(demand_data2[:, i])

        if i==0:
            df_asn = df_asn.groupby('asn_create_datetime')['quantity'].sum().reset_index()
            for index, row in df_asn.sort_values(by='asn_create_datetime', ascending=False).iterrows():
                days_diff =(first_date- row['asn_create_datetime']).days
                if days_diff <= lead_time and days_diff>0:
                    print(days_diff)
                    daohuo_data[(lead_time-days_diff)%pre_length,(lead_time-days_diff)//pre_length] = row['quantity']
        else:
            if lead_time<i*pre_length:
             for j in range(lead_time):
                daohuo_data[j,i]=buchong_data[(i*pre_length+j-lead_time)%pre_length,(i*pre_length+j-lead_time)//pre_length]
            elif lead_time>(i+1)*pre_length:
                print(1)
            elif lead_time>i*pre_length and lead_time<(i+1)*pre_length:
                for j in range(lead_time-i*pre_length,lead_time):
                    daohuo_data[j, i] = buchong_data[(i * pre_length + j - lead_time) % pre_length, (i * pre_length + j - lead_time) // pre_length]
        xiancun_data[0,i]=inventory
        for h in range(lead_time):
            xiancun_data[h+1,i]=xiancun_data[h,i]+daohuo_data[h,i]-demand_data2[h,i]
        for m in range(lead_time, len(demand_data1)+lead_time ):
            xiancun_data[m+1,i]=xiancun_data[m,i]-demand_data2[m,i]

            if xiancun_data[m + 1, i] <=anquanshuiwei:
                q = 0
                chu = xiancun_data[m+1,i]
                for n in range(m, m + freq):
                    q = q + demand_data2[n,i]
                t = q - chu+anquanshuiwei
                t = max(t, fahuo) + (baozhuang - (max(t, fahuo) % baozhuang))
                buchong_data[m - lead_time,i] = t
                daohuo_data[m,i] = t
                xiancun_data[m+1,i] = t + chu
        #动态更新：
        inventory=inventory+sum(daohuo_data[:pre_length,i])-sum(outbound_data[:,i])
        true=sum(outbound_data[:,i])
        pre=sum(demand_data2[:pre_length,i])
        if true<pre:
            tjyz=max(1.3,tjyz-0.1)
        else:
            tjyz=min(2,tjyz+0.1)


    print(daohuo_data)
    date_list = [first_date + timedelta(days=i) for i in range(len(demand_yuanshi))]
    date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
    buchong_data=buchong_data[:pre_length,:]
    buchong=buchong_data.reshape(demand_yuanshi.shape,order='F')
    out= {'date':date_strings,'buhuo': buchong}
    print(out)
    out=pd.DataFrame(out)
    out_name='buhuo_'+'.csv'
    print(out_name)
    out.to_csv(out_name,index=False,line_terminator='\n')
    #评价体系：计算从first_date到结束的真实与算法库存量，先比需求满足，在比库存下降。
    #首先计算两个库存量：
    #第一个真实的：
    selected_rows = daohuo_data[:pre_length, :]
    sf_inbound=selected_rows.reshape(outbound_yuanshi.shape,order='F')
    real_inty=[initial+inbound_yuanshi[0]-outbound_yuanshi[0]]
    sf_inty=[initial+sf_inbound[0]-outbound_yuanshi[0]]
    for i in range(1, len(outbound_yuanshi)):
       real_inty.append(real_inty[-1] + inbound[i] - outbound_yuanshi[i])
    for i in range(1, len(outbound_yuanshi)):
       sf_inty.append(sf_inty[-1] + sf_inbound[i] - outbound_yuanshi[i])
    print(real_inty)
    print(sf_inty)
    a= quantile_95
    print(a)
    count_a = len([x for x in real_inty if x >a])
    ratio_a = count_a / len(real_inty)
    count_b = len([x for x in sf_inty if x > a])
    ratio_b = count_b / len(sf_inty)
    lower_ratio = [(b - x) / x for x, b in zip(real_inty, sf_inty) if (x > a and b>a) ]
    mean=sum(lower_ratio)/len(lower_ratio)
    print(ratio_b,ratio_a,mean,lower_ratio)


customer_name='b44d54cbf91dca39e00e2037dcc8a5c6'
customer_part_no='42accf484119a16be6099bbe59d0f2d5'
supplier_name=0
supplier_part_no=0
manufacture_name=0
site_db=0
asn_name=r'E:\JusLink＆浙大数据20230523\v_islm_asn_1.csv'
inbound_name=r'E:\JusLink＆浙大数据20230523\v_islm_inbound.csv'
demand_name=r'E:\mxnet\b44d54cbf91dca39e00e2037dcc8a5c6_42accf484119a16be6099bbe59d0f2d5_a_a_a_a.csv'
inventory=100000 #1120000,100000,940000,1400000
baozhuang=1
fahuo=1
pre_length=7
validation_length=7*4*3
buhuoyanzheng(customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,
                  asn_name,inbound_name,demand_name,
                  inventory,baozhuang,fahuo,validation_length,pre_length)