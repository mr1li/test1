from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from pathlib import Path
from gluonts.mx.trainer import Trainer
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
def demand_validation(customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,
                      filename1,filename2,filename3,filename4,filename5,filename6,
                      pre_length,validation_length):
    df = pd.read_csv(filename1)
    df2=pd.read_csv(filename2)
    df3=pd.read_csv(filename3)
    df4=pd.read_csv(filename4)
    df5 = pd.read_csv(filename5)
    df6 = pd.read_csv(filename6)
    df['eta'] = pd.to_datetime(df['eta']).dt.strftime('%Y-%m-%d')
    df2['eta'] = pd.to_datetime(df2['eta']).dt.strftime('%Y-%m-%d')
    df3['eta'] = pd.to_datetime(df3['eta']).dt.strftime('%Y-%m-%d')
    df4['eta'] = pd.to_datetime(df4['eta']).dt.strftime('%Y-%m-%d')
    df5['eta'] = pd.to_datetime(df5['eta']).dt.strftime('%Y-%m-%d')
    df6['eta'] = pd.to_datetime(df6['eta']).dt.strftime('%Y-%m-%d')
    filtered_data = df[(df['customer_name'] == customer_name) & (df['customer_part_no'] == customer_part_no)]
    df2_data=df2[(df2['customer_name'] == customer_name) & (df2['customer_part_no'] == customer_part_no)]
    df3_data=df3[(df3['customer_name'] == customer_name) & (df3['customer_part_no'] == customer_part_no)]
    df4_data = df4[(df4['customer_name'] == customer_name) & (df4['customer_part_no'] == customer_part_no)]
    df5_data = df5[(df5['customer_name'] == customer_name) & (df5['customer_part_no'] == customer_part_no)]
    df6_data = df6[(df6['customer_name'] == customer_name) & (df6['customer_part_no'] == customer_part_no)]
    filtered_data=pd.concat([df2_data,df3_data,df4_data,df5_data,filtered_data],axis=0)
    filtered_data = filtered_data.reset_index(drop=True)
    filtered_data['eta'] = pd.to_datetime(filtered_data['eta'])
    filtered_data = filtered_data.groupby(['eta'], as_index=False)['quantity'].sum()
    filtered_data=filtered_data.sort_values(by='eta', ascending=True)
    df = pd.DataFrame(filtered_data)
    # # 将 'eta' 列设置为日期类型
    df['quantity']=df['quantity']
    # # 创建一个完整的日期范围60,160
    date_range = pd.date_range(df['eta'].min(), df['eta'].max(), freq='D')
    # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
    df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})
    # 平滑数据
    df1=df.copy()
    df1['7_mean'] = df1['quantity'].rolling(window=7, min_periods=1).mean()
    df1['28_mean'] = df1['quantity'].rolling(window=3, min_periods=1).mean()
    df1['14_mean'] = df1['quantity'].rolling(window=14, min_periods=1).mean()
    df1['4_max'] = df1['quantity'].rolling(window=4, min_periods=1).max()
    df1['4_min'] = df1['quantity'].rolling(window=4, min_periods=1).min()
    df1['lag'] = df1['quantity'].shift().fillna(0)
    a = np.concatenate((np.array(df1['7_mean']), np.array(df1['28_mean']), np.array(df1['14_mean']), np.array(df1['4_max']),
                        np.array(df1['4_min']), np.array(df1['lag'])))
    a = a.reshape((6, -1))

    # 原始数据
    df2=df.copy()
    df2['quantity'] = df2['quantity'].rolling(window=1, min_periods=1).mean()
    print(df1,df2)
    # df2.to_csv('kkzm.csv')
    con_len=pre_length
    #初始化模型和训练器
    estimator = DeepAREstimator(
        prediction_length=pre_length,
        context_length=con_len,
        freq="D",
        trainer=Trainer(epochs=80),
        use_feat_dynamic_real=True
    )
    # # 初始化训练集和测试集
    train_ds= ListDataset([{"start": df1['eta'].min(), "target": df1['quantity'][:-validation_length],'feat_dynamic_real':a[:,:-validation_length]}], freq="D")
    test_ds1 =ListDataset(
        [{"start": df1['eta'].iloc[-(validation_length + con_len)],
          "target": df2['quantity'][-(validation_length +con_len):-(validation_length - pre_length)],
          'feat_dynamic_real': a[:, -(validation_length + con_len):-(validation_length - pre_length)]}],
        freq="D"
    )
    test_ds2 = ListDataset(
        [{"start": df1['eta'].iloc[-(validation_length )],
          "target": df2['quantity'][-(validation_length-2*pre_length+pre_length+con_len ):-(validation_length -2* pre_length)],
          'feat_dynamic_real': a[:, -(validation_length-2*pre_length+pre_length+con_len  ):-(validation_length - 2*pre_length)]}],
        freq="D"
    )
    result_df = pd.DataFrame(columns=['date','mean','0.75','0.77' ,'0.79','0.81', '0.83','0.85','0.87','0.89','0.91','0.93','0.95'])
    for k in range(validation_length //( 2*pre_length)):
        print(train_ds)
        # 训练模型
        predictor = estimator.train(train_ds)
        # 在验证集上进行预测
        forecast_it1, ts_it1 = make_evaluation_predictions(
            dataset=test_ds1,
            predictor=predictor,
            num_samples=100
        )
        forecasts1 = list(forecast_it1)
        tss1 = list(ts_it1)
        prediction1 = abs(forecasts1[0].mean)
        print(forecasts1[0].mean)
        print(tss1)
        forecast_it2, ts_it2 = make_evaluation_predictions(
            dataset=test_ds2,
            predictor=predictor,
            num_samples=100
        )
        forecasts2 = list(forecast_it2)
        tss2 = list(ts_it2)
        prediction2 = abs(forecasts2[0].mean)
        print(forecasts2[0].mean)
        print(tss2)
        prediction=np.concatenate((prediction1,prediction2))
        print(prediction)
        #计算误差分位点
        num_windows = (len(df1['quantity']) - pre_length) //pre_length # 计算可以预测的窗口数量
        error = []
        for i in range(num_windows):
            start_idx = i * pre_length
            end_idx = 2*pre_length + (i + 1) * pre_length
            # 构造输入数据，使用历史数据作为 context
            input_data = [{'target': df1['quantity'][start_idx:end_idx],'feat_dynamic_real':a[:,start_idx:end_idx], 'start': pd.Timestamp('2023-09-01')}]
            input_ds = ListDataset(input_data, freq='D')
            # 使用模型进行预测
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=input_ds,
                predictor=predictor,
                num_samples=100,
            )
            forecasts = list(forecast_it)
            distribution = [abs(forecast.quantile(0.5)) for forecast in forecasts]
            true_values = df2['quantity'][end_idx-pre_length:end_idx]
            errors = [(d - t) / (t+1) for d, t in zip(distribution[0], true_values)]
            error = error + errors
        error = np.array(error)
        error=error[error<10]
        sort_error = np.sort(error)
        print(sort_error)

        if k!=validation_length // (2*pre_length)-1:
             date_strings=df1['eta'][-(validation_length-k*2*pre_length):-(validation_length-(k+1)*2*pre_length)]
             out = {'date': date_strings,
                    'mean': prediction,
                    '0.75':prediction*(sort_error[int(0.75*len(sort_error))]+1),
                    '0.77': prediction*(sort_error[int(0.77*len(sort_error))]+1),
                    '0.79':prediction*(sort_error[int(0.79*len(sort_error))]+1),
                    '0.81':prediction*(sort_error[int(0.81*len(sort_error))]+1),
                    '0.83': prediction*(sort_error[int(0.83*len(sort_error))]+1),
                    '0.85':prediction*(sort_error[int(0.85*len(sort_error))]+1),
                    '0.87':prediction*(sort_error[int(0.87*len(sort_error))]+1),
                    '0.89':prediction*(sort_error[int(0.89*len(sort_error))]+1),
                    '0.91':prediction*(sort_error[int(0.91*len(sort_error))]+1),
                    '0.93':prediction*(sort_error[int(0.93*len(sort_error))]+1),
                    '0.95':prediction*(sort_error[int(0.95*len(sort_error))]+1),
                    'real':df2['quantity'][-(validation_length-2*pre_length*k):-(validation_length-2*pre_length*(k+1))]}
             temp_df = pd.DataFrame(out)
            # 将临时的 DataFrame 添加到结果 DataFrame 中
             result_df = result_df.append(temp_df, ignore_index=True)
             print(result_df)
        else:
            date_strings = df1['eta'][-(validation_length - k*2 * pre_length):len(df1['eta'])]
            out = {'date': date_strings,
                   'mean': prediction,
                   '0.75': prediction * (sort_error[int(0.75 * len(sort_error))] + 1),
                   '0.77': prediction * (sort_error[int(0.77 * len(sort_error))] + 1),
                   '0.79': prediction * (sort_error[int(0.79 * len(sort_error))] + 1),
                   '0.81': prediction * (sort_error[int(0.81 * len(sort_error))] + 1),
                   '0.83': prediction * (sort_error[int(0.83 * len(sort_error))] + 1),
                   '0.85': prediction * (sort_error[int(0.85 * len(sort_error))] + 1),
                   '0.87': prediction * (sort_error[int(0.87 * len(sort_error))] + 1),
                   '0.89': prediction * (sort_error[int(0.89 * len(sort_error))] + 1),
                   '0.91': prediction * (sort_error[int(0.91 * len(sort_error))] + 1),
                   '0.93': prediction * (sort_error[int(0.93 * len(sort_error))] + 1),
                   '0.95': prediction * (sort_error[int(0.95 * len(sort_error))] + 1),
                   'real': df2['quantity'][-(validation_length - pre_length * 2*k):]}
            temp_df = pd.DataFrame(out)
            # 将临时的 DataFrame 添加到结果 DataFrame 中
            result_df = result_df.append(temp_df, ignore_index=True)
            print(result_df)

        # 更新训练集和测试集
        if k!=validation_length //(2* pre_length)-1:
            if k == validation_length // (2*pre_length) - 2:
                train_ds = ListDataset([{"start": df1['eta'].min(),
                                         "target": df1['quantity'][:-(validation_length - (k + 1) *2* pre_length)],
                                         'feat_dynamic_real':a[:,:-(validation_length - (k + 1) *2* pre_length)]}],
                                       freq="D")
                test_ds1 = ListDataset(
                    [{"start": df1['eta'].iloc[-(pre_length+pre_length+con_len)],
                      "target": df1['quantity'][-(pre_length+pre_length+con_len):-pre_length],
                      'feat_dynamic_real':a[:,-(pre_length+pre_length+con_len):-pre_length]}],
                    freq="D"
                )
                test_ds2 = ListDataset(
                    [{"start": df1['eta'].iloc[-(pre_length+con_len)],
                      "target": df1['quantity'][-(pre_length+con_len):],
                      'feat_dynamic_real': a[:, -(pre_length+con_len):]}],
                    freq="D"
                )

            else:
                train_ds =  ListDataset([{"start": df1['eta'].min(), "target": df1['quantity'][:-(validation_length-(k+1)*2*pre_length)],
                                          'feat_dynamic_real': a[:,:-(validation_length-(k+1)*2*pre_length) ]}], freq="D")
                test_ds1 =ListDataset(
                [{"start": df1['eta'].iloc[-(validation_length-(k+1)*2*pre_length+con_len-pre_length)],
                  "target": df1['quantity'][-(validation_length-(k+1)*2*pre_length+con_len):-(validation_length-(k+1)*2*pre_length-pre_length)],
                  'feat_dynamic_real': a[:,-(validation_length-(k+1)*2*pre_length+con_len):-(validation_length-(k+1)*2*pre_length-pre_length) ]}],

                freq="D"
                )
                test_ds2 = ListDataset(
                    [{                         "start": df1['eta'].iloc[-(validation_length - (k + 1) *2* pre_length-2*pre_length+con_len+pre_length)],
                      "target": df1['quantity'][-(validation_length - (k + 1) *2* pre_length-2*pre_length+con_len+pre_length):-(
                                  validation_length - (k + 1) * 2*pre_length -2* pre_length)],
                      'feat_dynamic_real': a[:, -(validation_length - (k + 1)*2 * pre_length -2*pre_length+con_len+pre_length):-(
                                  validation_length - (k + 1) * 2*pre_length - 2*pre_length)]}],

                    freq="D"
                )

    #根据result_df计算指标
    #计算排除异常值的天指标
    result_df['mpe'] = (abs(result_df['real'] - result_df['mean']) / result_df['real'])
    result_df.loc[result_df['real'] == 0, 'mpe'] = 0
    filtered_mpe = result_df[(result_df['mpe'] <= 1.5) & (result_df['mpe'] > 0)]['mpe']
    result_el = result_df[(result_df['mpe'] > 0)]['mpe']
    filter_ratio = (len(result_el) - len(filtered_mpe)) / len(result_el) * 100
    mape = filtered_mpe.mean()
    # # 输出结果
    print(f"天_剔除高误差预测比例：{filter_ratio:.2f}%")
    print(f"天_平均百分比误差：{mape * 100:.2f}%")

    #导出周表格，计算周指标
    result_df['week_sum_pre'] = result_df['mean'].rolling(window=7).sum().shift(-6)
    # 新建'week_sum_real'列
    result_df['week_sum_real'] = result_df['real'].rolling(window=7).sum().shift(-6)
    # 计算'week_sum_mpe'列
    result_df['week_sum_mpe'] = (
                abs(result_df['week_sum_real'] - result_df['week_sum_pre']) / result_df['week_sum_real'])
    filtered_mpe = result_df[(result_df['week_sum_mpe'] <= 1) & (result_df['week_sum_mpe'] > 0)]['week_sum_mpe']
    filter_ratio = (len(result_df) - len(filtered_mpe) - 6) / (len(result_df) - 6) * 100
    # # 输出结果
    print(f"周_剔除高误差预测比例：{filter_ratio:.2f}%")
    # 计算平均MPE
    average_mpe = filtered_mpe.mean()
    # 输出结果
    print(f"周_平均MPE：{average_mpe * 100:.2f}%")
    name=customer_name+'_'+customer_part_no+'_'+supplier_name+'_'+supplier_part_no+'_'+manufacture_name+'_'+site_db+'.csv'
    result_df.to_csv(name)


# #
# customer_name1='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no1='de2cadf481df52a947256526deeb9522'
# #
# customer_name2='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no2='42accf484119a16be6099bbe59d0f2d5'
#
# customer_name3='3b1111dec2f61521e9476248145606fb'
# customer_part_no3='39c6df04dd3827086bf6ae40bb0bbcbb'
#
# customer_name4='3b1111dec2f61521e9476248145606fb'
# customer_part_no4='0ad597517ce1b0486df4efbf27553cf6'
#
# customer_name5='3b1111dec2f61521e9476248145606fb'
# customer_part_no5='1fd1745c89247d9e3191b1ec30859450'

# customer_name1='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no1='f9743fbed9a869c263409a35b1e8d928'
#
# customer_name2='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no2='74ecba66be4500c370b33e11cb4f2dc9'
# #
# customer_name3='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no3='cc60304dda47bc4b29d0ffe75c68bca8'
#
# customer_name4='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no4='070e431c0ace462beb28fa4bc04da78e'
# #
# customer_name5='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no5='f690b14eda26832d2bfb297407dfb0d9'
# customer_name1='d7c9ab1eeb7f99e7bd0f6c79cce0f4f1'
# customer_part_no1='13c85430d097c2f0986fa9b7a1e662dd'
# customer_name2='d7c9ab1eeb7f99e7bd0f6c79cce0f4f1'
# customer_part_no2='2424eea8a5de79a9e37276dc3a1d8744'
# customer_name3='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no3='29ff8975ab4cec271dbd84233b8b69c8'
# customer_name4='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no4='351ebdd3b5c0a6612fc18ae54807da81'
# customer_name5='d7c9ab1eeb7f99e7bd0f6c79cce0f4f1'
# customer_part_no5='4c2d4ac45e2b4d539294fac1098f9cb4'
# customer_name6='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no6='5ee6171c02ed095af4098ce27bb2afa3'
# customer_name7='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no7='f690b14eda26832d2bfb297407dfb0d9'
# customer_name8='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no8='76785cc41f875a4f5409a5f256be95e3'
# customer_name9='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no9='a29d25d36633cee13877d17c937d16e4'
#
# customer_name10='d7c9ab1eeb7f99e7bd0f6c79cce0f4f1'
# customer_part_no10='a363ed0aaeccb64ee1607fd4b5696256'
# customer_name11='b44d54cbf91dca39e00e2037dcc8a5c6'
# customer_part_no11='eb8cc9f6d7e405e5db27de6cfb9a496f'
# customer_name12='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no12='eba03dfe5505f8150a4abf7adeaea6aa'
# customer_name13='7e0d847c78df7e95d8e8779df7557fbc'
# customer_part_no13='d79b01b31c982eb6d1e6ce87b2f59730'


supplier_name='a'
supplier_part_no='a'
manufacture_name='a'
site_db='a'
filename1=r'E:\data1\yuanshi3\split_v_islm_outbound_2021\v_islm_outbound_2021_1_1000001.csv'#outbound文件
filename2=r'E:\data1\yuanshi3\split_v_islm_outbound_2021\v_islm_outbound_2021_1000001_1048576.csv'#outbound文件
filename3=r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_2000001_2449400.csv'#outbound文件
filename4=r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_1_1000001.csv'#outbound文件
filename5=r'E:\data1\yuanshi3\split_v_islm_outbound_2022\v_islm_outbound_2022_1000001_2000001.csv'#outbound文件
filename6=r'E:\JusLink＆浙大数据20230523\v_islm_outbound_2023.csv'#outbound文件
pre_length=7
window=7
validation_length=7*4
for i in range(1,14):
    print(i)
    customer_name=globals()['customer_name'+str(i)]
    customer_part_no=globals()['customer_part_no'+str(i)]
    print(customer_name,customer_part_no)
    supplier_name =supplier_name
    supplier_part_no =supplier_part_no
    manufacture_name = manufacture_name
    site_db = 'a'
    demand_validation(customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,
                  filename1,filename2,filename3,filename4,filename5,filename6,
                  pre_length,validation_length)