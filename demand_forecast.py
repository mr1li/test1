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
def demand(pre_length,date,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,file_name1,filename2,if_train):
    base_name =customer_name + '_' + customer_part_no+'_' +supplier_name + '_'+supplier_part_no + '_'+manufacture_name + '_'+site_db
    path = filename2+ base_name
    error_name = path + '//' + 'error'
    forecast_name= path+'//'+'forecast.csv'
    con=pre_length
    if if_train==True:
        df = pd.read_csv(file_name1) # sorted_df = df.sort_values(by=["customer_name", "customer_part_no"])
        df['eta'] = pd.to_datetime(df['eta']).dt.strftime('%Y-%m-%d')
        filtered_data = df[(df['customer_name'] == customer_name) & (df['customer_part_no'] == customer_part_no)]
        filtered_data['eta'] = pd.to_datetime(filtered_data['eta'])
        filtered_data = filtered_data.groupby(['eta'], as_index=False)['quantity'].sum()
        filtered_data = filtered_data.sort_values(by='eta', ascending=True)
        df = pd.DataFrame(filtered_data)
        # # 将 'eta' 列设置为日期类型
        df['quantity'] = df['quantity']
        date_range = pd.date_range(df['eta'].min(), df['eta'].max(), freq='D')
        # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
        df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})
        # 平滑数据
        df1 = df.copy()
        df1['7_mean'] = df1['quantity'].rolling(window=7, min_periods=1).mean()
        df1['28_mean'] = df1['quantity'].rolling(window=3, min_periods=1).mean()
        df1['14_mean'] = df1['quantity'].rolling(window=14, min_periods=1).mean()
        df1['4_max'] = df1['quantity'].rolling(window=4, min_periods=1).max()
        df1['4_min'] = df1['quantity'].rolling(window=4, min_periods=1).min()
        df1['lag'] = df1['quantity'].shift().fillna(0)
        a = np.concatenate(
            (np.array(df1['7_mean']), np.array(df1['28_mean']), np.array(df1['14_mean']), np.array(df1['4_max']),
             np.array(df1['4_min']), np.array(df1['lag'])))
        a = a.reshape((6, -1))

        train_dataset = ListDataset([{"start": df['eta'].min(), "target": df['quantity'],'feat_dynamic_real':a}], freq="D")
        estimator = DeepAREstimator(
            prediction_length=pre_length,
            context_length=pre_length,
            freq="D",
            trainer=Trainer(epochs=80),
            use_feat_dynamic_real=True
        )
        predictor = estimator.train(train_dataset)
        num_windows = (len(train_dataset[0]['target']) - pre_length) //pre_length # 计算可以预测的窗口数量
        error = []
        for i in range(num_windows):
            start_idx = i * pre_length
            end_idx = pre_length + (i + 1) * pre_length
            # 构造输入数据，使用历史数据作为 context
            input_data = [{'target': train_dataset[0]['target'][start_idx:end_idx], 'start': pd.Timestamp('2023-09-01')}]

            input_ds = ListDataset(input_data, freq='D')

            # 使用模型进行预测
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=input_ds,
                predictor=predictor,
                num_samples=100,
            )

            forecasts = list(forecast_it)
            tss = list(ts_it)

            # 获取预测分布和真实值
            distribution = [forecast.quantile(0.5) for forecast in forecasts]

            true_values = [ts[-pre_length:] for ts in tss]
            true_values = true_values[0].values.flatten()

            errors = [abs(d - t) / (t+1) for d, t in zip(distribution[0], true_values)]
            error = error + errors
        error = np.array(error)
        sort_error = np.sort(error)

        if not os.path.exists(path):
            # 如果不存在则创建文件夹
            os.makedirs(path)
        predictor.serialize(Path(path))

        np.save(error_name,sort_error)
    if if_train==False:
        error=np.load(error_name+'.npy')
        predictor=Predictor.deserialize(Path(path))
        #构造预测集：
        date=datetime.strptime(date, "%Y-%m-%d")
        df = pd.read_csv(file_name1) # sorted_df = df.sort_values(by=["customer_name", "customer_part_no"])
        df['eta'] = pd.to_datetime(df['eta']).dt.strftime('%Y-%m-%d')
        filtered_data = df[(df['customer_name'] == customer_name) & (df['customer_part_no'] == customer_part_no)]
        filtered_data['eta'] = pd.to_datetime(filtered_data['eta'])
        filtered_data = filtered_data.groupby(['eta'], as_index=False)['quantity'].sum()
        filtered_data = filtered_data.sort_values(by='eta', ascending=True)
        df = pd.DataFrame(filtered_data)
        # # 将 'eta' 列设置为日期类型
        df['quantity'] = df['quantity']
        date_range = pd.date_range(df['eta'].min(), date, freq='D')
        # # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
        df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})
        # 平滑数据
        df1 = df.copy()
        df1['7_mean'] = df1['quantity'].rolling(window=7, min_periods=1).mean()
        df1['28_mean'] = df1['quantity'].rolling(window=3, min_periods=1).mean()
        df1['14_mean'] = df1['quantity'].rolling(window=14, min_periods=1).mean()
        df1['4_max'] = df1['quantity'].rolling(window=4, min_periods=1).max()
        df1['4_min'] = df1['quantity'].rolling(window=4, min_periods=1).min()
        df1['lag'] = df1['quantity'].shift().fillna(0)
        a = np.concatenate(
            (np.array(df1['7_mean']), np.array(df1['28_mean']), np.array(df1['14_mean']), np.array(df1['4_max']),
             np.array(df1['4_min']), np.array(df1['lag'])))
        a = a.reshape((6, -1))
        b = np.zeros((6, a.shape[1]+pre_length))
        b[:, : a.shape[1]] = a

        test_dataset = ListDataset([{"start": date- timedelta(days=con), "target":np.concatenate((df['quantity'][-pre_length:],np.zeros(pre_length))),
                                     'feat_dynamic_real':b}], freq="D")
        print(test_dataset)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,  # 使用训练数据集的信息进行预测
            predictor=predictor,
            num_samples=100  # 可以根据需要调整采样次数
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        prediction=forecasts[0].mean
        date_list = [date + timedelta(days=i) for i in range(pre_length)]
        date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
        out = {'date': date_strings,
               'mean': prediction,
               '0.75': prediction * (error[int(0.75 * len(error))] + 1),
               '0.77': prediction * (error[int(0.77 * len(error))] + 1),
               '0.79': prediction * (error[int(0.79 * len(error))] + 1),
               '0.81': prediction * (error[int(0.81 * len(error))] + 1),
               '0.83': prediction * (error[int(0.83 * len(error))] + 1),
               '0.85': prediction * (error[int(0.85 * len(error))] + 1),
               '0.87': prediction * (error[int(0.87 * len(error))] + 1),
               '0.89': prediction * (error[int(0.89 * len(error))] + 1),
               '0.91': prediction * (error[int(0.91 * len(error))] + 1),
               '0.93': prediction * (error[int(0.93 * len(error))] + 1),
               '0.95': prediction * (error[int(0.95 * len(error))] + 1)
               }
        temp_df = pd.DataFrame(out)
        out_name=forecast_name
        print(out_name)
        out.to_csv(out_name,index=False,line_terminator='\n')

if __name__ == "__main__":
    pre_length=7
    date='2023-09-01'
    customer_name=
    customer_part_no=
    supplier_name=
    supplier_part_no=
    manufacture_name=
    site_db=
    file_name1=r'F:\cpan\桌面\data2\out.csv'#outbound文件
    file_name2='E://mxnet//tmp//'

    demand(pre_length,date,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,file_name1,file_name2,True)
    demand(pre_length,date, customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db, file_name1,file_name2, False)
