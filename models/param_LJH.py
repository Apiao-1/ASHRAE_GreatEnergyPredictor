# coding: utf-8

import sys

# sys.path.append('/home/aistudio/external-libraries')
# sys.path.append('/cos_person/notebook/100009019970/external-libraries')
import os

# os.system('pip install lightgbm --user')
# os.system('pip install meteocalc --user')
# os.system('pip install seaborn --user')

os.system('pip install lightgbm')
os.system('pip install meteocalc')
# os.system('pip install seaborn')
os.system('pip install bayesian-optimization')
os.system('pip install category_encoders')

# 查看内存和cpu
os.system('free -g')
os.system('cat /proc/cpuinfo| grep "processor"| wc -l')


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)

import gc
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime
from meteocalc import feels_like, Temp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=False, verbose=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    if verbose: print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude
def fill_weather_dataset(weather_df):
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(), time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df, new_rows])

        weather_df = weather_df.reset_index(drop=True)

        # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month

    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(),
                                          columns=["air_temperature"])
    weather_df.update(air_temperature_filler, overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler, overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['dew_temperature'].mean(),
                                          columns=["dew_temperature"])
    weather_df.update(due_temperature_filler, overwrite=False)

    # # Step 1
    # sea_level_filler = weather_df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()
    # # Step 2
    # sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])
    #
    # weather_df.update(sea_level_filler, overwrite=False)
    #
    # wind_direction_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_direction'].mean(),
    #                                      columns=['wind_direction'])
    # weather_df.update(wind_direction_filler, overwrite=False)
    #
    # wind_speed_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(),
    #                                  columns=['wind_speed'])
    # weather_df.update(wind_speed_filler, overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler, overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    def get_meteorological_features(data):
        def calculate_rh(df):
            df['relative_humidity'] = 100 * (
                    np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp(
                (17.625 * df['air_temperature']) / (243.04 + df['air_temperature'])))

        def calculate_fl(df):
            flike_final = []
            flike = []
            # calculate Feels Like temperature
            for i in range(len(df)):
                at = df['air_temperature'][i]
                rh = df['relative_humidity'][i]
                ws = df['wind_speed'][i]
                flike.append(feels_like(Temp(at, unit='C'), rh, ws))
            for i in range(len(flike)):
                flike_final.append(flike[i].f)
            df['feels_like'] = flike_final
            del flike_final, flike, at, rh, ws

        calculate_rh(data)
        calculate_fl(data)
        return data

    weather_df = get_meteorological_features(weather_df)

    weather_df['air_temperature_m3'] = weather_df['air_temperature'].shift(-3)
    weather_df['air_temperature_m2'] = weather_df['air_temperature'].shift(-2)
    weather_df['air_temperature_m1'] = weather_df['air_temperature'].shift(-1)
    return weather_df



def features_engineering(df):
    # Sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)

    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # df["dayofweek"] = df["timestamp"].dt.weekday
    # holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
    #             "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
    #             "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
    #             "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
    #             "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
    #             "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
    #             "2019-01-01"]
    # df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)

    df['month'] = df['timestamp'].dt.month
    df['month'].replace((12, 1, 2), 1, inplace=True)
    df['month'].replace((3, 4, 5), 2, inplace=True)
    df['month'].replace((6, 7, 8), 3, inplace=True)
    df['month'].replace((9, 10, 11), 4, inplace=True)

    # Remove Unused Columns
    drop = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]
    df = df.drop(drop, axis=1)
    gc.collect()

    return df


def find_best_param(train, param):
    kf = StratifiedKFold(n_splits=4, shuffle=False, random_state=2319)
    models = []
    oof = np.zeros(len(train))

    for tr_idx, val_idx in kf.split(train_df, train_df['month']):
        tr_x, tr_y = train[features].iloc[tr_idx], train[target].iloc[tr_idx]
        vl_x, vl_y = train[features].iloc[val_idx], train[target].iloc[val_idx]
        tr_data = lgb.Dataset(tr_x, label=tr_y, categorical_feature=categorical)
        vl_data = lgb.Dataset(vl_x, label=vl_y, categorical_feature=categorical)
        clf = lgb.train(param, tr_data, 20000, valid_sets=[tr_data, vl_data], verbose_eval=False,
                        early_stopping_rounds=50)
        models.append(clf)
        oof[val_idx] = clf.predict(vl_x)
        gc.collect()
    score = np.sqrt(mean_squared_error(train[target], np.clip(oof, a_min=0, a_max=None)))
    print('Our oof cv is :', score)

    global best_score, best_param
    if score <= best_score:
        best_score = score
        print("update best_score: %f" % best_score)
        best_param = param
        print("update best params: %s" % best_param)
    return score


def GBM_evaluate(min_data_in_leaf, min_child_weight, feature_fraction, num_leaves, bagging_fraction, lambda_l2,
                 bagging_freq):
    """自定义的模型评估函数"""

    global flag
    if flag:
        params = {
            'bagging_fraction': 0.6958018426239316, 'feature_fraction': 0.8201093412534093, 'verbosity': -1,
            'bagging_freq': 9, 'num_leaves': 1680, 'metric': 'rmse', 'boosting': 'gbdt', 'objective': 'regression',
            'seed': 4534, 'lambda_l2': 0.1814272579281604, 'min_data_in_leaf': 149, 'min_child_weight': 48,
            'learning_rate': 0.1,
        }
        find_best_param(X, params)
        flag = False

    # 模型固定的超参数
    param = {
        'objective': 'regression',
        'learning_rate': 0.1,
        "boosting": "gbdt",
        "metric": 'rmse',
        # 'num_threads': config['workers'],
        'seed': 4534,
        "verbosity": -1,
    }

    # 贝叶斯优化器生成的超参数
    param['min_child_weight'] = int(min_child_weight)
    param['feature_fraction'] = float(feature_fraction)
    # param['max_depth'] = int(max_depth)
    param['num_leaves'] = int(num_leaves)
    param['bagging_fraction'] = float(bagging_fraction)
    param['bagging_freq'] = int(bagging_freq)
    param['lambda_l2'] = float(lambda_l2)
    # param['lambda_l1'] = float(lambda_l1)
    param['min_data_in_leaf'] = int(min_data_in_leaf)

    # 5-flod 交叉检验，注意BayesianOptimization会向最大评估值的方向优化，因此对于回归任务需要取负数。
    # 这里的评估函数为neg_mean_squared_error，即负的MSE。
    val = -find_best_param(X, param)

    return val


def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 30
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    print(params['max_params'])

    return params

def data_building(file_dir=None):
    building = pd.read_csv(file_dir)
    building = reduce_mem_usage(building, use_float16=True)

    le = LabelEncoder()
    building["primary_use"] = le.fit_transform(building["primary_use"])

    sq_m = building.loc[(building['site_id'].isin([1])) & (~building['building_id'].isin([150, 106])), 'square_feet']
    building.loc[
        (building['site_id'].isin([1])) & (~building['building_id'].isin([150, 106])), 'square_feet'] = sq_m * 10.7639
    building["square_feet_floor"] = building['square_feet'] / building['floor_count']
    building['square_feet_floor'] = building['square_feet_floor'].replace(np.inf, building['square_feet'])
    building['square_feet'] = np.log1p(building['square_feet'])
    building['square_feet_floor'] = np.log1p(building['square_feet_floor'])

    return building


best_score = 9999
best_param = {}
X, y, group = None, None, None
DATA_PATH = "/cos_person/notebook/100009019970/data/"
RESULT_PATH = "/cos_person/notebook/100009019970/results_bayis/"
flag = True
m = 0

if __name__ == '__main__':
    train_df = pd.read_csv(DATA_PATH + 'train.csv')
    weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')

    # eliminate bad rows
    bad_rows = pd.read_csv(DATA_PATH + 'rows_to_drop.csv')
    train_df.drop(bad_rows.loc[:, '0'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    # weather manipulation
    weather_df = fill_weather_dataset(weather_df)
    train_df = reduce_mem_usage(train_df, use_float16=True)

    weather_df = reduce_mem_usage(weather_df, use_float16=True)
    building_df = data_building(DATA_PATH + 'building_metadata.csv')

    # merge data
    train_df = train_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
    train_df = train_df.merge(weather_df, how='left', left_on=['site_id', 'timestamp'],
                              right_on=['site_id', 'timestamp'])
    del weather_df
    gc.collect()

    # feature engineering
    train_df = features_engineering(train_df)

    # transform target variable
    train_df['meter_reading'] = np.log1p(train_df["meter_reading"])

    # drop = ["sea_level_pressure", "wind_direction", "wind_speed"]
    # train_df = train_df.drop(drop, axis=1)
    # gc.collect()

    # declare target, categorical and numeric columns
    target = 'meter_reading'
    categorical = ['building_id', 'site_id', 'primary_use', 'meter', 'dayofweek']
    numeric_cols = [col for col in train_df.columns if col not in categorical + [target, 'timestamp', 'month']]
    features = categorical + numeric_cols

    global X, y, group
    y = train_df['meter_reading']
    X = train_df

    # 调参范围
    adj_params = {
        'min_child_weight': (3, 50),
        'feature_fraction': (0.3, 1),
        # 'max_depth': (4, 15),
        'num_leaves': (30, 2000),
        'bagging_fraction': (0.3, 1),
        'bagging_freq': (1, 10),
        'lambda_l2': (0.1, 2),
        # 'lambda_l1': (0.1, 1),
        'min_data_in_leaf': (1, 150)
    }

    # 调用贝叶斯优化
    BayesianSearch(GBM_evaluate, adj_params)

    print("final best param: %s" % best_param)
    print("final best score: %f" % best_score)
