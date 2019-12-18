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
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, date, timedelta
import category_encoders as ce
from collections import defaultdict
from collections import Counter
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


def data_building(file_dir=None):
    building = pd.read_csv(file_dir)
    building = reduce_mem_usage(building, use_float16=True)

    le = LabelEncoder()
    building["primary_use"] = le.fit_transform(building["primary_use"])

    sq_m = building.loc[(building['site_id'].isin([1])) & (~building['building_id'].isin([150, 106])), 'square_feet']
    building.loc[
        (building['site_id'].isin([1])) & (~building['building_id'].isin([150, 106])), 'square_feet'] = sq_m * 10.7639
    # building['floor_count'] = building['floor_count']
    # building['year_built'] = building['year_built']
    building["square_feet_floor"] = building['square_feet'] / building['floor_count']
    building['square_feet_floor'] = building['square_feet_floor'].replace(np.inf, building['square_feet'])
    building['square_feet'] = np.log1p(building['square_feet'])
    building['square_feet_floor'] = np.log1p(building['square_feet_floor'])

    return building


def data_weather(file_dir=None):
    weather = pd.read_csv(file_dir)
    weather = impute_weather(weather)
    weather = reduce_mem_usage(weather, use_float16=True)

    # emwa...
    weather['air_temperature_m3'] = weather['air_temperature'].shift(-3)
    weather['air_temperature_m2'] = weather['air_temperature'].shift(-2)
    weather['air_temperature_m1'] = weather['air_temperature'].shift(-1)

    return weather


# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude
def impute_weather(weather_df):
    start_date = datetime.strptime(weather_df['timestamp'].min(), "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(weather_df['timestamp'].max(), "%Y-%m-%d %H:%M:%S")
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - timedelta(hours=x)).strftime("%Y-%m-%d %H:%M:%S") for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df, new_rows], sort=True)
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

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler, overwrite=False)

    wind_direction_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_direction'].mean(),
                                         columns=['wind_direction'])
    weather_df.update(wind_direction_filler, overwrite=False)

    wind_speed_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(),
                                     columns=['wind_speed'])
    weather_df.update(wind_speed_filler, overwrite=False)

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
    return weather_df


def data(df):
    df.reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(arg=df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['year_day'] = df['timestamp'].dt.dayofyear
    df.loc[df['day_of_week'].isin([1, 2, 3]), 'day_of_week'] = 1

    holidays = [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
        "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
        "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
        "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
        "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
        "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
        "2019-01-01"]
    df["is_holiday"] = (df.timestamp.dt.date.isin(holidays)).astype(int)

    df['group'] = df['timestamp'].dt.month
    df['group'].replace((1, 2, 3, 4), 1, inplace=True)
    df['group'].replace((5, 6, 7, 8), 2, inplace=True)
    df['group'].replace((9, 10, 11, 12), 3, inplace=True)

    # df = df.drop(["sea_level_pressure",  "wind_direction", "wind_speed"], axis=1)
    return df


def create_train(meter=0):
    global DATA_PATH
    train = pd.read_csv(DATA_PATH + "train.csv")
    bad_rows = pd.read_csv(DATA_PATH + "rows_to_drop.csv")
    train = train.drop(bad_rows.loc[:, '0']).reset_index(drop=True)
    train = reduce_mem_usage(train, use_float16=True)
    del bad_rows
    gc.collect()

    train = train.loc[train['meter'] == meter].reset_index(drop=True)

    building = data_building(file_dir=DATA_PATH + "building.csv")
    train = train.merge(building, left_on='building_id', right_on='building_id', how='left')

    weather = data_weather(file_dir=DATA_PATH + "weather_train.csv")
    train = train.merge(weather, how='left', on=['site_id', 'timestamp'])

    train = data(train)
    train['meter_reading'] = np.log1p(train["meter_reading"])

    train = train.drop(["timestamp"], axis=1)
    print(train.shape)
    print(train.head())
    return train


def find_best_param(m, train, y_true, params):
    models = []
    oof_train = np.zeros((train.shape[0]))
    kf = GroupKFold(n_splits=3)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, groups=group)):
        # print("len(train_idx) at fold %d: %d" % (fold, len(train_idx)))
        # print("len(valid_idx) at fold %d: %d" % (fold, len(valid_idx)))

        d_train = lgb.Dataset(train.iloc[train_idx], label=y_true.iloc[train_idx], categorical_feature=None)
        d_valid = lgb.Dataset(train.iloc[valid_idx], label=y_true.iloc[valid_idx], categorical_feature=None)
        watchlist = [d_train, d_valid]

        categorical_feats = [train.columns.get_loc(c) for c
                             in ['building_id', 'site_id', 'primary_use', 'day_of_week', 'is_holiday']]

        mdl = lgb.train(
            params,
            train_set=d_train,
            categorical_feature=categorical_feats,
            valid_sets=watchlist,
            verbose_eval=False,
            num_boost_round=100000,
            early_stopping_rounds=50,
        )

        # predict on out-of-fold samples...
        y_valid_pred = mdl.predict(train.iloc[valid_idx], num_iteration=mdl.best_iteration)
        oof_train[valid_idx] += y_valid_pred

        score = np.sqrt(mean_squared_error(y_true.iloc[valid_idx], y_valid_pred))
        print('Fold: %s \t Validation: %s' % (fold, score))

        models.append(mdl)

        del mdl
        gc.collect()

    validation = pd.DataFrame({
        'meter_reading': y_true,
        "meter_reading_oof": oof_train
    })
    # global RESULT_PATH
    # save_file = RESULT_PATH + "validation_model=%s_meter=%s.csv" % ("lightgbm", m)
    # validation.to_csv(save_file, index=False)
    print("write file finished, location: validation_model=%s_meter=%s.csv" % ("lightgbm", m))

    score = np.sqrt(mean_squared_error(y_true, oof_train))
    print('total Validation: %s' % (score))

    del oof_train
    gc.collect()

    global best_score, best_param
    if score <= best_score:
        best_score = score
        print("update best_score: %f" % best_score)
        best_param = params
        print("update best params: %s" % best_param)
    return score


def GBM_evaluate(min_data_in_leaf, min_child_weight, feature_fraction, num_leaves, bagging_fraction, lambda_l2,
                 bagging_freq):
    """自定义的模型评估函数"""

    global flag
    if flag:
        params = {
            # 'min_data_in_leaf': 1, 'bagging_freq': 1, 'min_child_weight': 50,
            #           'lambda_l2': 0.35679677039439106, 'bagging_fraction': 0.9999999999911137, 'num_leaves': 372,
            #           'feature_fraction': 0.3060617622234653,
            'bagging_freq': 9, 'lambda_l2': 0.20866961853961202, 'min_child_weight': 47,
                      'feature_fraction': 0.4816474356267998, 'bagging_fraction': 0.8211493228794307,
                      'num_leaves': 1295, 'min_data_in_leaf': 147,

            # 'min_child_weight': 50, 'feature_fraction': 0.30000000000656196, 'bagging_fraction': 0.8846794189248148,
            #          'lambda_l2': 0.15164557857011038, 'bagging_freq': 1, 'min_data_in_leaf': 1,
            #          'num_leaves': 570,

            # 'lambda_l2': 1.9999999998017384, 'num_leaves': 1300, 'min_child_weight': 50,
            #           'feature_fraction': 0.3, 'min_data_in_leaf': 1, 'bagging_fraction': 0.3, 'bagging_freq': 1,

            'objective': 'regression',
            'learning_rate': 0.1,
            "boosting": "gbdt",
            "metric": 'rmse',
            # 'num_threads': config['workers'],
            'seed': 4534,
            "verbosity": -1,
        }
        find_best_param(m, X, y, params)
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
    val = -find_best_param(m, X, y, param)

    return val


def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 40
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    print(params['max_params'])

    return params


best_score = 9999
best_param = {}
X, y, group = None, None, None
DATA_PATH = "/cos_person/notebook/100009019970/data/"
RESULT_PATH = "/cos_person/notebook/100009019970/results_bayis/"
flag = True
m = 0

if __name__ == '__main__':
    start = datetime.now()
    print("start at:", start.strftime('%Y-%m-%d %H:%M:%S'))

    # DATA_PATH = "/home/aistudio/data/data17604/"
    # DATA_PATH = "/cos_person/notebook/100009019970/data/"
    # RESULT_PATH = "/cos_person/notebook/100009019970/results/"

    train = create_train(meter=m)
    target_encoder = ce.TargetEncoder(cols=["building_id"]).fit(train, train['meter_reading'])
    train = target_encoder.transform(train)
    print(train.shape)
    print(train.head())

    global X, y, group
    y = train['meter_reading']
    group = train['group']
    train.drop(['meter', 'meter_reading', 'group', "month"], axis=1, inplace=True)
    X = train

    # 调参范围
    adj_params = {
        'min_child_weight': (3, 50),
        'feature_fraction': (0.3, 1),
        # 'max_depth': (4, 15),
        'num_leaves': (30, 1300),
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

    end = datetime.now()
    print("end at:", end.strftime('%Y-%m-%d %H:%M:%S'))
