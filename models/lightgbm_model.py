# coding: utf-8

import sys
sys.path.append('/home/aistudio/external-libraries')

import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import datetime
from meteocalc import feels_like, Temp

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
from matplotlib import pyplot as plt

from datetime import datetime, date, timedelta

from collections import defaultdict
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

# ope = os.path.exists
# osp = os.path
# opj = os.path.join
#
# config = yaml.safe_load(
#     """
#     work_dir: "C:/Users/cakey/ashrae-energy-prediction"
#     data_dir: "C:/Users/cakey/ashrae-energy-prediction/input"
#
#     seed : 4534
#     workers: 5
#
#     train:
#         folds: 3
#         model: lightgbm
#         learning_rate: 0.05
#         num_rounds: 1000000
#         early_stopping: 50
#
#     test:
#         batch_size: 1000000
#     """
# )

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def reduce_mem_usage(df, use_float16=False, verbose=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: logging.info("Memory usage of dataframe is {:.2f} MB".format(start_mem))

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
    if verbose: logging.info("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    if verbose: logging.info("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

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

    return df


def create_train(meter=0):
    train = pd.read_csv(DATA_PATH + "train.csv")
    bad_rows = pd.read_csv(DATA_PATH + "rows_to_drop.csv")
    train = train.drop(bad_rows.loc[:, '0']).reset_index(drop=True)
    train = reduce_mem_usage(train, use_float16=True)

    train = train.loc[train['meter'] == meter].reset_index(drop=True)

    building = data_building(file_dir=DATA_PATH + "building.csv")
    train = train.merge(building, left_on='building_id', right_on='building_id', how='left')

    weather = data_weather(file_dir=DATA_PATH + "weather_train.csv")
    train = train.merge(weather, how='left', on=['site_id', 'timestamp'])

    train = data(train)
    train['meter_reading'] = np.log1p(train["meter_reading"])

    train = train.drop(["timestamp"], axis=1)
    logging.info(train.shape)
    logging.info(train.head())
    return train


def create_test(meter=0):
    test = pd.read_csv(DATA_PATH + "test.csv")
    test = reduce_mem_usage(test, use_float16=True)
    building = data_building(file_dir=DATA_PATH + "building.csv")
    test = test.merge(building, left_on='building_id', right_on='building_id', how='left')
    weather = data_weather(file_dir=DATA_PATH + "weather_test.csv")
    test = test.merge(weather, how='left', on=['site_id', 'timestamp'])
    test = data(test)
    test = test.drop(["timestamp"], axis=1)

    test = test.loc[test['meter'] == meter].reset_index(drop=True)

    return test

import logging
from logging.handlers import TimedRotatingFileHandler
import re
def init_log():
    logging.getLogger('bloomfilter').setLevel('WARN')
    log_file_handler = TimedRotatingFileHandler(filename="bloomfilter.log", when="D", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s- %(filename)s:%(lineno)s - %(threadName)s - %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler.setFormatter(formatter)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(log_file_handler)


if __name__ == '__main__':
    init_log()

    DATA_PATH = "/home/aistudio/data/data17604/"
    # DATA_PATH = "../data/"

    for m in range(4):
        params = {}
        if m == 0:
            params = {'num_leaves': 526, 'max_depth': 12, 'bagging_fraction': 0.10081820060774621,
                      'feature_fraction': 0.7742020727078566}
        if m == 1:
            params = {'num_leaves': 261, 'max_depth': 10, 'bagging_fraction': 0.26016952708832514,
                      'feature_fraction': 0.5491805155524557}
        if m == 2:
            params = {'num_leaves': 305, 'max_depth': 12, 'bagging_fraction': 0.9788344136664392,
                      'feature_fraction': 0.7869347797200678}
        if m == 3:
            params = {'num_leaves': 917, 'max_depth': 11, 'bagging_fraction': 0.15036565575986183,
                      'feature_fraction': 0.3663760577105429}

        params.update({
            'objective': 'regression',
            'learning_rate': 0.05,
            "boosting": "gbdt",
            "metric": 'rmse',
            # 'num_threads': config['workers'],
            'seed': 4534,
            "verbosity": -1,
        })
        logging.info(params)

        train = create_train(meter=m)
        test = create_test(meter=m)

        oof_train = np.zeros((train.shape[0]))

        models = []
        kf = GroupKFold(n_splits=3)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train, groups=train['group'])):
            d_train = lgb.Dataset(
                train.iloc[train_idx].drop(columns=['meter', 'meter_reading', 'group', "month"]),
                label=train['meter_reading'].iloc[train_idx],
                categorical_feature=None)

            d_valid = lgb.Dataset(
                train.iloc[valid_idx].drop(columns=['meter', 'meter_reading', 'group', "month"]),
                label=train['meter_reading'].iloc[valid_idx],
                categorical_feature=None)

            watchlist = [d_train, d_valid]

            categorical_feats = [train.drop(columns=['meter', 'meter_reading', 'group', "month"]).columns.get_loc(c) for
                                 c
                                 in ['building_id', 'site_id', 'primary_use', 'day_of_week', 'is_holiday']]

            mdl = lgb.train(
                params,
                train_set=d_train,
                categorical_feature=categorical_feats,
                valid_sets=watchlist,
                verbose_eval=False,
                num_boost_round=1000000,
                early_stopping_rounds=50,
            )

            # predict on out-of-fold samples...
            y_valid_pred = mdl.predict(train.iloc[valid_idx].drop(columns=['meter', 'meter_reading', 'group', "month"]),
                                       num_iteration=mdl.best_iteration)
            oof_train[valid_idx] += y_valid_pred

            score = np.sqrt(mean_squared_error(train['meter_reading'].iloc[valid_idx], y_valid_pred))
            logging.info('Fold: %s \t Validation: %s' % (fold, score))

            models.append(mdl)

            del mdl
            gc.collect()

        validation = pd.DataFrame({
            'meter_reading': train['meter_reading'],
            "meter_reading_oof": oof_train
        })

        save_file = "../results/validation_model=%s_meter=%s.csv" % ("lightgbm", m)
        validation.to_csv(save_file, index=False)

        score = np.sqrt(mean_squared_error(train['meter_reading'], oof_train))
        logging.info('Validation: %s' % (score))

        del oof_train
        gc.collect()


        def predictions(meter, test, models):

            # batch_size = int(100)
            batch_size = int(1000000)
            iterations = (test.shape[0] + batch_size - 1) // batch_size

            meter_reading = []

            for i in range(iterations):
                pos = i * batch_size
                fold_preds = [np.expm1(
                    model.predict(test.drop(columns=['row_id', 'meter', 'group', "month"]).iloc[pos: pos + batch_size],
                                  num_iteration=model.best_iteration)) for model in models]
                meter_reading.extend(np.mean(fold_preds, axis=0))

            submission = pd.DataFrame({
                'row_id': test.row_id,
                "meter_reading": np.clip(meter_reading, a_min=0, a_max=None)
            })

            save_file = "../results/submission_model=%s_meter=%s.csv" % ("lightgbm", m)

            submission.to_csv(save_file, index=False)


        predictions(meter=m, test=test, models=models)

    prediction_files = [
        ('../results/validation_model=lightgbm_meter=0.csv'),
        ('../results/validation_model=lightgbm_meter=1.csv'),
        ('../results/validation_model=lightgbm_meter=2.csv'),
        ('../results/validation_model=lightgbm_meter=3.csv'),
    ]

    predictions = pd.DataFrame()

    for file in prediction_files:
        predictions = predictions.append(pd.read_csv(file))

    logging.info(np.sqrt(mean_squared_error(predictions['meter_reading'], predictions['meter_reading_oof'])))

    save_file = "../results/validation.csv"
    predictions.to_csv(save_file, index=False)

    plt.figure()
    sns.distplot(predictions['meter_reading']).set_title("Train-Test Distribution")
    sns.distplot(predictions['meter_reading_oof'])

    prediction_files = [
        ('../results/submission_model=lightgbm_meter=0.csv'),
        ('../results/submission_model=lightgbm_meter=1.csv'),
        ('../results/submission_model=lightgbm_meter=2.csv'),
        ('../results/submission_model=lightgbm_meter=3.csv'),
    ]

    predictions = pd.DataFrame()

    for file in prediction_files:
        predictions = predictions.append(pd.read_csv(file))

    predictions = predictions.sort_values(by=['row_id'])

    save_file = '../results/submission.csv'
    predictions.to_csv(save_file, index=False)

    logging.info(predictions.head())