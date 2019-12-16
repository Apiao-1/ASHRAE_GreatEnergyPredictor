import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc
import warnings
from sklearn.metrics import mean_squared_error
import os


warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

# Original code # https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks

# def create_lag_features(df, window):
#     """
#     Creating lag-based features looking back in time.
#     """
#
#     feature_cols = ["cloud_coverage"]
#     # feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"]
#     df_site = df.groupby("site_id")
#
#     df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)
#
#     df_mean = df_rolled.mean().reset_index().astype(np.float16)
#     df_median = df_rolled.median().reset_index().astype(np.float16)
#     df_min = df_rolled.min().reset_index().astype(np.float16)
#     df_max = df_rolled.max().reset_index().astype(np.float16)
#     # df_std = df_rolled.std().reset_index().astype(np.float16)
#     # df_skew = df_rolled.skew().reset_index().astype(np.float16)
#
#     for feature in feature_cols:
#         df[f"{feature}_mean_lag{window}"] = df_mean[feature]
#         df[f"{feature}_median_lag{window}"] = df_median[feature]
#         df[f"{feature}_min_lag{window}"] = df_min[feature]
#         df[f"{feature}_max_lag{window}"] = df_max[feature]
#         # df[f"{feature}_std_lag{window}"] = df_std[feature]
#         # df[f"{feature}_skew_lag{window}"] = df_skew[feature]
#
#     # df.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1,inplace=True)
#
#     return df

def fill_weather_dataset(weather_df):
    # 插值
    # weather_df = weather_df.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))

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

    weather_df['humidity'] = relative_humidity(weather_df.air_temperature, weather_df.dew_temperature).astype(np.float16)

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
    precip_depth_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler, overwrite=False)

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

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    # weather_df = get_meteorological_features(weather_df)

    # weather_df = create_lag_features(weather_df, 18)


    return weather_df

# !pip install meteocalc
# from meteocalc import feels_like, Temp
# def get_meteorological_features(data):
#     def calculate_rh(df):
#         df['relative_humidity'] = 100 * (
#                 np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp(
#             (17.625 * df['air_temperature']) / (243.04 + df['air_temperature'])))
#
#     def calculate_fl(df):
#         flike_final = []
#         flike = []
#         # calculate Feels Like temperature
#         for i in range(len(df)):
#             at = df['air_temperature'][i]
#             rh = df['relative_humidity'][i]
#             ws = df['wind_speed'][i]
#             flike.append(feels_like(Temp(at, unit='C'), rh, ws))
#         for i in range(len(flike)):
#             flike_final.append(flike[i].f)
#         df['feels_like'] = flike_final
#         del flike_final, flike, at, rh, ws
#
#     calculate_rh(data)
#     calculate_fl(data)
#     return data

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

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
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def features_engineering(df):
    # Sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)

    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    # holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
    #             "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
    #             "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
    #             "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
    #             "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
    #             "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
    #             "2019-01-01"]
    # df['group'] = df['timestamp'].dt.month
    # df['group'].replace((1, 2, 3, 4), 1, inplace=True)
    # df['group'].replace((5, 6, 7, 8), 2, inplace=True)
    # df['group'].replace((9, 10, 11, 12), 3, inplace=True)
    # df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)
    # df["is_weekend"] = df["weekend"].apply(lambda x: x // 5).astype(int)

    df['square_feet'] = np.log1p(df['square_feet'])


    # Remove Unused Columns
    drop = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed", "year_built", "floor_count"]
    df = df.drop(drop, axis=1)
    gc.collect()

    # Encode Categorical Data
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])

    return df

def relative_humidity(Tc,Tdc):
    E = 6.11*10.0**(7.5*Tdc/(237.7+Tdc))
    Es = 6.11*10.0**(7.5*Tc/(237.7+Tc))
    RH = (E/Es)*100
    return RH


if __name__ == '__main__':
    # DATA_PATH = "../input/ashrae-energy-prediction/"
    DATA_PATH = "../data/"

    train_df = pd.read_csv(DATA_PATH + 'train.csv')

    # Remove outliers
    train_df = train_df[train_df['building_id'] != 1099]
    train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')
    weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')

    weather_df = fill_weather_dataset(weather_df)

    train_df = reduce_mem_usage(train_df, use_float16=True)
    building_df = reduce_mem_usage(building_df, use_float16=True)
    weather_df = reduce_mem_usage(weather_df, use_float16=True)

    train_df = train_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
    train_df = train_df.merge(weather_df, how='left', left_on=['site_id', 'timestamp'],
                              right_on=['site_id', 'timestamp'])
    del weather_df
    gc.collect()

    train_df = features_engineering(train_df)
    # df_building_meter = train_df.groupby(["building_id", "meter"]).agg(
    #     mean_building_meter=("meter_reading", "mean"),
    #     median_building_meter=("meter_reading", "median")).reset_index()
    # df_building_meter_hour = train_df.groupby(["building_id", "meter", "hour"]).agg(
    #     mean_building_meter=("meter_reading", "mean"),
    #     median_building_meter=("meter_reading", "median")).reset_index()
    # train_df = train_df.merge(df_building_meter, on=["building_id", "meter"])
    # train_df = train_df.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])

    print(train_df.head(10))
    print(train_df.shape)

    target = np.log1p(train_df["meter_reading"])
    features = train_df.drop('meter_reading', axis=1)
    del train_df
    gc.collect()

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
    # categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend","group","is_holiday", "is_weekend"]
    # params = {
    #     "objective": "regression",
    #     "boosting": "gbdt",
    #     "num_leaves": 1280,
    #     "learning_rate": 0.05,
    #     "feature_fraction": 0.85,
    #     "reg_lambda": 2,
    #     "metric": "rmse",
    # }
    params = {
        'num_leaves': 800,
        'objective': 'regression',
        'learning_rate': 0.05,
        'boosting': 'gbdt',
        'subsample': 0.4,
        'feature_fraction': 0.7,
        'n_jobs': -1,
        'seed': 50,
        'metric': 'rmse'
    }

    kf = KFold(n_splits=3)
    models = []
    RMSEs = []
    for train_index, test_index in kf.split(features):
        tr_x, tr_y= features.loc[train_index], target.loc[train_index]
        vl_x, vl_y = features.loc[test_index], target.loc[test_index]

        d_training = lgb.Dataset(tr_x, label=tr_y, categorical_feature=categorical_features)
        d_test = lgb.Dataset(vl_x, label=vl_y, categorical_feature=categorical_features)

        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training, d_test],
                          verbose_eval=False, early_stopping_rounds=50)
        models.append(model)

        y_pred = model.predict(vl_x, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(vl_y, y_pred))
        print("single rmse:", rmse)
        RMSEs.append(rmse)

        del tr_x, tr_y, vl_x, vl_y, d_training, d_test
        gc.collect()

    print("3 floder mean RMSE：", np.mean(RMSEs))
    del features, target
    gc.collect()

    # Important Features
    for model in models:
        plt.figure(figsize=(12, 12))
        lgb.plot_importance(model, importance_type="gain")
        plt.show()

    # Load Test Data
    test_df = pd.read_csv(DATA_PATH + 'test.csv')
    row_ids = test_df["row_id"]
    test_df.drop("row_id", axis=1, inplace=True)
    test_df = reduce_mem_usage(test_df)

    test_df = test_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
    del building_df
    gc.collect()

    weather_df = pd.read_csv(DATA_PATH + 'weather_test.csv')
    weather_df = fill_weather_dataset(weather_df)
    weather_df = reduce_mem_usage(weather_df)

    test_df = test_df.merge(weather_df, how='left', on=['timestamp', 'site_id'])
    del weather_df
    gc.collect()

    test_df = features_engineering(test_df)
    # test_df = test_df.merge(df_building_meter, on=["building_id", "meter"])
    # test_df = test_df.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])

    # predict
    results = []
    for model in models:
        if results == []:
            results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        else:
            results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        del model
        gc.collect()

    del test_df, models
    gc.collect()

    results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
    del row_ids, results
    gc.collect()
    results_df.to_csv("submission.csv", index=False)
