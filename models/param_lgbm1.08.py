import sys

sys.path.append('/home/aistudio/external-libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import datetime
import gc
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

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


def create_lag_features(df, window):
    """
    Creating lag-based features looking back in time.
    """

    feature_cols = ["air_temperature"]
    # feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"]
    df_site = df.groupby("site_id")

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)

    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
    df_std = df_rolled.std().reset_index().astype(np.float16)
    df_skew = df_rolled.skew().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f"{feature}_mean_lag{window}"] = df_mean[feature]
        df[f"{feature}_median_lag{window}"] = df_median[feature]
        df[f"{feature}_min_lag{window}"] = df_min[feature]
        df[f"{feature}_max_lag{window}"] = df_max[feature]
        # df[f"{feature}_std_lag{window}"] = df_std[feature]
        # df[f"{feature}_skew_lag{window}"] = df_skew[feature]

    # df.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1,inplace=True)

    return df


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

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    weather_df = create_lag_features(weather_df, 18)

    return weather_df


def features_engineering(df):
    # Sort by timestamp
    df.sort_values("timestamp")
    df.reset_index(drop=True)

    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df['square_feet'] = np.log1p(df['square_feet'])

    # Remove Unused Columns
    drop = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed", "year_built", "floor_count"]
    df = df.drop(drop, axis=1)
    # gc.collect()

    # Encode Categorical Data
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])

    return df


def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 25
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    logging.info(params['max_params'])

    return params


def GBM_evaluate(min_data_in_leaf, min_child_weight, feature_fraction, max_depth, bagging_fraction, lambda_l1,
                 lambda_l2, bagging_freq):
    """自定义的模型评估函数"""

    # 模型固定的超参数
    param = {
        # 'n_estimators': 500,
        'learning_rate': 0.1,

        # 'num_leaves': 32,  # Original 50
        'max_depth': 5,

        'min_data_in_leaf': 49,  # min_child_samples
        # 'max_bin': 58,
        'min_child_weight': 19,

        "feature_fraction": 0.56,  # 0.9 colsample_bytree
        "bagging_freq": 9,
        "bagging_fraction": 0.9,  # 'subsample'
        "bagging_seed": 2019,

        # 'min_split_gain': 0.0,
        "lambda_l1": 0.21,
        "lambda_l2": 0.65,

        "boosting": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 2019,
    }
    global flag
    if flag:
        find_best_param(X, y, param)
        flag = False

    # 贝叶斯优化器生成的超参数
    param['min_child_weight'] = int(min_child_weight)
    param['feature_fraction'] = float(feature_fraction)
    param['max_depth'] = int(max_depth)
    param['bagging_fraction'] = float(bagging_fraction)
    param['bagging_freq'] = int(bagging_freq)
    param['lambda_l2'] = float(lambda_l2)
    param['lambda_l1'] = float(lambda_l1)
    param['min_data_in_leaf'] = int(min_data_in_leaf)

    # 5-flod 交叉检验，注意BayesianOptimization会向最大评估值的方向优化，因此对于回归任务需要取负数。
    # 这里的评估函数为neg_mean_squared_error，即负的MSE。
    val = -find_best_param(X, y, param)

    return val


def find_best_param(features, target, params):
    kf = KFold(n_splits=3)
    models = []
    RMSEs = []
    for train_index, test_index in kf.split(features):
        train_features = features.loc[train_index]
        train_target = target.loc[train_index]

        test_features = features.loc[test_index]
        test_target = target.loc[test_index]

        d_training = lgb.Dataset(train_features, label=train_target, categorical_feature=categorical_features,
                                 free_raw_data=False)
        d_test = lgb.Dataset(test_features, label=test_target, categorical_feature=categorical_features,
                             free_raw_data=False)

        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training, d_test],
                          verbose_eval=False, early_stopping_rounds=50)
        models.append(model)

        y_pred = model.predict(test_features, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(test_target, y_pred))
        # print("single rmse:", rmse)
        RMSEs.append(rmse)

        del train_features, train_target, test_features, test_target, d_training, d_test

    mean_RMSE = np.mean(RMSEs)
    del features, target
    gc.collect()
    logging.info("mean_score: %f" % mean_RMSE)

    global best_score, best_param
    if mean_RMSE <= best_score:
        best_score = mean_RMSE
        logging.info("update best_score: %f" % best_score)
        best_param = params
        logging.info("update best params: %s" % best_param)
    return mean_RMSE


best_score = 9999
best_param = {}
flag=True
X, y = None, None

if __name__ == '__main__':
    init_log()

    # DATA_PATH = "../input/ashrae-energy-prediction/"
    DATA_PATH = "/home/aistudio/data/data17604/"
    # DATA_PATH = "../data/"

    train_df = pd.read_csv(DATA_PATH + 'train.csv')

    # Remove outliers
    train_df = train_df[train_df['building_id'] != 1099]
    train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    building_df = pd.read_csv(DATA_PATH + 'building.csv')
    weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')

    weather_df = fill_weather_dataset(weather_df)

    train_df = train_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
    train_df = train_df.merge(weather_df, how='left', left_on=['site_id', 'timestamp'],
                              right_on=['site_id', 'timestamp'])
    del weather_df
    # gc.collect()

    train_df = features_engineering(train_df)
    # print(train_df.head(10))
    # print(train_df.shape)

    target = np.log1p(train_df["meter_reading"])
    features = train_df.drop('meter_reading', axis=1)
    del train_df
    # gc.collect()

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
    X = features
    y = target

    # 调参范围
    adj_params = {
        'min_child_weight': (3, 50),
        'feature_fraction': (0.4, 1),
        'max_depth': (4, 15),
        'bagging_fraction': (0.5, 1),
        'bagging_freq': (1, 10),
        'lambda_l2': (0.1, 1),
        'lambda_l1': (0.1, 1),
        'min_data_in_leaf': (1, 150)
    }

    # 调用贝叶斯优化
    BayesianSearch(GBM_evaluate, adj_params)

    logging.info("final best param: %s" % best_param)
    logging.info("final best score: %f" % best_score)

    # Important Features
    # for model in models:
    #     plt.figure(figsize=(12, 6))
    #     lgb.plot_importance(model, importance_type="gain")
    #     plt.show()

    # # Load Test Data
    # test_df = pd.read_csv(DATA_PATH + 'test.csv')
    # row_ids = test_df["row_id"]
    # test_df.drop("row_id", axis=1, inplace=True)
    # test_df = reduce_mem_usage(test_df)
    #
    # test_df = test_df.merge(building_df, left_on='building_id', right_on='building_id', how='left')
    # del building_df
    # gc.collect()
    #
    # weather_df = pd.read_csv(DATA_PATH + 'weather_test.csv')
    # weather_df = fill_weather_dataset(weather_df)
    # weather_df = reduce_mem_usage(weather_df)
    #
    # test_df = test_df.merge(weather_df, how='left', on=['timestamp', 'site_id'])
    # del weather_df
    # gc.collect()
    #
    # test_df = features_engineering(test_df)
    #
    # # predict
    # results = []
    # for model in models:
    #     if results == []:
    #         results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    #     else:
    #         results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
    #     del model
    #     gc.collect()
    #
    # del test_df, models
    # gc.collect()
    #
    # results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
    # del row_ids, results
    # gc.collect()
    # results_df.to_csv("submission.csv", index=False)
