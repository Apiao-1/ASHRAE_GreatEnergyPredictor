import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import datetime
from sklearn import metrics
from meteocalc import feels_like, Temp
import gc
import warnings
import category_encoders as ce


warnings.filterwarnings("ignore")
import warnings
from sklearn.metrics import mean_squared_error

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)


def leak_validation(test_df):
    leak_df = pd.read_csv(DATA_PATH + 'leak.csv')
    leak_df.fillna(0, inplace=True)
    leak_df["time"] = pd.to_datetime(leak_df["timestamp"])
    leak_df = leak_df[(leak_df.time.dt.year > 2016) & (leak_df.time.dt.year < 2019)]
    leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0  # remove large negative values
    leak_df = leak_df[leak_df.building_id != 245]
    print(leak_df.head(20))

    leak_df = leak_df.merge(test_df,
                            left_on=['building_id', 'meter', 'timestamp'],
                            right_on=['building_id', 'meter', 'timestamp'], how="left")
    print(leak_df.head(20))
    leak_df['pred1_l1p'] = np.log1p(leak_df.meter_reading_y)
    leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading_x)
    curr_score = np.sqrt(mean_squared_error(leak_df.pred1_l1p, leak_df.meter_reading_l1p))
    del leak_df
    print('leak Validation: %s' % (curr_score))
    return curr_score


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
    return weather_df


def q80(x):
    return x.quantile(0.8)


def q30(x):
    return x.quantile(0.3)


if __name__ == '__main__':
    DATA_PATH = "../data/"
    SUB_PATH = "../submission/"

    # train = pd.read_csv(DATA_PATH + 'train.csv')
    # bad_rows = pd.read_csv(DATA_PATH + "rows_to_drop.csv")
    # train = train.drop(bad_rows.loc[:, '0']).reset_index(drop=True)
    # print(len(train[train['meter'] == 0]))
    # print(len(train[train['meter'] == 1]))
    # print(len(train[train['meter'] == 2]))
    # print(len(train[train['meter'] == 3]))

    # leak_df = pd.read_feather(DATA_PATH + 'leak.feather')
    # print(leak_df.shape)
    # print(leak_df.head)
    # leak_df.to_csv(DATA_PATH + 'leak.csv', index=False)

    # leak validation
    leak_df = pd.read_csv(DATA_PATH + 'leak.csv')
    sub1053 = pd.read_csv('/Users/a_piao/Desktop/9703.csv')
    test = pd.read_csv(DATA_PATH + "test.csv")
    test = test.merge(sub1053, on=['row_id'])
    leak_validation(test)

    # weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')
    # weather_df = fill_weather_dataset(weather_df)

    # train_df = pd.read_csv(DATA_PATH + 'train.csv')
    # print(train_df.columns)
    # print(train_df.head(5))
    # print(train_df.reset_index(drop=False).head())
    # tmp = train_df["building_id"]
    # target_encoder = ce.TargetEncoder(cols=["building_id"]).fit(train_df, train_df['meter_reading'])
    # train_df = target_encoder.transform(train_df)
    # print(train_df.shape)
    # train_df = pd.concat([train_df,tmp],axis=1)
    # print(train_df.shape)
    # print(train_df.head())
    # temp = train_df[['building_id', 'meter', 'meter_reading']].groupby(['building_id', 'meter']).agg({"meter_reading": [
    #     'min', 'max', 'mean', 'std', 'skew', 'median', q80, q30, pd.DataFrame.kurt, 'mad', np.ptp]})
    # temp.columns = ['meter_min', 'meter_max', 'meter_mean', 'meter_std', 'meter_skew', 'meter_median', 'meter_q80',
    #                 'meter_q30', 'meter_kurt', 'meter_mad', 'meter_ptp']
    # print(temp)
    # train_df = pd.merge(train_df, temp, how='left', on=['building_id', 'meter'])
    # print(train_df.shape)
    # print(train_df.head(20))
    #
    # target = 'meter_reading'
    # categorical = ['building_id', 'site_id', 'primary_use', 'meter', 'dayofweek']
    # # categorical = ['building_id', 'site_id', 'primary_use', 'meter', 'is_holiday', 'dayofweek']
    # numeric_cols = [col for col in train_df.columns if col not in categorical + [target, 'timestamp', 'month']]
    # features = categorical + numeric_cols
    #
    # print(numeric_cols)

    # test = pd.read_csv(DATA_PATH + "test.csv")
    # test = pd.merge(test, temp,  how='left', on=['building_id', 'meter'])
    # print(test.shape)
    # print(test.head(20))
    # print(test.isnull().sum())

    # print(len(test))
    # print(len(leak[leak['meter_reading'] > 0.0]))
    # print(leak.head(20))
    # print(leak.value)

    #
    # # Remove outliers
    # # train_df = train_df[train_df['building_id'] != 1099]
    # # train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
    # #
    # # building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')
    # # weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')
    #
    #
    # df1 = pd.DataFrame({'name': ['kate', 'herz', 'catherine', 'sally'],
    #                     'age': [25, 28, 39, 35]})
    #
    # df2 = pd.DataFrame({'name': ['kate', 'herz', 'sally'],
    #                     'score': [70, 60, 90]})
    # print(df1, df2)
    #
    # # print(pd.merge(df1, df2, left_on="name", right_on='age'))
    #
    # print(pd.merge(df1, df2, left_on="name", right_on='name', how="left"))
    #
