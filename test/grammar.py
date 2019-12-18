import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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

    leak_df = pd.read_csv(DATA_PATH + 'leak.csv')
    sub1053 = pd.read_csv(SUB_PATH + 'submission.csv')
    test = pd.read_csv(DATA_PATH + "test.csv")
    test = test.merge(sub1053, on=['row_id'])
    leak_validation(test)

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
