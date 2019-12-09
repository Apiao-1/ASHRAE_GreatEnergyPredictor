import pandas as pd

if __name__ == '__main__':
    df1 = pd.DataFrame({'name': ['kate', 'herz', 'catherine', 'sally'],
                        'age': [25, 28, 39, 35]})

    df2 = pd.DataFrame({'name': ['kate', 'herz', 'sally'],
                        'score': [70, 60, 90]})
    print(df1, df2)

    # print(pd.merge(df1, df2, left_on="name", right_on='age'))

    print(pd.merge(df1, df2, left_on="name", right_on='name', how="left"))

