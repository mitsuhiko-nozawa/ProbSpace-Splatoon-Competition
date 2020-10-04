import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def make_input(df1, df2, drop_col, verbose):
    cols = df1.columns
    for col in cols:
        if verbose:
            print(col)
        if col in drop_col:
            df1 = df1.drop(col, axis=1)
            df2 = df2.drop(col, axis=1)
            continue
        elif df2[col].dtype in [int, float]:
            df = pd.concat([df1[col], df2[col]])
            med = df.median()
            df1 = df1.fillna({col:med})
            df2 = df2.fillna({col:med})
            continue
        df1 = df1.fillna({col:'none'})
        df2 = df2.fillna({col:'none'})
        lbl = LabelEncoder()
        obj = list(set(df1[col].to_list() + df2[col].to_list()))
        lbl.fit(obj)
        df1[col] = lbl.transform(df1[col])
        df2[col] = lbl.transform(df2[col])

    return df1, df2


def addTeamInfo(df1, df2):
    """
    description
    各スペシャル、サブウェポンがチームに何人いるのかを表すカラムを追加
    """


    TRAIN_SIZE = df1.shape[0]

    all_df = pd.concat([df1, df2])
    all_df = all_df.reset_index(drop=True)
    all_df.index += 1
    print(all_df.shape)

    base_cols = ["special", "subweapon"]
    teams = ["-A", "-B"]
    members = ["1", "2", "3", "4"]
    sps = []
    subs = []

    for b, t, m in itertools.product(base_cols, teams, members):
        col = b + t + m
        all_df = all_df.fillna({col:"nan"})
        if (b == "special"):
            sps.append(all_df[col].unique().tolist())
        else:
            subs.append(all_df[col].unique().tolist())
    sps = list(set(itertools.chain.from_iterable(sps)))
    sps.remove("nan")

    subs = list(set(itertools.chain.from_iterable(subs)))
    subs.remove("nan")

    print(sps)
    print(subs)

    for b, t in itertools.product(base_cols, teams):
        col = b + t
        print(col)
        t1 = pd.crosstab(all_df.index, all_df[col + "1"])
        t2 = pd.crosstab(all_df.index, all_df[col + "2"])
        t3 = pd.crosstab(all_df.index, all_df[col + "3"])
        t4 = pd.crosstab(all_df.index, all_df[col + "4"])
        itr = sps
        if b == "subweapon":
            itr = subs
        for item in itr:
            all_df[item + '-' + b + t] = (t1[item] + t2[item] + t3[item] + t4[item])

    print(all_df.shape)

    df1 = all_df.iloc[:TRAIN_SIZE]
    df2 = all_df.iloc[TRAIN_SIZE:]
    df2 = df2.reset_index(drop=True)
    df2.index += 1

    print("complete")
    return df1, df2


def add_disconnection(df):
    df["disconnection-A"] = df["A1-level"].isnull().astype("int") + \
                            df["A2-level"].isnull().astype("int") + \
                            df["A3-level"].isnull().astype("int") + \
                            df["A4-level"].isnull().astype("int")
    df["disconnection-B"] = df["B1-level"].isnull().astype("int") + \
                            df["B2-level"].isnull().astype("int") + \
                            df["B3-level"].isnull().astype("int") + \
                            df["B4-level"].isnull().astype("int")
    return df


def add_numeric_info(df, cols):
    """
    level
    """
    teams = ["A", "B"]
    for team, col in itertools.product(teams, cols):
        new_col = team + '-' + col + '-'
        col_list = [col_ for col_ in df.columns.to_list() if team in col_ and col in col_]
        df[new_col+'max'] = df[col_list].max(axis=1)
        df[new_col + 'min'] = df[col_list].min(axis=1)
        # df[new_col + 'std'] = df[col_list].std(axis=1)

    return df


def flat(df):
    df["A1-level"] = df["A1-level"].apply(np.log2)
    df["A2-level"] = df["A2-level"].apply(np.log2)
    df["A3-level"] = df["A3-level"].apply(np.log2)
    df["A4-level"] = df["A4-level"].apply(np.log2)
    df["B1-level"] = df["B1-level"].apply(np.log2)
    df["B2-level"] = df["B2-level"].apply(np.log2)
    df["B3-level"] = df["B3-level"].apply(np.log2)
    df["B4-level"] = df["B4-level"].apply(np.log2)

    return df
