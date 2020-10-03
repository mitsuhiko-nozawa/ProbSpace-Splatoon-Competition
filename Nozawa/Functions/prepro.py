import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder

def make_input(train_df_, test_df_, drop_cols_):
    for col in test_df_.columns:
        print(col)
        if col in drop_cols_:
            train_df_.drop(col, axis=1, inplace=True)
            test_df_.drop(col, axis=1, inplace=True)
            continue
        elif test_df_[col].dtype in [int, float]:
            df = pd.concat([train_df_[col], test_df_[col]])
            med = df.median()
            train_df_[col].fillna(med, inplace=True)
            test_df_[col].fillna(med, inplace=True)
            continue
        train_df_[col].fillna('none', inplace=True)
        test_df_[col].fillna('none', inplace=True)
        lbl = LabelEncoder()
        obj = list(set(train_df_[col].to_list() + test_df_[col].to_list()))
        lbl.fit(obj)
        train_df_[col] = lbl.transform(train_df_[col])
        test_df_[col] = lbl.transform(test_df_[col])

    return train_df_, test_df_


def addTeamInfo(train_df_, test_df_):
    """
    description
    各スペシャル、サブウェポンがチームに何人いるのかを表すカラムを追加
    """


    TRAIN_SIZE = train_df_.shape[0]

    all_df = pd.concat([train_df_, test_df_])
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
        all_df[col].fillna("nan", inplace=True)
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
            all_df[item + t] = (t1[item] + t2[item] + t3[item] + t4[item])

    print(all_df.shape)

    train_df_ = all_df.iloc[:TRAIN_SIZE]
    test_df_ = all_df.iloc[TRAIN_SIZE:]
    test_df_ = test_df_.reset_index(drop=True)
    test_df_.index += 1

    print("complete")
    return train_df_, test_df_