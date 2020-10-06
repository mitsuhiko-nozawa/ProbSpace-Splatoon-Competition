import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random

def make_input(df1, df2, drop_col, categorical_encode, verbose):
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
        df1 = df1.fillna({col: 'none'})
        df2 = df2.fillna({col: 'none'})
        if categorical_encode:
            lbl = LabelEncoder()
            obj = list(set(df1[col].to_list() + df2[col].to_list()))
            lbl.fit(obj)
            df1[col] = lbl.transform(df1[col])
            df2[col] = lbl.transform(df2[col])

    return df1, df2


def addTeamInfo_(df, col_):
    teams = ["-A", "-B"]
    members = ["1", "2", "3", "4"]
    contents = []

    for t, m in itertools.product(teams, members):
        col = col_ + t + m
        df = df.fillna({col: "nan"})
        contents.append(df[col].unique().tolist())

    contents = list(set(itertools.chain.from_iterable(contents)))
    contents.remove("nan")

    for t in teams:
        col = col_ + t
        print(col)
        t1 = pd.crosstab(df.index, df[col + "1"])
        t2 = pd.crosstab(df.index, df[col + "2"])
        t3 = pd.crosstab(df.index, df[col + "3"])
        t4 = pd.crosstab(df.index, df[col + "4"])

        for item in contents:
            df[item + '-' + col_ + t] = (t1[item] + t2[item] + t3[item] + t4[item])

    return df



def addTeamInfo(df1, df2, cols):
    """
    description
    各スペシャル、サブウェポンがチームに何人いるのかを表すカラムを追加
    """

    TRAIN_SIZE = df1.shape[0]

    all_df = pd.concat([df1, df2])
    all_df = all_df.reset_index(drop=True)
    all_df.index += 1
    print(all_df.shape)
    for col in cols:
        all_df = addTeamInfo_(all_df, col)

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
        df[new_col + 'mean'] = df[col_list].mean(axis=1)
        df[new_col + 'median'] = df[col_list].median(axis=1)
        df[new_col + 'std'] = df[col_list].std(axis=1)
        df[new_col + 'sum'] = df[col_list].sum(axis=1)

    return df





def flat(df, cols):
    for col_name in cols:
        for col in [col_ for col_ in df.columns if col_name in col_]:
            df[col] = df[col].apply(np.log2)

    return df


def categorize_team(df1, df2, col_name):
    # team - category1 - A 138
    # team - category1 - B 138
    # team - category2 - A 735
    # team - category2 - B 735
    # team - subweapon - A 1853
    # team - subweapon - B 1853
    # team - special - A 2588
    # team - special - B 2588
    # team - mainweapon - A 56115
    # team - mainweapon - B 56115
    # 1だと135, 2だと735種類
    teams = ["A", "B"]
    for team in teams:
        cols = [col for col in df1.columns if "-"+col_name in col and team == col[-1]]
        # print(cols)
        t_col = "team" + "-" + col_name + "-" + team
        df1[t_col] = ""
        df2[t_col] = ""


        for col in cols:
            df1[t_col] += df1[col].astype("str") + "-"
            df2[t_col] += df2[col].astype("str") + "-"

    return df1, df2


def make_kfolds(SIZE, K):
    # return list object, each element is indices of its fold
    FOLD_SIZE = int(SIZE/K)
    res = []
    indices = [i for i in range(SIZE)]
    for i in range(K-1):
        fold = random.sample(indices, FOLD_SIZE)
        indices = list(set(indices) - set(fold))
        res.append(fold)
    res.append(indices)
    return res

def target_encoding(df1, df2, y_, col, y_col="y", nfolds=5):
    tgt_col = col+"-tgt-enc"
    random.seed(random.randint(0, 10000))
    SIZE = df1.shape[0]
    folds = make_kfolds(SIZE, nfolds)
    all_indices = sum(folds, [])
    if y_col == "y":
        df1[y_col] = y_.values
    print(df1[y_col])
    df1_ = df1[[col, y_col]]


    contents = list(set(df1[col].unique().tolist() + df2[col].unique().tolist()))
    df1[tgt_col] = 0
    df2[tgt_col] = 0
    for i, fold in enumerate(folds):
        print("fold {}".format(i))
        out_fold = list(set(all_indices) - set(fold))

        for content in contents:
            indices = df1_.iloc[fold][df1_[col] == content].index-1  # fold内のある種別を持つインデックスの抽出
            tgt_sum = df1_.iloc[out_fold][df1_[col] == content][y_col].sum()
            tgt_size = df1_.iloc[out_fold][df1_[col] == content][y_col].shape[0]
            df1[tgt_col].iloc[indices] = tgt_sum/(tgt_size+1)  # outfold内の同じ種別のターゲットの平均

    for content in contents:
        if content in df2[col].unique():
            df2[tgt_col][df2[col] == content] = df1[df1[col] == content][tgt_col].mean()
        else :
            pass
            # print("test df doesn't have {} column.".format(content))

    df1 = df1.drop(y_col, axis=1)
    return df1, df2


