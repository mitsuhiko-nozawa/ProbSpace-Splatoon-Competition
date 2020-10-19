import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import category_encoders.target_encoder as te

import random

def make_input(df1, df2, drop_col, categorical_encode, scaler, verbose):

    all_df = pd.concat([df1, df2])
    df1 = df1.drop(columns=drop_col, axis=1)
    df2 = df2.drop(columns=drop_col, axis=1)
    cols = df1.columns
    for col in cols:
        if verbose:
            print(col)


        elif df2[col].dtype in [int, float]:
            if df1[col].value_counts().shape[0] < 5:
                continue
            if scaler and "onehot" not in col:
                # df1[col] = df1[col].apply(np.log1p)
                # df2[col] = df2[col].apply(np.log1p)
                # all_df[col] = all_df[col].apply(np.log1p)
                # NN 二つ併用するとlossがnanになる
                scaler = StandardScaler()
                print("scale")
                scaler.fit(all_df[col].values.reshape(-1, 1))
                df1[col] = scaler.transform(df1[col].values.reshape(-1, 1))
                df2[col] = scaler.transform(df2[col].values.reshape(-1, 1))
            continue
        if categorical_encode:
            lbl = LabelEncoder()
            obj = all_df[col].unique()
            lbl.fit(obj)
            df1[col] = lbl.transform(df1[col])
            df2[col] = lbl.transform(df2[col])
    print("complete")
    return df1, df2


def addTeamInfo_(df, col_):
    teams = ["-A", "-B"]
    members = ["1", "2", "3", "4"]
    contents = []

    for t, m in itertools.product(teams, members):
        col = col_ + t + m
        #df = df.fillna({col: "nan"})
        contents.append(df[col].unique().tolist())

    contents = list(set(itertools.chain.from_iterable(contents)))
    #contents.remove("nan")

    for t in teams:
        col = col_ + t
        print(col)
        t1 = pd.crosstab(df.index, df[col + "1"])
        t2 = pd.crosstab(df.index, df[col + "2"])
        t3 = pd.crosstab(df.index, df[col + "3"])
        t4 = pd.crosstab(df.index, df[col + "4"])

        for item in contents:
            t_ = t.replace('-', '') + '-'
            col_name = t_ + item + '-' + col_ + "-onehot"
            df[col_name] = 0
            tabs = [t1, t2, t3, t4]
            for tab in tabs:
                try:
                    df[col_name] += tab[item]
                except:
                    #print("Key {} does not exist".format(item))
                    pass


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
        col_list = [col_ for col_ in df.columns if team in col_ and col in col_]
        df[new_col+'max'] = df[col_list].max(axis=1)
        df[new_col + 'min'] = df[col_list].min(axis=1)
        df[new_col + 'mean'] = df[col_list].mean(axis=1)
        df[new_col + 'median'] = df[col_list].median(axis=1)
        df[new_col + 'std'] = df[col_list].std(axis=1)
        # df[new_col + 'sum'] = df[col_list].sum(axis=1)

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
        cols = [col for col in df1.columns if "-"+col_name in col and team == col[0] and "onehot" in col]
        # print(cols)
        t_col = "team" + "-" + col_name + "-" + team
        df1[t_col] = ""
        df2[t_col] = ""


        for col in cols:
            col_nam = col.replace(team, "").replace(col_name, "").replace("onehot", "").replace("-", "")
            df1[t_col] += col_nam + df1[col].astype("str") + "-"
            df2[t_col] += col_nam + df2[col].astype("str") + "-"

    return df1, df2


def make_kfolds(SIZE, K):
    # return list object, each element is indices of its fold, completely random
    random.seed(random.randint(0, 10000))
    FOLD_SIZE = int(SIZE/K)
    res = []
    indices = [i for i in range(SIZE)]
    for i in range(K-1):
        fold = random.sample(indices, FOLD_SIZE)
        indices = list(set(indices) - set(fold))
        res.append(fold)
    res.append(indices)
    return res


def target_encoder(df1, df2, col, y, n_splits=5, drop=True):
    kf = KFold(n_splits=n_splits, random_state=random.randint(0, 100000), shuffle=True)
    col_name = "tgtenc-" + col
    df1[col_name] = 0
    for train_index, test_index in kf.split(df1):
        enc = te.TargetEncoder().fit(df1.iloc[train_index][col].astype("str"), y[train_index])
        df1[col_name].iloc[test_index] = enc.transform(df1.iloc[test_index][col].astype("str"))[col]
    enc = te.TargetEncoder().fit(df1[col].astype("str"), y)
    df2[col_name] = enc.transform(df2[col].astype("str"))[col]
    if drop:
        df1 = df1.drop(col, axis=1)
        df2 = df2.drop(col, axis=1)
    return df1, df2


def target_encoding(df1, df2, y_, col, y_col="y", nfolds=5):
    df1 = df1.reset_index(drop=True)
    tgt_col = col + "-tgt-enc"
    SIZE = df1.shape[0]
    folds = make_kfolds(SIZE, nfolds)
    all_indices = sum(folds, [])
    if y_col == "y":
        df1[y_col] = y_
    df1_ = df1[[col, y_col]]

    contents = list(set(df1[col].unique().tolist() + df2[col].unique().tolist()))
    df1[tgt_col] = 0
    df2[tgt_col] = 0
    for i, fold in enumerate(folds):
        out_fold = list(set(all_indices) - set(fold))
        group_sum = df1_.iloc[out_fold].groupby(col)[y_col].sum()
        group_count = df1_.iloc[out_fold].groupby(col)[y_col].count() + 1
        for content in contents:
            indices = df1_.iloc[fold][df1_[col] == content].index - 1
            try:
                # df1[df1[col] == content][tgt_col].iloc[fold] = group_sum[content] / group_count[content]
                df1[tgt_col].iloc[indices] = group_sum[content] / group_count[content]
            except:
                df1[tgt_col].iloc[indices] = 0

    group_sum = df1_.groupby(col)[y_col].sum()
    group_count = df1_.groupby(col)[y_col].count() + 1

    for content in contents:
        try:
            tag = group_sum[content] / group_count[content]
        except:
            tag = 0

        try:
            df2[tgt_col][df1[col] == content] = tag
        except:
            pass

    df1 = df1.drop(y_col, axis=1)
    df1 = df1.drop(col, axis=1)
    df2 = df2.drop(col, axis=1)
    return df1, df2


def mizumashi(df, y):
    try:
        y = y.values
    except:
        pass

    cat_cols = df.select_dtypes(include=["category"]).columns

    df["y"] = y
    df2 = df.copy()
    A1_col = [col for col in df.columns if "A1" in col]
    A2_col = [col for col in df.columns if "A2" in col]
    A3_col = [col for col in df.columns if "A3" in col]
    A4_col = [col for col in df.columns if "A4" in col]

    A_col = [col for col in df.columns if "A-" in col]

    for col in A1_col:
        c_col = col.replace("A1", "C1")
        b_col = col.replace("A1", "B1")
        df2.rename(columns={col: c_col}, inplace=True)
        df2.rename(columns={b_col: col}, inplace=True)
        df2.rename(columns={c_col: b_col}, inplace=True)

    for col in A2_col:
        c_col = col.replace("A2", "C2")
        b_col = col.replace("A2", "B2")
        df2.rename(columns={col: c_col}, inplace=True)
        df2.rename(columns={b_col: col}, inplace=True)
        df2.rename(columns={c_col: b_col}, inplace=True)

    for col in A3_col:
        c_col = col.replace("A3", "C3")
        b_col = col.replace("A3", "B3")
        df2.rename(columns={col: c_col}, inplace=True)
        df2.rename(columns={b_col: col}, inplace=True)
        df2.rename(columns={c_col: b_col}, inplace=True)

    for col in A4_col:
        c_col = col.replace("A4", "C4")
        b_col = col.replace("A4", "B4")
        df2.rename(columns={col: c_col}, inplace=True)
        df2.rename(columns={b_col: col}, inplace=True)
        df2.rename(columns={c_col: b_col}, inplace=True)

    for col in A_col:
        c_col = col.replace("A-", "C-")
        b_col = col.replace("A-", "B-")
        df2.rename(columns={col: c_col}, inplace=True)
        df2.rename(columns={b_col: col}, inplace=True)
        df2.rename(columns={c_col: b_col}, inplace=True)

    df2["y"] = df2["y"].apply(lambda x: 1 - x)
    all_df = pd.concat([df, df2])
    y = all_df["y"].values
    all_df = all_df.drop("y", axis=1)
    all_df[cat_cols] = all_df[cat_cols].astype("category")

    return all_df, y


def count_weapon(df1, df2):
    all_df = pd.concat([df1, df2])
    SIZE = all_df.shape[0] * 7
    weap_cols = [col for col in df1.columns if "-weapon" in col and "A1" not in col]
    weap_counts = all_df[weap_cols[-1]].value_counts()

    for col in weap_cols[:-1]:
        weap_counts.add(all_df[col].value_counts())

    def func(x):
        return weap_counts[x]/SIZE if x == x else 0

    for col in [col for col in df1.columns if "-weapon" in col]:
        df1[col + "-count"] = df1[col].map(func)
        df2[col + "-count"] = df2[col].map(func)

    return df1, df2


def count_weapon_by_mode(df1, df2):
    all_df = pd.concat([df1, df2])


    for col in [col for col in df1.columns if "-weapon" in col and "count" not in col]:
        col_name = col + "-count-by-mode"
        df1[col_name] = 0
        df2[col_name] = 0
    for mode in df1["mode"].unique():
        SIZE = all_df[all_df["mode"] == mode].shape[0] * 7

        weap_cols = [col for col in df1.columns if "-weapon" in col and "A1" not in col and "count" not in col]
        weap_counts = all_df[all_df["mode"] == mode][weap_cols[-1]].value_counts()

        for col in weap_cols[:-1]:
            weap_counts.add(all_df[all_df["mode"] == mode][col].value_counts())

        def func(x):
            return weap_counts[x] / SIZE if x == x else 0

        for col in [col for col in df1.columns if "-weapon" in col and "count" not in col]:
            col_name = col + "-count-by-mode"
            df1[col_name][df1["mode"] == mode] = df1[col][df1["mode"] == mode].map(func)
            df2[col_name][df2["mode"] == mode] = df2[col][df2["mode"] == mode].map(func)

    return df1, df2


def is_nawabari(df1, df2):
    df1["is_nawabari"] = df1["mode"].map(lambda x: 1 if x == "nawabari" else 0)
    df2["is_nawabari"] = df2["mode"].map(lambda x: 1 if x == "nawabari" else 0)
    return df1, df2


def match_rank(df1, df2):
    col_name = "match_rank"
    def func(x):
        if x != x:
            return "-"
        x = x.replace("+", "").replace("-", "")
        return x
    df1[col_name] = df1["A1-rank"].map(func)
    df2[col_name] = df2["A1-rank"].map(func)
    return df1, df2


def fillna_rank(df1, df2):
    df1["A4-rank"][df1["A4-level"].isnull()] = "none"
    df2["A4-rank"][df2["A4-level"].isnull()] = "none"
    df1["B3-rank"][df1["B3-level"].isnull()] = "none"
    df2["B3-rank"][df2["B3-level"].isnull()] = "none"
    df1["B4-rank"][df1["B4-level"].isnull()] = "none"
    df2["B4-rank"][df2["B4-level"].isnull()] = "none"

    df1["A1-rank"][df1["A1-rank"].isnull()] = "nawabari"
    df2["A1-rank"][df2["A1-rank"].isnull()] = "nawabari"
    df1["A2-rank"][df1["A2-rank"].isnull()] = "nawabari"
    df2["A2-rank"][df2["A2-rank"].isnull()] = "nawabari"
    df1["A3-rank"][df1["A3-rank"].isnull()] = "nawabari"
    df2["A3-rank"][df2["A3-rank"].isnull()] = "nawabari"
    df1["A4-rank"][df1["A4-rank"].isnull()] = "nawabari"
    df2["A4-rank"][df2["A4-rank"].isnull()] = "nawabari"

    df1["B1-rank"][df1["B1-rank"].isnull()] = "nawabari"
    df2["B1-rank"][df2["B1-rank"].isnull()] = "nawabari"
    df1["B2-rank"][df1["B2-rank"].isnull()] = "nawabari"
    df2["B2-rank"][df2["B2-rank"].isnull()] = "nawabari"
    df1["B3-rank"][df1["B3-rank"].isnull()] = "nawabari"
    df2["B3-rank"][df2["B3-rank"].isnull()] = "nawabari"
    df1["B4-rank"][df1["B4-rank"].isnull()] = "nawabari"
    df2["B4-rank"][df2["B4-rank"].isnull()] = "nawabari"
    return df1, df2

def fillna(df1, df2):
    all_df = pd.concat([df1, df2])
    missing_cols = [col for col in all_df.columns if all_df[col].isnull().any()]
    for col in missing_cols:
        d_type = all_df[col].dtype
        if d_type in [int, float]:
            df1[col] = df1[col].fillna(-9999)
            df2[col] = df2[col].fillna(-9999)
        else:
            df1[col] = df1[col].fillna("none")
            df2[col] = df2[col].fillna("none")
    return df1, df2


def make_stratified_kfolds(X_, y_, K, shuffle, random_state=0):
    skf = StratifiedKFold(n_splits=K, shuffle=shuffle, random_state=random_state)
    folds_ = []
    for train_fold, valid_fold in skf.split(X_, y_):
        folds_.append(list(valid_fold))
    return folds_


def make_even_kfolds(X, y_1, K):
    random.seed(random.randint(0, 10000))
    X_ = X.copy()
    X_["y"] = y_1
    folds_ = []
    for i in range(K):
        folds_.append([])

    for mode, stage, y_ in itertools.product(X_["mode"].unique(), X_["stage"].unique(), X_["y"].unique()):
        indices = list(X_[X_["mode"] == mode][X_["stage"] == stage][X_["y"] == y_].index - 1)

        SAMPLE_SIZE = int(len(indices) / K)

        for i in range(K - 1):
            fold = random.sample(indices, SAMPLE_SIZE)
            indices = list(set(indices) - set(fold))
            folds_[i] += fold

        folds_[-1] += indices

    return folds_


def make_mode_even_kfolds(X, y_1, K):
    random.seed(random.randint(0, 10000))
    X_ = X.copy()
    X_["y"] = y_1
    folds_ = []
    for i in range(K):
        folds_.append([])

    for mode, y_ in itertools.product(X_["mode"].unique(), X_["y"].unique()):
        indices = list(X_[X_["mode"] == mode][X_["y"] == y_].index - 1)

        SAMPLE_SIZE = int(len(indices) / K)

        for i in range(K - 1):
            fold = random.sample(indices, SAMPLE_SIZE)
            indices = list(set(indices) - set(fold))
            folds_[i] += fold

        folds_[-1] += indices

    return folds_


def count_reskin(df1, df2):
    all_df = pd.concat([df1, df2])
    SIZE = all_df.shape[0] * 7
    reskin_cols = [col for col in df1.columns if "reskin" in col and "A1" not in col]
    reskin_counts = all_df[reskin_cols[-1]].value_counts()

    for col in reskin_cols[:-1]:
        reskin_counts.add(all_df[col].value_counts())

    def func(x):
        return reskin_counts[x]/SIZE if x == x else 0

    for col in [col for col in df1.columns if "reskin" in col]:
        df1[col + "-count"] = df1[col].map(func)
        df2[col + "-count"] = df2[col].map(func)

    return df1, df2


def count_reskin_by_mode(df1, df2):
    all_df = pd.concat([df1, df2])


    for col in [col for col in df1.columns if "reskin" in col and "count" not in col]:
        col_name = col + "-count-by-mode"
        df1[col_name] = 0
        df2[col_name] = 0
    for mode in df1["mode"].unique():
        SIZE = all_df[all_df["mode"] == mode].shape[0] * 7

        weap_cols = [col for col in df1.columns if "reskin" in col and "A1" not in col and "count" not in col]
        weap_counts = all_df[all_df["mode"] == mode][weap_cols[-1]].value_counts()

        for col in weap_cols[:-1]:
            weap_counts.add(all_df[all_df["mode"] == mode][col].value_counts())

        def func(x):
            return weap_counts[x] / SIZE if x == x else 0

        for col in [col for col in df1.columns if "reskin" in col and "count" not in col]:
            col_name = col + "-count-by-mode"
            df1[col_name][df1["mode"] == mode] = df1[col][df1["mode"] == mode].map(func)
            df2[col_name][df2["mode"] == mode] = df2[col][df2["mode"] == mode].map(func)

    return df1, df2

def count_subweapon(df1, df2):
    all_df = pd.concat([df1, df2])
    SIZE = all_df.shape[0] * 7
    reskin_cols = [col for col in df1.columns if "subweapon" in col and "A1" not in col]
    reskin_counts = all_df[reskin_cols[-1]].value_counts()

    for col in reskin_cols[:-1]:
        reskin_counts.add(all_df[col].value_counts())

    def func(x):
        return reskin_counts[x]/SIZE if x == x else 0

    for col in [col for col in df1.columns if "subweapon" in col]:
        df1[col + "-count"] = df1[col].map(func)
        df2[col + "-count"] = df2[col].map(func)

    return df1, df2


def count_subweapon_by_mode(df1, df2):
    all_df = pd.concat([df1, df2])


    for col in [col for col in df1.columns if "subweapon" in col and "count" not in col]:
        col_name = col + "-count-by-mode"
        df1[col_name] = 0
        df2[col_name] = 0
    for mode in df1["mode"].unique():
        SIZE = all_df[all_df["mode"] == mode].shape[0] * 7

        weap_cols = [col for col in df1.columns if "subweapon" in col and "A1" not in col and "count" not in col]
        weap_counts = all_df[all_df["mode"] == mode][weap_cols[-1]].value_counts()

        for col in weap_cols[:-1]:
            weap_counts.add(all_df[all_df["mode"] == mode][col].value_counts())

        def func(x):
            return weap_counts[x] / SIZE if x == x else 0

        for col in [col for col in df1.columns if "subweapon" in col and "count" not in col]:
            col_name = col + "-count-by-mode"
            df1[col_name][df1["mode"] == mode] = df1[col][df1["mode"] == mode].map(func)
            df2[col_name][df2["mode"] == mode] = df2[col][df2["mode"] == mode].map(func)

    return df1, df2

def count_mainweapon(df1, df2):
    all_df = pd.concat([df1, df2])
    SIZE = all_df.shape[0] * 7
    reskin_cols = [col for col in df1.columns if "mainweapon" in col and "A1" not in col]
    reskin_counts = all_df[reskin_cols[-1]].value_counts()

    for col in reskin_cols[:-1]:
        reskin_counts.add(all_df[col].value_counts())

    def func(x):
        return reskin_counts[x]/SIZE if x == x else 0

    for col in [col for col in df1.columns if "mainweapon" in col]:
        df1[col + "-count"] = df1[col].map(func)
        df2[col + "-count"] = df2[col].map(func)

    return df1, df2


def count_mainweapon_by_mode(df1, df2):
    all_df = pd.concat([df1, df2])


    for col in [col for col in df1.columns if "mainweapon" in col and "count" not in col]:
        col_name = col + "-count-by-mode"
        df1[col_name] = 0
        df2[col_name] = 0
    for mode in df1["mode"].unique():
        SIZE = all_df[all_df["mode"] == mode].shape[0] * 7

        weap_cols = [col for col in df1.columns if "mainweapon" in col and "A1" not in col and "count" not in col]
        weap_counts = all_df[all_df["mode"] == mode][weap_cols[-1]].value_counts()

        for col in weap_cols[:-1]:
            weap_counts.add(all_df[all_df["mode"] == mode][col].value_counts())

        def func(x):
            return weap_counts[x] / SIZE if x == x else 0

        for col in [col for col in df1.columns if "mainweapon" in col and "count" not in col]:
            col_name = col + "-count-by-mode"
            df1[col_name][df1["mode"] == mode] = df1[col][df1["mode"] == mode].map(func)
            df2[col_name][df2["mode"] == mode] = df2[col][df2["mode"] == mode].map(func)

    return df1, df2

def count_special(df1, df2):
    all_df = pd.concat([df1, df2])
    SIZE = all_df.shape[0] * 7
    reskin_cols = [col for col in df1.columns if "special" in col and "-special" not in col and "A1" not in col]
    reskin_counts = all_df[reskin_cols[-1]].value_counts()

    for col in reskin_cols[:-1]:
        reskin_counts.add(all_df[col].value_counts())

    def func(x):
        return reskin_counts[x]/SIZE if x == x else 0

    for col in [col for col in df1.columns if "special" in col and "-special" not in col]:
        df1[col + "-count"] = df1[col].map(func)
        df2[col + "-count"] = df2[col].map(func)

    return df1, df2


def count_special_by_mode(df1, df2):
    all_df = pd.concat([df1, df2])


    for col in [col for col in df1.columns if "special" in col and "-special" not in col and "count" not in col]:
        col_name = col + "-count-by-mode"
        df1[col_name] = 0
        df2[col_name] = 0
    for mode in df1["mode"].unique():
        SIZE = all_df[all_df["mode"] == mode].shape[0] * 7

        weap_cols = [col for col in df1.columns if "special" in col and "-special" not in col and "A1" not in col and "count" not in col]
        weap_counts = all_df[all_df["mode"] == mode][weap_cols[-1]].value_counts()

        for col in weap_cols[:-1]:
            weap_counts.add(all_df[all_df["mode"] == mode][col].value_counts())

        def func(x):
            return weap_counts[x] / SIZE if x == x else 0

        for col in [col for col in df1.columns if "special" in col and "-special" not in col and "count" not in col]:
            col_name = col + "-count-by-mode"
            df1[col_name][df1["mode"] == mode] = df1[col][df1["mode"] == mode].map(func)
            df2[col_name][df2["mode"] == mode] = df2[col][df2["mode"] == mode].map(func)

    return df1, df2


def prod(df1, df2, col1, col2):
    col_name = col1 + " x " + col2
    df1[col_name] = df1[col1] + " * " + df1[col2]
    df2[col_name] = df2[col1] + " * " + df2[col2]

    return df1, df2


def identify_A1(df1, df2):
    all_df = pd.concat([df1, df2]).reset_index(drop=True)

    def get_seq_labels(seq, threshold=0):
        """
        seq : 時系列順のリスト
        threshold : level up のための最低試合数

        [3,3,3,4,4,4,4,4,7,7,7,7,2,2,2,1,1,1,2,8,8,8] : level
         => [1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,3,2,2,2] : player id
        というように、レベルに応じてA1の特定を考えます
        """
        box = np.zeros(len(seq), dtype=int)  # 最終的にラベルが入るボックス
        count = 1  # label
        for _ in (range(1000)):
            # level : 時系列順のレベルでユニークなもの
            # s     : levelの値を格納
            ind = np.where(box == 0)[0][0]
            s = seq[ind]
            renew_box = []
            for i in range(len(seq)):
                if box[i] == 0:
                    if s == seq[i]:
                        box[i] = count
                        renew_box.append(seq[i])
                    elif (s + 1 == seq[i]) and ((np.array(renew_box) == s).sum() >= threshold):
                        s += 1
                        box[i] = count
                else:
                    continue
            count += 1
            if (box == 0).sum() == 0:
                # box が全部埋まれば break
                break
        return box
    all_df = all_df.sort_values(["period", "A1-level"])
    levels = all_df["A1-level"].tolist()  # A1 level を時系列順にソートしたリスト
    all_df["a1-player"] = get_seq_labels(levels, 15)
    all_df = all_df.sort_index()
    df1 = all_df[:df1.shape[0]]
    df2 = all_df[df1.shape[0]:].reset_index(drop=True)
    return df1, df2

def level_inverse(df1,df2):
    all_df = pd.concat([df1, df2])

    A1_level_df = all_df[["A1-rank", "A1-level"]]
    A1_level_df.columns = ["rank", "level"]
    A2_level_df = all_df[["A2-rank", "A2-level"]]
    A2_level_df.columns = ["rank", "level"]
    A3_level_df = all_df[["A3-rank", "A3-level"]]
    A3_level_df.columns = ["rank", "level"]
    A4_level_df = all_df[["A4-rank", "A4-level"]]
    A4_level_df.columns = ["rank", "level"]

    B1_level_df = all_df[["B1-rank", "B1-level"]]
    B1_level_df.columns = ["rank", "level"]
    B2_level_df = all_df[["B2-rank", "B2-level"]]
    B2_level_df.columns = ["rank", "level"]
    B3_level_df = all_df[["B3-rank", "B3-level"]]
    B3_level_df.columns = ["rank", "level"]
    B4_level_df = all_df[["B4-rank", "B4-level"]]
    B4_level_df.columns = ["rank", "level"]

    level_df = pd.concat([A2_level_df, A3_level_df, A4_level_df, B1_level_df, B2_level_df, B3_level_df, B4_level_df])

    rank_g = level_df.groupby("rank")["level"].mean()

    df1["A1-level-inverse"] = df1[["A1-level", "A1-rank"]].apply(lambda x: x["A1-level"] / rank_g[x["A1-rank"]], axis=1)
    df1["A2-level-inverse"] = df1[["A2-level", "A2-rank"]].apply(lambda x: x["A2-level"] / rank_g[x["A2-rank"]], axis=1)
    df1["A3-level-inverse"] = df1[["A3-level", "A3-rank"]].apply(lambda x: x["A3-level"] / rank_g[x["A3-rank"]], axis=1)
    df1["A4-level-inverse"] = df1[["A4-level", "A4-rank"]].apply(lambda x: x["A4-level"] / rank_g[x["A4-rank"]], axis=1)
    df1["B1-level-inverse"] = df1[["B1-level", "B1-rank"]].apply(lambda x: x["B1-level"] / rank_g[x["B1-rank"]], axis=1)
    df1["B2-level-inverse"] = df1[["B2-level", "B2-rank"]].apply(lambda x: x["B2-level"] / rank_g[x["B2-rank"]], axis=1)
    df1["B3-level-inverse"] = df1[["B3-level", "B3-rank"]].apply(lambda x: x["B3-level"] / rank_g[x["B3-rank"]], axis=1)
    df1["B4-level-inverse"] = df1[["B4-level", "B4-rank"]].apply(lambda x: x["B4-level"] / rank_g[x["B4-rank"]], axis=1)

    df2["A1-level-inverse"] = df2[["A1-level", "A1-rank"]].apply(lambda x: x["A1-level"] / rank_g[x["A1-rank"]], axis=1)
    df2["A2-level-inverse"] = df2[["A2-level", "A2-rank"]].apply(lambda x: x["A2-level"] / rank_g[x["A2-rank"]], axis=1)
    df2["A3-level-inverse"] = df2[["A3-level", "A3-rank"]].apply(lambda x: x["A3-level"] / rank_g[x["A3-rank"]], axis=1)
    df2["A4-level-inverse"] = df2[["A4-level", "A4-rank"]].apply(lambda x: x["A4-level"] / rank_g[x["A4-rank"]], axis=1)
    df2["B1-level-inverse"] = df2[["B1-level", "B1-rank"]].apply(lambda x: x["B1-level"] / rank_g[x["B1-rank"]], axis=1)
    df2["B2-level-inverse"] = df2[["B2-level", "B2-rank"]].apply(lambda x: x["B2-level"] / rank_g[x["B2-rank"]], axis=1)
    df2["B3-level-inverse"] = df2[["B3-level", "B3-rank"]].apply(lambda x: x["B3-level"] / rank_g[x["B3-rank"]], axis=1)
    df2["B4-level-inverse"] = df2[["B4-level", "B4-rank"]].apply(lambda x: x["B4-level"] / rank_g[x["B4-rank"]], axis=1)
    return df1, df2


def mizumashi_perm(df, y):
    pass


def reskin_tgt_encoding(df1, df2, y, n_splits=5):
    kf = KFold(n_splits=n_splits, random_state=random.randint(0, 100000), shuffle=True)
    df1["y"] = y
    i = 0
    df1["mode x reskin-A1-tgtenc"] = 0
    df1["mode x reskin-A2-tgtenc"] = 0
    df1["mode x reskin-A3-tgtenc"] = 0
    df1["mode x reskin-A4-tgtenc"] = 0
    df1["mode x reskin-B1-tgtenc"] = 0
    df1["mode x reskin-B2-tgtenc"] = 0
    df1["mode x reskin-B3-tgtenc"] = 0
    df1["mode x reskin-B4-tgtenc"] = 0
    df2["mode x reskin-A1-tgtenc"] = 0
    df2["mode x reskin-A2-tgtenc"] = 0
    df2["mode x reskin-A3-tgtenc"] = 0
    df2["mode x reskin-A4-tgtenc"] = 0
    df2["mode x reskin-B1-tgtenc"] = 0
    df2["mode x reskin-B2-tgtenc"] = 0
    df2["mode x reskin-B3-tgtenc"] = 0
    df2["mode x reskin-B4-tgtenc"] = 0

    for train_index, test_index in kf.split(df1):
        i += 1
        print(
            "----------------------------------------------- fold {} ----------------------------------------------- ".format(
                i))
        reskin_g = pd.concat([
            df1.iloc[train_index][["mode x reskin-A2", "y"]].rename(columns={"mode x reskin-A2": "reskin"}),
            df1.iloc[train_index][["mode x reskin-A3", "y"]].rename(columns={"mode x reskin-A3": "reskin"}),
            df1.iloc[train_index][["mode x reskin-A4", "y"]].rename(columns={"mode x reskin-A4": "reskin"}),
            pd.concat([df1.iloc[train_index]["mode x reskin-B1"], df1.iloc[train_index]["y"].apply(lambda x: 1 - x)],
                      axis=1).rename(columns={"mode x reskin-B1": "reskin"}),
            pd.concat([df1.iloc[train_index]["mode x reskin-B2"], df1.iloc[train_index]["y"].apply(lambda x: 1 - x)],
                      axis=1).rename(columns={"mode x reskin-B2": "reskin"}),
            pd.concat([df1.iloc[train_index]["mode x reskin-B3"], df1.iloc[train_index]["y"].apply(lambda x: 1 - x)],
                      axis=1).rename(columns={"mode x reskin-B3": "reskin"}),
            pd.concat([df1.iloc[train_index]["mode x reskin-B4"], df1.iloc[train_index]["y"].apply(lambda x: 1 - x)],
                      axis=1).rename(columns={"mode x reskin-B4": "reskin"}),
        ], axis=0, ignore_index=True).groupby("reskin")
        win_rate = win_rate = reskin_g.sum() / reskin_g.count()
        df1["mode x reskin-A1-tgtenc"].iloc[test_index] = df1["mode x reskin-A1"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-A2-tgtenc"].iloc[test_index] = df1["mode x reskin-A2"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-A3-tgtenc"].iloc[test_index] = df1["mode x reskin-A3"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-A4-tgtenc"].iloc[test_index] = df1["mode x reskin-A4"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-B1-tgtenc"].iloc[test_index] = df1["mode x reskin-B1"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-B2-tgtenc"].iloc[test_index] = df1["mode x reskin-B2"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-B3-tgtenc"].iloc[test_index] = df1["mode x reskin-B3"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
        df1["mode x reskin-B4-tgtenc"].iloc[test_index] = df1["mode x reskin-B4"].iloc[test_index].apply(
            lambda x: win_rate.loc[x]).values.reshape(-1, )
    print("----------------------------------------------- 終わり ----------------------------------------------- ")
    reskin_g = pd.concat([
        # df1[["mode x reskin-A1", "y"]].rename(columns={"mode x reskin-A1" : "reskin"}),
        df1[["mode x reskin-A2", "y"]].rename(columns={"mode x reskin-A2": "reskin"}),
        df1[["mode x reskin-A3", "y"]].rename(columns={"mode x reskin-A3": "reskin"}),
        df1[["mode x reskin-A4", "y"]].rename(columns={"mode x reskin-A4": "reskin"}),
        pd.concat([df1["mode x reskin-B1"], df1["y"].apply(lambda x: 1 - x)],
                  axis=1).rename(columns={"mode x reskin-B1": "reskin"}),
        pd.concat([df1["mode x reskin-B2"], df1["y"].apply(lambda x: 1 - x)],
                  axis=1).rename(columns={"mode x reskin-B2": "reskin"}),
        pd.concat([df1["mode x reskin-B3"], df1["y"].apply(lambda x: 1 - x)],
                  axis=1).rename(columns={"mode x reskin-B3": "reskin"}),
        pd.concat([df1["mode x reskin-B4"], df1["y"].apply(lambda x: 1 - x)],
                  axis=1).rename(columns={"mode x reskin-B4": "reskin"}),
    ], axis=0, ignore_index=True).groupby("reskin")
    win_rate = win_rate = reskin_g.sum() / reskin_g.count()
    df2["mode x reskin-A1-tgtenc"] = df2["mode x reskin-A1"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-A2-tgtenc"] = df2["mode x reskin-A2"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-A3-tgtenc"] = df2["mode x reskin-A3"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-A4-tgtenc"] = df2["mode x reskin-A4"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-B1-tgtenc"] = df2["mode x reskin-B1"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-B2-tgtenc"] = df2["mode x reskin-B2"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-B3-tgtenc"] = df2["mode x reskin-B3"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df2["mode x reskin-B4-tgtenc"] = df2["mode x reskin-B4"].apply(lambda x: win_rate.loc[x]).values.reshape(-1, )
    df1.drop(columns="y", inplace=True)
    drop_cols = [
        "mode x reskin-A1", "mode x reskin-A2", "mode x reskin-A3", "mode x reskin-A4",
        "mode x reskin-B1", "mode x reskin-B2", "mode x reskin-B3", "mode x reskin-B4"
    ]
    df1.drop(columns=drop_cols, inplace=True)
    df2.drop(columns=drop_cols, inplace=True)

    return df1, df2

def ranker(df1, df2):
    df1["rank-mark-A1"] = df1["A1-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["A1-rank"] = df1["A1-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-A2"] = df1["A2-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["A2-rank"] = df1["A2-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-A3"] = df1["A3-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["A3-rank"] = df1["A3-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-A4"] = df1["A4-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["A4-rank"] = df1["A4-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-B1"] = df1["B1-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["B1-rank"] = df1["B1-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-B2"] = df1["B2-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["B2-rank"] = df1["B2-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-B3"] = df1["B3-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["B3-rank"] = df1["B3-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df1["rank-mark-B4"] = df1["B4-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df1["B4-rank"] = df1["B4-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-A1"] = df2["A1-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["A1-rank"] = df2["A1-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-A2"] = df2["A2-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["A2-rank"] = df2["A2-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-A3"] = df2["A3-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["A3-rank"] = df2["A3-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-A4"] = df2["A4-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["A4-rank"] = df2["A4-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-B1"] = df2["B1-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["B1-rank"] = df2["B1-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-B2"] = df2["B2-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["B2-rank"] = df2["B2-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-B3"] = df2["B3-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["B3-rank"] = df2["B3-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    df2["rank-mark-B4"] = df2["B4-rank"].apply(lambda x: "*" if len(x)==1 else x[1])
    #df2["B4-rank"] = df2["B4-rank"].apply(lambda x: x[0] if len(x) <= 2 else x)
    drop_cols = [
        "A1-rank", "A2-rank", "A3-rank", "A4-rank",
        "B1-rank", "B2-rank", "B3-rank", "B4-rank"
    ]
    df1.drop(columns=drop_cols, inplace=True)
    df2.drop(columns=drop_cols, inplace=True)



    return df1, df2


def find_rare(df1, df2, col, threshold=4):
    v_tra = df1[col + "-A"].value_counts()
    v_trb = df1[col + "-B"].value_counts()
    v_tea = df2[col + "-A"].value_counts()
    v_teb = df2[col + "-B"].value_counts()

    v_counts = v_tra.add(v_trb, fill_value=0)
    v_counts = v_counts.add(v_tea, fill_value=0)
    v_counts = v_counts.add(v_teb, fill_value=0)

    # threshold = v_counts.iloc[int(v_counts.shape[0]*0.8)]

    tra = df1[col + "-A"].unique()
    trb = df1[col + "-B"].unique()
    tea = df2[col + "-A"].unique()
    teb = df2[col + "-B"].unique()

    not_appeared = []
    print("fin count")
    for item in v_counts.index:
        if item not in tra or item not in trb or item not in tea or item not in teb or v_tra.loc[item] < 5 or v_trb.loc[
            item] < 5:
            not_appeared.append(item)

    print("fin find rare", len(not_appeared))
    df1[col + "-A"] = df1[col + "-A"].map(lambda x: "rare" if x in not_appeared else x)
    df1[col + "-B"] = df1[col + "-B"].map(lambda x: "rare" if x in not_appeared else x)
    df2[col + "-A"] = df2[col + "-A"].map(lambda x: "rare" if x in not_appeared else x)
    df2[col + "-B"] = df2[col + "-B"].map(lambda x: "rare" if x in not_appeared else x)

    return df1, df2
