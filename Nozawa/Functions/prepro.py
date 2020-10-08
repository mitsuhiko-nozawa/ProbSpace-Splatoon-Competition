import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

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
            if scaler:
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
        cols = [col for col in df1.columns if "-"+col_name in col and team == col[0] and "onehot" in col]
        # print(cols)
        t_col = "team" + "-" + col_name + "-" + team
        df1[t_col] = ""
        df2[t_col] = ""


        for col in cols:
            df1[t_col] += df1[col].astype("str") + "-"
            df2[t_col] += df2[col].astype("str") + "-"

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

def target_encoding(df1, df2, y_, col, y_col="y", nfolds=5):
    df1 = df1.reset_index(drop=True)
    tgt_col = col+"-tgt-enc"
    random.seed(random.randint(0, 10000))
    SIZE = df1.shape[0]
    folds = make_kfolds(SIZE, nfolds)
    all_indices = sum(folds, [])
    if y_col == "y":
        df1[y_col] = y_.values
    df1_ = df1[[col, y_col]]


    contents = list(set(df1[col].unique().tolist() + df2[col].unique().tolist()))
    df1[tgt_col] = 0
    df2[tgt_col] = 0
    for i, fold in enumerate(folds):
        # print("fold {}".format(i))
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


def make_stratified_kfolds(X_, y_, K):
    skf = StratifiedKFold(n_splits=K, shuffle=True)
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
    col_name = col1 + " * " + col2
    df1[col_name] = df1[col1] + df1[col2]
    df2[col_name] = df2[col1] + df2[col2]

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