import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import sklearn
from Functions import prepro
import warnings
warnings.filterwarnings('ignore')

import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
from catboost import Pool, CatBoostClassifier, CatBoost
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(random.randint(0, 10000))

train_df = pd.read_csv("../../data/Processed/train2.csv")
test_df = pd.read_csv("../../data/Processed/test2.csv")
print(train_df.shape)
print(test_df.shape)

y = train_df["y"].values
train_df = train_df.drop("y", axis=1)

train_df = prepro.add_disconnection(train_df)
test_df = prepro.add_disconnection(test_df)

num_cols = [
    "level", "range-main", "range-bullet-main", "distant-range_sub",
    "rapid", "atack", "ink-sub", "fav-main", "good-special", "DPS", "kill_time_ika-main",
    "front_gap_human-main", "rensya_frame-main", "saidai_damege-main", "damage_min-sub",
    "damage_max-sub", "install_num-sub", "good-sub", "damage_max-special",
    "duration-special", "good-special", "direct_rad-special", "distant_rad-special"
]

cols = [col for col in train_df.columns if "A1" in col or "A2" in col or "A3" in col or "A4" in col or
        "B1" in col or "B2" in col or "B3" in col or "B4" in col]
drop_cols = []
for col1 in cols:
    f = True
    for col2 in num_cols:
        if col2 in col1:
            f = False
    if f and train_df[col1].dtype in [int, float]:
        drop_cols.append(col1)

train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

train_df = prepro.add_numeric_info(train_df, num_cols)
test_df = prepro.add_numeric_info(test_df, num_cols)

# rankの欠損値を埋める
train_df, test_df = prepro.fillna_rank(train_df, test_df)

#そのほかの欠損値を埋める
train_df, test_df = prepro.fillna(train_df, test_df)

print(train_df.isnull().sum().sum())
print(test_df.isnull().sum().sum())

# count reskin, by mode
print("reskin")
train_df, test_df = prepro.count_reskin(train_df, test_df)
train_df, test_df = prepro.count_reskin_by_mode(train_df, test_df)

# count mainweapon, by mode
print("mainweapon")
train_df, test_df = prepro.count_mainweapon(train_df, test_df)
train_df, test_df = prepro.count_mainweapon_by_mode(train_df, test_df)

# count subweapon, by mode
print("subweapon")
train_df, test_df = prepro.count_subweapon(train_df, test_df)
train_df, test_df = prepro.count_subweapon_by_mode(train_df, test_df)

# count special, by mode
print("special")
train_df, test_df = prepro.count_special(train_df, test_df)
train_df, test_df = prepro.count_special_by_mode(train_df, test_df)


#identify A1
train_df, test_df = prepro.identify_A1(train_df, test_df)


# 水増し, A1も統計量に含めた特徴を作る場合は水ましより先にやる
print("mizumashi")
train_df, y = prepro.mizumashi(train_df, y)

# is_nawabari
train_df, test_df = prepro.is_nawabari(train_df, test_df)

# match rank、単体で意味なし
train_df, test_df = prepro.match_rank(train_df, test_df)

# rankを二列に分ける
train_df, test_df = prepro.ranker(train_df, test_df)


# add team info、メインはなくてもいい
train_df,  test_df = prepro.addTeamInfo(train_df, test_df, cols=["special", "subweapon", "category1", "category2", "mainweapon", "rank-mark"])

# categorize team , 良い, メインはカテゴリ数が多すぎてやめた方がいい

categorize_col = ["category1", "category2", "subweapon", "special", "mainweapon"]
#categorize_col = ["category1", "category2", "subweapon", "special"]
for col in categorize_col:
    print(col)
    train_df, test_df = prepro.categorize_team(train_df, test_df, col)


# product categorical feature
train_df, test_df = prepro.prod(train_df, test_df, "mode", "stage")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-category1-A")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-category1-B")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-category2-A")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-category2-B")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-mainweapon-A")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-mainweapon-B")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-subweapon-A")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-subweapon-B")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-special-A")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "team-special-B")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "match_rank")

train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-category1-A")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-category1-B")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-category2-A")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-category2-B")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-mainweapon-A")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-mainweapon-B")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-subweapon-A")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-subweapon-B")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-special-A")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "team-special-B")
train_df, test_df = prepro.prod(train_df, test_df, "stage", "match_rank")


train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-A1")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-A2")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-A3")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-A4")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-B1")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-B2")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-B3")
train_df, test_df = prepro.prod(train_df, test_df, "mode", "reskin-B4")

# make input

drop_cols = [
    "id", "lobby", "lobby-mode",  "period", "game-ver", "A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon",
    "B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon", "A-a-rank-mark-onehot", "A-o-rank-mark-onehot",
    "B-a-rank-mark-onehot", "B-o-rank-mark-onehot",
    "reskin-A1", "reskin-A2", "reskin-A3", "reskin-A4",
    "reskin-B1", "reskin-B2", "reskin-B3", "reskin-B4",
]
X, test_X = prepro.make_input(train_df, test_df, drop_cols, categorical_encode=False, scaler=False, verbose=False)

categorical_features_indices = np.where(X.dtypes == "object")[0]

for iter in range(31, 101):
    print("iteration {}".format(iter))
    random.seed(random.randint(0, 10000))
    SIZE = X.shape[0]
    K = 5

    # folds = prepro.make_stratified_kfolds(X, y, K, shuffle=True)
    folds = prepro.make_stratified_kfolds(X, X["mode"].astype(str) + y.astype(str), K, shuffle=True,
                                          random_state=random.randint(0, 10000))


    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.05,
        "use_best_model": True,
        "random_seed": random.randint(0, 100000),
        'task_type': 'GPU',

    }

    THRESHOLD = 0.50
    models = []
    cv_scores = []
    temp = 0
    train_pred = []
    valid_ys = []

    all_indices = sum(folds, [])
    for i in range(K):
        print("======================== fold {} ========================".format(i + 1))
        valid_indices = folds[i]
        train_indices = list(set(all_indices) - set(valid_indices))
        # print("train ", len(train_indices), " , valid ", len(valid_indices))

        train_X = X.iloc[train_indices]
        train_y = y[train_indices]
        valid_X = X.iloc[valid_indices]
        valid_y = y[valid_indices]

        train_data = Pool(train_X, train_y, cat_features=categorical_features_indices)
        valid_data = Pool(valid_X, valid_y, cat_features=categorical_features_indices)

        model = CatBoostClassifier(**params)

        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=50,
            verbose=100,
            use_best_model=True,
        )

        pred = model.predict(valid_X, prediction_type='Probability')[:, 1]
        train_pred.append(pred)
        pred = np.where(pred < THRESHOLD, 0, 1)

        temp += np.sum(pred)

        score = accuracy_score(pred, valid_y)

        models.append(model)
        cv_scores.append(score)

    print("cv score : ", np.mean(cv_scores))
    print("cv ratio : ", temp / SIZE)
    preds = []
    for i in range(K):
        model = models[i]
        pred = model.predict(test_X, prediction_type='Probability')[:, 1]
        preds.append(pred)
        print(np.sum(pred) / pred.shape[0])

    preds = np.array(preds)
    preds = np.mean(preds, axis=0)  #
    print(np.sum(preds) / preds.shape[0])


    train_df["pred"] = 0
    for i in range(K):
        train_df["pred"].iloc[folds[i]] = train_pred[i]

    train_fe_df = pd.DataFrame({'cat_feature_'+str(iter): train_df["pred"].values})
    train_fe_df.index.name = 'id'
    train_fe_df.to_csv('Train/train_cat_feature_{}.csv'.format(iter), index=False)
    test_fe_df = pd.DataFrame({'cat_feature_'+str(iter): preds})
    test_fe_df.index.name = 'id'
    test_fe_df.to_csv('Test/test_cat_feature_{}.csv'.format(iter), index=False)














