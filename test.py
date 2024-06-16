import pandas as pd
import numpy as np

train = pd.read_csv("mercari-price-suggestion-challenge/train.tsv", delimiter='\t', low_memory=True)
test = pd.read_csv("mercari-price-suggestion-challenge/test.tsv", delimiter='\t', low_memory=True)

#trainデータ
train.name = train.name.astype("category")
train.category_name = train.category_name.astype("category")
train.brand_name = train.brand_name.astype("category")
train.item_description = train.item_description.astype("category")

#testデータ
test.name = test.name.astype("category")
test.category_name = test.category_name.astype("category")
test.brand_name = test.brand_name.astype("category")
test.item_description = test.item_description.astype("category")

train_test_combine = pd.concat([train.drop(["price"],axis=1), test], axis=0) #axis=0で行、axis=1で列に結合　デフォルトはaxis=0


train_test_combine.name = train_test_combine.name.astype("category")
train_test_combine.category_name = train_test_combine.category_name.astype("category")
train_test_combine.brand_name = train_test_combine.brand_name.astype("category")
train_test_combine.item_description = train_test_combine.item_description.astype("category")

train_test_combine.train_id = train_test_combine.train_id.fillna(pd.Series(train_test_combine.index))
train_test_combine.test_id = train_test_combine.test_id.fillna(pd.Series(train_test_combine.index))

train_test_combine.train_id = train_test_combine.train_id.astype(np.int64)
train_test_combine.test_id = train_test_combine.test_id.astype(np.int64)

train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes

print('前処理完了"')

df_train = train_test_combine.iloc[:train.shape[0],:]
df_test = train_test_combine.iloc[train.shape[0]:,:]

# #df_trainでtest_idを削除
# df_train = df_train.drop(["test_id"], axis=1)
# #df_testでtrain_idを削除
# df_test = df_test.drop(["train_id"], axis=1)

# df_test = df_test[["test_id"] + [col for col in df_test.columns if col != "test_id"]]

df_train["price"] = train.price
print("学習開始")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(["price"], axis=1), df_train.price, test_size=0.2, random_state=42)

# clf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# clf.fit(X_train, y_train)

# print(clf.score(X_train, y_train))

# 保存したモデルをロードする
import pickle
loaded_model = pickle.load(open(r"/Users/1612h/Kaggle_PG/model.pkl", 'rb'))
result = loaded_model.score(X_train, y_train)
print(result)

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
preds = loaded_model.predict(df_test)
# 予測値 predsをnp.exp()で処理
np.exp(preds)
# Numpy配列からpandasシリーズへ変換
preds = pd.Series(np.exp(preds))
# テストデータのIDと予測値を連結
submit = pd.concat([df_test.id, preds], axis=1)
# カラム名をメルカリの提出指定の名前をつける
submit.columns = ['test_id', 'price']
# 提出ファイルとしてCSVへ書き出し
submit.to_csv('submit_rf_base.csv', index=False)

#optunaを使う
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#lightgbmを使う
import lightgbm as lgb

def objective(trial):
    params={
        'n_estimators': trial.suggest_int('n_estimators', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 128),
    }
    X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(["price"], axis=1), df_train.price, test_size=0.2, random_state=42)
    lgb_train=lgb.Dataset(X_train, y_train)
    lgb_eval=lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    clf=lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=1000)
    y_pred_valid=clf.predict(X_valid, num_iteration=clf.best_iteration)
    score=mean_squared_error(y_valid, y_pred_valid)
    return score

#paramsに最適なパラメータを格納
study=optuna.create_study()
study.optimize(objective, n_trials=100)
params=study.best_params

#最適なパラメータで学習
X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(["price"], axis=1), df_train.price, test_size=0.2, random_state=42)
lgb_train=lgb.Dataset(X_train, y_train)
lgb_eval=lgb.Dataset(X_valid, y_valid, reference=lgb_train)
clf=lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=1000)

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float64:
    rmsle = mean_squared_error(np.log1p(y_true), np.log1p(y_pred))
    return np.sqrt(rmsle)

rmsle(y_valid,y_pred_valid)