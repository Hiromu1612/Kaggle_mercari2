import pandas as pd
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.3f' % x)
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train_df = pd.read_csv('train.tsv', delimiter='\t')
test_df = pd.read_csv('test.tsv', delimiter='\t')

all_df = pd.concat([train_df.drop(["price"], axis=1), test_df],axis=0).reset_index(drop=True)
target = np.log1p(train_df["price"].values)
shape = train_df.shape[0]

#欠損値を処理
all_df["category_name"].fillna("NaN", inplace=True)
all_df["brand_name"].fillna("None", inplace=True)
all_df["item_description"].fillna("No description yet", inplace=True)

#ブランド上位300個のみ抽出、それ以外はNoneに変換
drop_brand_list = all_df["brand_name"].value_counts().index[300:]

def drop_brand(brand):
  if brand in drop_brand_list:
    return "None"
  else:
    return brand

all_df["brand_name"] = all_df["brand_name"].map(drop_brand)

print("欠損値処理完了")
#後にget_dummiesを利用するため、カテゴリー型に変換
all_df["brand_name"] = all_df["brand_name"].astype("category")
all_df["item_condition_id"] = all_df["item_condition_id"].astype("category")
all_df["shipping"] = all_df["shipping"].astype("category")

count_name = CountVectorizer(min_df=10)
X_name = count_name.fit_transform(all_df["name"])

tfidf_description = TfidfVectorizer(max_features = 200, stop_words = "english", ngram_range = (1,3))
X_description = tfidf_description.fit_transform(all_df["item_description"])

print("テキスト処理完了")


#各カテゴリーの末端を単語抽出
def pick_last_category(text):
  if text == "NaN":
    return "NaN"
  else:
    return str(text).split("/")[-1]

all_df["last_category_name"] = all_df["category_name"].map(pick_last_category)

#4項目をダミー変数化、疎行列変換
import scipy
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(all_df[["item_condition_id", "shipping", "last_category_name", "brand_name"]], sparse = True).values)

#データを結合し、trainとtestに分割
X = scipy.sparse.hstack((X_name , X_description, X_dummies)).tocsr()

train = X[:shape]
test = X[shape:]
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.3)

print("学習開始")
#モデル作成、学習の後、submitファイルを出力
model_Ridge = Ridge(alpha=4)
model_Ridge.fit(X_train, y_train)

print("学習完了")
test_pred = model_Ridge.predict(test)
sample_sub = pd.read_csv("sample_submission.csv")
sample_sub["price"] = np.expm1(test_pred)
sample_sub.to_csv("submission.csv", index=False)
