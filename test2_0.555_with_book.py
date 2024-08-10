import pandas as pd
import numpy as np

train = pd.read_csv("train.tsv", delimiter='\t', low_memory=True)
test = pd.read_csv("test.tsv", delimiter='\t', low_memory=True)

train = train.drop(train[(train.price < 3.0)].index)

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label") #categoryがない場合はNo Labelを返す

train['general_cat'], train['sub_cat1'], train['sub_cat2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
test['general_cat'], test['sub_cat1'], test['sub_cat2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))

#general_cat, sub_cat1, sub_cat2をカテゴリ変数に変換
train['general_cat'] = train['general_cat'].astype('category')
train['sub_cat1'] = train['sub_cat1'].astype('category')
train['sub_cat2'] = train['sub_cat2'].astype('category')

#train,testを縦方向に結合し、まとめて前処理できるようにする
train_test_combine = pd.concat([train,test]) #axis=0で行、axis=1で列に結合 デフォルトはaxis=0

#brand_nameの重複なしのリストを作成
brand_name_list = set(train_test_combine["brand_name"].values) #set()で重複を削除

# 'brand_name'の欠損値NaNを'missing'に置き換える
train['brand_name'].fillna(value='missing', inplace=True)
test['brand_name'].fillna(value='missing', inplace=True)

# 訓練データの'brand_name'が'missing'に一致するレコード数を取得
train_premissing = len(train.loc[train['brand_name'] == 'missing'])
# テストデータの'brand_name'が'missing'に一致するレコード数を取得
test_premissing = len(test.loc[test['brand_name'] == 'missing'])

print(train_premissing, test_premissing)

def brandfinder(line):
    brand = line[0] # 第1要素はブランド名
    name = line[1]  # 第2要素は商品名
    namesplit = name.split(' ') # 商品名をスペースで切り分ける
    
    if brand == 'missing':  # ブランド名が'missing'の場合
        for x in namesplit: # 商品名から切り分けた単語を取り出す
            if x in brand_name_list: # 単語がブランドリストに存在すればブランド名を返す              
                return name # 単語がブランドリストに一致したら商品名を返す
    if name in brand_name_list:  # 商品名がブランドリストに存在すれば商品名を返す
        return name
    
    return brand            # どれにも一致しなければブランド名を返す

# ブランド名の付替えを実施
train['brand_name'] = train[['brand_name','name']].apply(brandfinder, axis = 1) #axis=1で行方向
test['brand_name'] = test[['brand_name','name']].apply(brandfinder, axis = 1)

# 書き換えられた'missing'の数を取得
train_found = train_premissing-len(train.loc[train['brand_name'] == 'missing'])
test_found = test_premissing-len(test.loc[test['brand_name'] == 'missing'])


print("エンコーディング開始")
# train,testをそれぞれ変換するのは面倒なので、train_test_combineで一括変換
train_test_combine = pd.concat([train, test])

#fillnaで欠損値を埋める
train_test_combine['name'].fillna(value='missing', inplace=True)
train_test_combine['category_name'].fillna(value='missing', inplace=True)
train_test_combine['general_cat'].fillna(value='missing', inplace=True)
train_test_combine['sub_cat1'].fillna(value='missing', inplace=True)
train_test_combine['sub_cat2'].fillna(value='missing', inplace=True)
train_test_combine['brand_name'].fillna(value='missing', inplace=True)
train_test_combine['item_description'].fillna(value='missing', inplace=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#カテゴリ変数を数値に変換
train_test_combine['category_name'] = le.fit_transform(train_test_combine['category_name'])
train_test_combine['general_cat'] = le.fit_transform(train_test_combine['general_cat'])
train_test_combine['sub_cat1'] = le.fit_transform(train_test_combine['sub_cat1'])
train_test_combine['sub_cat2'] = le.fit_transform(train_test_combine['sub_cat2'])
train_test_combine['brand_name'] = le.fit_transform(train_test_combine['brand_name'])


print("エンコーディング開始(自然言語処理)")
from tensorflow.keras.preprocessing.text import Tokenizer

print("Transforming text data to sequences...")
raw_text = np.hstack(
    [train_test_combine.item_description.str.lower(), # 説明文
     train_test_combine.name.str.lower()]           # 商品名
)
print('sequences shape', raw_text.shape)

# 説明文、商品名、カテゴリ名を連結した配列でTokenizerを作る
print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

# Tokenizerで説明文、商品名をそれぞれラベルエンコードする
print("   Transforming text to sequences...")
train_test_combine['seq_item_description'] = tok_raw.texts_to_sequences(train_test_combine.item_description.str.lower())
train_test_combine['seq_name'] = tok_raw.texts_to_sequences(train_test_combine.name.str.lower())

del tok_raw #delは変数を削除する

print("0でパディング")
from keras.preprocessing.sequence import pad_sequences
print(pad_sequences(train_test_combine.seq_item_description, maxlen=80),'\n') # 商品説明
print(pad_sequences(train_test_combine.seq_name, maxlen=10))  


def wordCount(text):
    """
    Parameters:
      text(str): 商品名、商品の説明文
    """
    try:
        if text == 'No description yet':
            return 0  # 商品名や説明が'No description yet'の場合は0を返す
        else:
            text = text.lower()                  # すべて小文字にする
            words = [w for w in text.split(" ")] # スペースで切り分ける
            return len(words)                    # 単語の数を返す
    except: 
        return 0

# 'name'の各フィールドの単語数を'name_len'に登録
train_test_combine['name_len'] = train_test_combine['name'].apply(lambda x: wordCount(x))
# 'item_description'の各フィールドの単語数を'desc_len'に登録
train_test_combine['desc_len'] = train_test_combine['item_description'].apply(lambda x: wordCount(x))

# 文字列(object)カテゴリー変数に変換
train_test_combine['name'] = train_test_combine['name'].astype('category')
train_test_combine['item_description'] = train_test_combine['item_description'].astype('category')

# 文字列(object)カテゴリー変数に変換
train_test_combine['name'] = train_test_combine['name'].astype('category')
train_test_combine['item_description'] = train_test_combine['item_description'].astype('category')

# リストを文字列に変換してからカテゴリー変数に変換
train_test_combine['seq_item_description'] = train_test_combine['seq_item_description'].apply(lambda x: str(x)).astype('category')
train_test_combine['seq_name'] = train_test_combine['seq_name'].apply(lambda x: str(x)).astype('category')

print(train_test_combine.dtypes)

# 訓練データの'price'を対数変換する
target = np.log1p(train.price)

#train_test_combineからtrainとtestに分割
train = train_test_combine[:len(train)]
test = train_test_combine[len(train):]
print("前処理終了")


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.3)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Ridge

# 文字列データを数値データに変換
label_encoders = {}
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column].astype(str))
        X_val[column] = le.transform(X_val[column].astype(str))
        label_encoders[column] = le

# NaN値を平均値で埋める
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

## RMSLEの評価関数
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5

# モデルの学習 4分かかる
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
model.fit(X_train_imputed, y_train)

# RMSLE関数の定義
from sklearn.metrics import mean_squared_log_error
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# モデルの評価
y_pred = model.predict(X_val_imputed)
print("RMSLE: ", rmsle(np.expm1(y_val), np.expm1(y_pred)))