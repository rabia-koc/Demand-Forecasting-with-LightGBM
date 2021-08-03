#####################################################
# Demand Forecasting
#####################################################

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

########################
# Loading the data
########################

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

train = pd.read_csv('HAFTA_10/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('HAFTA_10/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('HAFTA_10/demand_forecasting/sample_submission.csv')

# Veri ön işleme işlemleri için bütün veriyi bir araya getirme işlemi:
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()

check_df(train)

check_df(test)

check_df(sample_sub)

check_df(df)


# Satış dağılımı incelemesi:
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Kaç eşsiz mağaza var?
df[["store"]].nunique()

# Kaç eşsiz ürün var?
df[["item"]].nunique()

# Her mağaza'da eşit sayıda mı eşsiz ürün var?
df.groupby(["store"])["item"].nunique()

# Her mağaza'da eşit sayıda mı satış var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# Mağaza-ürün kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

########################
# Date Features
########################

def create_date_features(df):
    df['month'] = df.date.dt.month  # hangi ayda
    df['day_of_month'] = df.date.dt.day  # ayın hangi gününde
    df['day_of_year'] = df.date.dt.dayofyear  # yılın hangi gününde
    df['week_of_year'] = df.date.dt.weekofyear  # yılın hangi haftasında
    df['day_of_week'] = df.date.dt.dayofweek    # haftanın hangi gününde
    df['year'] = df.date.dt.year              # hangi yılda
    df["is_wknd"] = df.date.dt.weekday // 4   # haftasonu mu değil mi
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)   # ayın başlangıcı mı
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)      # ayın bitişi mi

    df['quarter'] = df.date.dt.quarter
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.date.dt.is_quarter_end.astype(int)
    df['days_in_month'] = df.date.dt.daysinmonth  # bulunduğu ay kaç çekiyor

    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df.loc[df["month"]  == 12, "season"] = 0
    df.loc[df["month"]  == 1, "season"] = 0
    df.loc[df["month"]  == 2, "season"] = 0

    df.loc[df["month"]  == 3, "season"] = 1
    df.loc[df["month"]  == 4, "season"] = 1
    df.loc[df["month"]  == 5, "season"] = 1

    df.loc[df["month"]  == 6, "season"] = 2
    df.loc[df["month"]  == 7, "season"] = 2
    df.loc[df["month"]  == 8, "season"] = 2

    df.loc[df["month"]  == 9, "season"] = 3
    df.loc[df["month"]  == 10, "season"] = 3
    df.loc[df["month"]  == 11, "season"] = 3
    return df

df = create_date_features(df)

check_df(df)

# Mağaza-ürün-ay kırılımında satış istatistiklerini inceleme:
df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

########################
# Random Noise
########################

# Gürültü ekleme:
random_noise = np.random.normal(scale=1.6, size=(len(df),))

########################
# Lag/Shifted Features:
########################

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)

df["sales"].head(10)

# Birinci gecikme
df["sales"].shift(1).values[0:10]

# İkinci gecikme
df["sales"].shift(2).values[0:10]

# Üçüncü gecikme
df["sales"].shift(3).values[0:10]

# lags: gecikme bilgisi
def features_of_lags(dataframe, lags, target= "sales"):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])[target].transform(
            lambda x: x.shift(lag)) + random_noise
    return dataframe

# Test verisinde 3 aylık bir tahmin bekleniyor. Bu sebepten dolayı Lag featureları en az 91 olmalıdır.
df = features_of_lags(df, [91, 92, 95, 98, 105, 112, 119, 126, 182, 350, 363, 364, 370, 546, 550, 728])

check_df(df)

########################
# Rolling Mean Features:
########################

def features_of_roll_mean(dataframe, windows, target="sales"):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])[target]. \
                                                          transform(lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise
    return dataframe

df = features_of_roll_mean(df, [365, 546, 730])

df.tail()

########################
# Exponentially Weighted Mean Features:
########################

def features_of_ewm(dataframe, alphas, lags, target="sales"):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])[target].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = features_of_ewm(df, alphas, lags)
check_df(df)

########################
# One-Hot Encoding
########################

list =['store', 'item', 'day_of_week', 'month']

def one_hot_encoder(dataframe, cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=cols, drop_first=drop_first)
    return dataframe
one_hot_encoder(df,cols=list)

########################
# Converting sales to log(1+sales)
########################
# Bağımlı değişkeni dönüştürme işlemi:

df['sales'] = np.log1p(df["sales"].values)
check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# Kaggle verdiği fonksiyon
# smape'nin asıl noktası burası hesaplanması:
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# Tahmin edilen değerlerin gönderildiği kısım:
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################

# Kaggle'ın beklediği 2018 in ilk 3 ayı.

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Bağımsız değişkenleri seçme işlemi:
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

# Validasyon için de aynı işlemler:
Y_val = val['sales']
X_val = val[cols]

# Boyut incelemesi:
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,  # en sonda hatayı hesaplayacak fonk.
                  verbose_eval=1000)

# En iyi parametrelerle tahmin edilen değerler
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# Düzeltilmiş hata
smape(np.expm1(y_pred_val), np.expm1(Y_val))

########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# Bütün veri ile model kurma işlemi:
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

# Tahmin edilen değerler
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)
