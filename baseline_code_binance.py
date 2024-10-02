# -*- coding: utf-8 -*-
"""baseline_code_binance.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1od5dYMcW7gY0RqK4yO904o_X24uIptw2
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install optuna
# %pip install imblearn
# %pip install matplotlib
# %pip install lightgbm

"""### Library Import"""

import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
import optuna
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

"""### Data Load"""

# 파일 호출
data_path: str = "../data"
train_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "train.csv")).assign(_type="train") # train 에는 _type = train
test_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")).assign(_type="test") # test 에는 _type = test
submission_df: pd.DataFrame = pd.read_csv(os.path.join(data_path, "test.csv")) # ID, target 열만 가진 데이터 미리 호출
df_origin: pd.DataFrame = pd.concat([train_df, test_df], axis=0)

# HOURLY_ 로 시작하는 .csv 파일 이름을 file_names 에 할당
file_names: List[str] = [
    f for f in os.listdir(data_path) if f.startswith("HOURLY_NETWORK") and f.endswith(".csv")
]

additional_files = [
    "HOURLY_MARKET-DATA_COINBASE-PREMIUM-INDEX.csv",
    "HOURLY_MARKET-DATA_FUNDING-RATES_BINANCE.csv",
    "HOURLY_MARKET-DATA_LIQUIDATIONS_BINANCE_ALL_SYMBOL.csv",
    "HOURLY_MARKET-DATA_OPEN-INTEREST_BINANCE_ALL_SYMBOL.csv",
    "HOURLY_MARKET-DATA_TAKER-BUY-SELL-STATS_BINANCE.csv"
]

file_names.extend(additional_files)

# 파일명 : 데이터프레임으로 딕셔너리 형태로 저장
file_dict: Dict[str, pd.DataFrame] = {
    f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f)) for f in file_names
}

for _file_name, _df in tqdm(file_dict.items()):
    # 열 이름 중복 방지를 위해 {_file_name.lower()}_{col.lower()}로 변경, datetime 열을 ID로 변경
    _rename_rule = {
        col: f"{_file_name.lower()}_{col.lower()}" if col != "datetime" else "ID"
        for col in _df.columns
    }
    _df = _df.rename(_rename_rule, axis=1)
    df_origin = df_origin.merge(_df, on="ID", how="left")

"""### Feature engineering"""

# 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
cols_dict: Dict[str, str] = {
    "ID": "ID",
    "target": "target",
    "_type": "_type",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
    "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
    "hourly_market-data_funding-rates_binance_funding_rates": "funding_rates",
    "hourly_market-data_liquidations_binance_all_symbol_long_liquidations": "long_liquidations",
    "hourly_market-data_liquidations_binance_all_symbol_long_liquidations_usd": "long_liquidations_usd",
    "hourly_market-data_liquidations_binance_all_symbol_short_liquidations": "short_liquidations",
    "hourly_market-data_liquidations_binance_all_symbol_short_liquidations_usd": "short_liquidations_usd",
    "hourly_market-data_open-interest_binance_all_symbol_open_interest": "open_interest",
    "hourly_market-data_taker-buy-sell-stats_binance_taker_buy_ratio": "buy_ratio",
    "hourly_market-data_taker-buy-sell-stats_binance_taker_buy_volume": "buy_volume",
    "hourly_market-data_taker-buy-sell-stats_binance_taker_sell_ratio": "sell_ratio",
    "hourly_market-data_taker-buy-sell-stats_binance_taker_sell_volume": "sell_volume",
    "hourly_network-data_supply_supply_total": "supply_total",
    "hourly_network-data_blockreward_blockreward": "blockreward",
    "hourly_network-data_blockreward_blockreward_usd": "blockreward_usd",
}
df = df_origin[cols_dict.keys()].rename(cols_dict, axis=1)
df.shape

# 새로운 feature를 생성하고 기존의 feature 제거 or 실험 결과로 제외할 feature를 할당
items_to_check = ["supply_total", "blockreward",  "coinbase_premium_index", "funding_rates", "open_interest"]

#기존에 존재하거나 새로 만들었지만 제외한것
'''
"buy_ratio", "sell_ratio","long_liquidations", "short_liquidations", "long_liquidations_usd", "short_liquidations_usd",
liquidation_diff=df["long_liquidations"] - df["short_liquidations"],
liquidation_usd_diff=df["long_liquidations_usd"] - df["short_liquidations_usd"],
volume_imbalance=(df["buy_volume"] - df["sell_volume"])/(df["buy_volume"] + df["sell_volume"]+ 1e-8),
'''
# eda 에서 파악한 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성
df_train = df.assign(
    liquidation_usd_imbalance=(df['long_liquidations_usd'] - df['short_liquidations_usd']) / (df['long_liquidations_usd'] + df['short_liquidations_usd'] + 1e-8),
    liquidation_imbalance=(df['long_liquidations'] - df['short_liquidations']) / (df['long_liquidations'] + df['short_liquidations'] + 1e-8),
)

df_train = df_train.drop(items_to_check, axis=1)

# category, continuous 열을 따로 할당해둠
# 증강 및 표준화 등의 전처리를 할 때 구분을 위해 사용
#category_cols: List[str] = ["liquidation_diffg", "liquidation_usd_diffg", "volume_diffg"]
conti_cols: List[str] = [_ for _ in cols_dict.values() if _ not in (["ID", "target", "_type"] + items_to_check)] + [
    "liquidation_usd_imbalance",
    "liquidation_imbalance",
]

# Create rolling averages for all continuous columns
for col in conti_cols:
    df_train[f'{col}_rolling_5'] = df_train[col].rolling(window=5).mean()
    '''df_train[f'{col}_rolling_10'] = df_train[col].rolling(window=10).mean()
    df_train[f'{col}_rolling_20'] = df_train[col].rolling(window=20).mean()
    df_train[f'{col}_rolling_60'] = df_train[col].rolling(window=60).mean()
    df_train[f'{col}_rolling_100'] = df_train[col].rolling(window=120).mean()
    #df_train[f'{col}_rolling_200'] = df_train[col].rolling(window=200).mean()'''

def shift_feature(
    df_train: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> List[pd.Series]:
    """
    연속형 변수의 shift feature 생성
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        List[pd.Series]
    """
    df_shift_dict = [
        df_train[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
        for conti_col in conti_cols
        for interval in intervals
    ]
    return df_shift_dict

# shift된 피처를 conti_cols에 추가
def update_conti_cols_with_shifts(conti_cols: List[str], intervals: List[int]) -> List[str]:
    shifted_cols = [f"{conti_col}_{interval}" for conti_col in conti_cols for interval in intervals]
    return conti_cols + shifted_cols

# 최대 8시간의 shift 피쳐를 계산
shift_list = shift_feature(
    df_train=df_train, conti_cols=conti_cols, intervals=[_ for _ in range(1, 9)]
)

# conti_cols에 shift된 피처들의 이름 추가
conti_cols = update_conti_cols_with_shifts(conti_cols, intervals=[_ for _ in range(1, 9)])

print(f"Updated conti_cols: {conti_cols}")

# concat 하여 df 에 할당
df_train = pd.concat([df_train, pd.concat(shift_list, axis=1)], axis=1)

# 타겟 변수를 제외한 변수를 forwardfill, -999로 결측치 대체
_target = df_train["target"]
df_train = df_train.ffill().fillna(-999).assign(target = _target)

# _type에 따라 train, test 분리
train_df = df_train.loc[df["_type"]=="train"].drop(columns=["_type"])
test_df = df_train.loc[df["_type"]=="test"].drop(columns=["_type"])

train_df

"""## Model Training

### data set, params 선언
"""

x_train = train_df.drop(["target", "ID"], axis=1)
y_train = train_df["target"].astype(int)

x_train.shape

"""### lgb"""

params = {
    "boosting_type": "dart",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 4,
    "learning_rate": 0.005,
    "min_data_in_leaf": 70,
    "random_state": 42,
    "n_estimators": 500,
    "num_leaves": 30,
}

# 증강을 사용할 경우 성능 저하
# RobustScaler를 이용한 정규화를 사용할 경우 성능 향상
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

# KFold 설정
n_splits = 5  # 폴드 수 설정
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 전체 예측값을 담을 리스트
y_valid_preds = np.zeros((x_train.shape[0], 4))  # 전체 데이터의 예측값 저장 (클래스 개수는 4)

# 모델들을 저장할 리스트 (각 폴드의 모델을 저장)
models = []

# 정규화 스케일러 초기화
scaler = RobustScaler()

# 10 epoch마다 진행도를 출력하는 콜백 함수
def print_evaluation(period=10):
    def callback(env):
        if (env.iteration + 1) % period == 0:
            print(f'Iteration {env.iteration + 1}, Train Logloss: {env.evaluation_result_list[0][2]:.4f}, Valid Logloss: {env.evaluation_result_list[1][2]:.4f}')
    return callback
# 교차 검증
for fold, (train_idx, valid_idx) in enumerate(kf.split(x_train, y_train)):
    print(f"Training fold {fold+1}/{n_splits}")

    # 훈련 세트와 검증 세트 나누기
    X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
    Y_tr, Y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    # 정규화 적용 (훈련 데이터와 검증 데이터에 RobustScaler 적용)
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # lgb dataset
    train_data = lgb.Dataset(X_tr, label=Y_tr)
    valid_data = lgb.Dataset(X_val, label=Y_val, reference=train_data)

    # lgb train
    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        callbacks=[print_evaluation(10)]  # 수정된 콜백 함수 사용
    )


    # 모델 저장
    models.append(lgb_model)

    # 검증 세트 예측 (각 폴드마다 예측 결과를 저장)
    y_valid_preds[valid_idx] = lgb_model.predict(X_val)

# 예측값의 클래스 결정 (argmax로 가장 높은 확률의 클래스를 선택)
y_valid_pred_class = np.argmax(y_valid_preds, axis=1)

# score check
accuracy = accuracy_score(y_train, y_valid_pred_class)  # 전체 y_train과 비교
auroc = roc_auc_score(y_train, y_valid_preds, multi_class="ovr")  # 전체 y_train과 비교

print(f"Final Accuracy: {accuracy}, AUROC: {auroc}")

"""### Output File Save"""

# lgb predict
y_test_pred = np.zeros((test_df.shape[0], 4))  # 4는 클래스 개수 (multiclass일 경우)

# 각 폴드의 모델을 사용하여 예측
for model in models:
    y_test_pred += model.predict(test_df.drop(["target", "ID"], axis=1)) / n_splits

# 예측값의 클래스 결정 (argmax로 가장 높은 확률의 클래스를 선택)
y_test_pred_class = np.argmax(y_test_pred, axis=1)

# output file 할당후 save
submission_df = submission_df.assign(target=y_test_pred_class)
submission_df.to_csv("output.csv", index=False)

submission_df["target"].value_counts()/len(submission_df)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 실제 값과 예측 값을 사용하여 혼동 행렬 생성
cm = confusion_matrix(y_train, y_valid_pred_class)

# 혼동 행렬 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# 그래프 제목 추가
plt.title('Confusion Matrix')
plt.show()

from lightgbm import plot_importance
import numpy as np

# 훈련 시 사용한 feature 이름 리스트
train_feature_names = x_train.columns.tolist()

# 각 모델에 대해 feature importance 플롯 생성
for i, model in enumerate(models):
    # 특성 중요도 추출
    importance = model.feature_importance(importance_type='gain')

    # 중요도와 feature 이름을 딕셔너리로 매핑
    feature_importance = dict(zip(train_feature_names, importance))

    # 중요도 순으로 정렬 (내림차순)
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_feature_names, sorted_importance = zip(*sorted_feature_importance[:10])  # 상위 10개만 선택

    # 그림 크기를 설정 (가로 20인치, 세로는 특성 수에 따라 조정)
    fig, ax = plt.subplots(figsize=(20, len(sorted_feature_names) * 0.3))

    # 수평 막대 그래프 생성
    y_pos = np.arange(len(sorted_feature_names))
    ax.barh(y_pos, sorted_importance, align='center')

    # y축 레이블 설정 (특성 이름)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feature_names)

    # x축 레이블과 제목 설정
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance (Gain) - Model {i+1}')

    # 그리드 추가
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # 각 막대에 수치 표시
    for i, v in enumerate(sorted_importance):
        ax.text(v, i, f' {v:.4f}', va='center')

    # x축 범위 설정 (여백 추가)
    ax.set_xlim(0, max(sorted_importance) * 1.1)

    # y축 반전 (가장 중요한 특성이 위에 오도록)
    ax.invert_yaxis()

    # 레이블이 잘리지 않도록 여백 조정
    plt.tight_layout()

    # 그래프 표시
    plt.show()