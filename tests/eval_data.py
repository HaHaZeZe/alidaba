"""
eval_data.py — 数据加载与对齐
===========================
特征 CSV 合并 / 输出 CSV 加载 / 三表对齐 / 特征矩阵标准化
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eval_utils import detect_encoding


def load_all_features(features_dir: str) -> pd.DataFrame:
    """合并 data/features/ 下所有特征 CSV，返回全量特征矩阵"""
    files = [f for f in os.listdir(features_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"特征目录无 CSV 文件: {features_dir}")

    dfs = []
    for fname in sorted(files):
        fpath = os.path.join(features_dir, fname)
        enc = detect_encoding(fpath)
        df = pd.read_csv(fpath, encoding=enc)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  [加载] 特征数据: {len(df_all)} 条样本, {len(df_all.columns)} 列")
    return df_all


def load_outputs(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 pattern_reco.csv 和 predict_result.csv"""
    pattern_path = os.path.join(output_dir, 'pattern_reco.csv')
    predict_path = os.path.join(output_dir, 'predict_result.csv')

    df_pattern = pd.read_csv(pattern_path)
    df_predict = pd.read_csv(predict_path)

    print(f"  [加载] pattern_reco: {len(df_pattern)} 条")
    print(f"  [加载] predict_result: {len(df_predict)} 条")
    return df_pattern, df_predict


def align_data(df_features: pd.DataFrame,
               df_pattern: pd.DataFrame,
               df_predict: pd.DataFrame) -> pd.DataFrame:
    """
    将特征矩阵与两个输出 CSV 对齐。
    优先按 (stock_code, transaction_date) 全匹配；
    日期不一致时降级为仅按 stock_code 匹配。
    """
    key_cols = ['stock_code', 'transaction_date']

    df_features['transaction_date'] = df_features['transaction_date'].astype(str)
    df_pattern['transaction_date'] = df_pattern['transaction_date'].astype(str)
    df_predict['transaction_date'] = df_predict['transaction_date'].astype(str)

    df_merged = df_features.merge(
        df_pattern[key_cols + ['pattern_type']],
        on=key_cols, how='inner'
    )

    if len(df_merged) == 0:
        print("  [对齐] 日期不一致，降级为仅按 stock_code 匹配...")
        print(f"         特征日期: {sorted(df_features['transaction_date'].unique())}")
        print(f"         输出日期: {sorted(df_pattern['transaction_date'].unique())}")

        df_f = df_features.drop_duplicates('stock_code', keep='first').copy()
        df_pat = df_pattern.drop_duplicates('stock_code', keep='first').copy()
        df_pre = df_predict.drop_duplicates('stock_code', keep='first').copy()

        df_merged = df_f.merge(
            df_pat[['stock_code', 'pattern_type']],
            on='stock_code', how='inner'
        )
        df_merged = df_merged.merge(
            df_pre[['stock_code', 'capital_type', 'capital_intention']],
            on='stock_code', how='inner'
        )
    else:
        df_merged = df_merged.merge(
            df_predict[key_cols + ['capital_type', 'capital_intention']],
            on=key_cols, how='inner',
            suffixes=('', '_predict')
        )

    if len(df_merged) == 0:
        raise ValueError("特征数据与输出 CSV 无法对齐！请检查 stock_code 是否一致。")

    print(f"  [对齐] 匹配样本数: {len(df_merged)}")
    return df_merged


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """
    准备数值特征矩阵，排除非特征列，标准化。
    返回: (X_raw, X_scaled, feature_cols)
    """
    exclude = ['stock_code', 'transaction_date', 'cluster_id',
               'pattern_type', 'pattern_explanation',
               'capital_type', 'capital_intention']
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    X_raw = np.asarray(X.values, dtype=np.float64)
    X_scaled = StandardScaler().fit_transform(X_raw)

    return X_raw, X_scaled, feature_cols
