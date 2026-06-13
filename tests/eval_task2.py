"""
eval_task2.py — Task2 分类评估
============================
预测分布分析 / 加权 F1 Score（需真值）/ 综合评分
"""
import os
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from eval_utils import VALID_CAPITAL_TYPES, VALID_INTENTIONS


def analyze_predictions(df: pd.DataFrame) -> dict:
    """分析预测分布（无真值时的降级方案）"""
    n = len(df)

    capital_dist = df['capital_type'].value_counts().to_dict()
    intention_dist = df['capital_intention'].value_counts().to_dict()

    invalid_capital = df[~df['capital_type'].isin(VALID_CAPITAL_TYPES)]
    invalid_intention = df[~df['capital_intention'].isin(VALID_INTENTIONS)]

    return {
        'n_samples': n,
        'capital_distribution': capital_dist,
        'intention_distribution': intention_dist,
        'capital_type_合规': len(invalid_capital) == 0,
        'capital_intention_合规': len(invalid_intention) == 0,
        'invalid_capital_count': int(len(invalid_capital)),
        'invalid_intention_count': int(len(invalid_intention)),
    }


def calc_f1_with_labels(df: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """
    计算加权 F1 Score。
    ground_truth 需包含: stock_code, transaction_date, true_capital_type, true_intention
    """
    key_cols = ['stock_code', 'transaction_date']
    df = df.copy()
    gt = ground_truth.copy()
    df['transaction_date'] = df['transaction_date'].astype(str)
    gt['transaction_date'] = gt['transaction_date'].astype(str)

    merged = df.merge(gt, on=key_cols, how='inner')
    if len(merged) == 0:
        return {'status': 'SKIP', 'reason': '真值标签无法匹配任何样本'}

    y_true_cap = merged['true_capital_type']
    y_pred_cap = merged['capital_type']
    cap_f1 = float(f1_score(y_true_cap, y_pred_cap, average='weighted', zero_division=0))
    cap_p = float(precision_score(y_true_cap, y_pred_cap, average='weighted', zero_division=0))
    cap_r = float(recall_score(y_true_cap, y_pred_cap, average='weighted', zero_division=0))

    y_true_int = merged['true_intention']
    y_pred_int = merged['capital_intention']
    int_f1 = float(f1_score(y_true_int, y_pred_int, average='weighted', zero_division=0))
    int_p = float(precision_score(y_true_int, y_pred_int, average='weighted', zero_division=0))
    int_r = float(recall_score(y_true_int, y_pred_int, average='weighted', zero_division=0))

    return {
        'capital_f1': round(cap_f1 * 100, 2),
        'capital_precision': round(cap_p * 100, 2),
        'capital_recall': round(cap_r * 100, 2),
        'intention_f1': round(int_f1 * 100, 2),
        'intention_precision': round(int_p * 100, 2),
        'intention_recall': round(int_r * 100, 2),
        'matched_samples': int(len(merged)),
    }


def compute_task2_score(df: pd.DataFrame,
                        ground_truth_path: Optional[str] = None) -> dict:
    """
    Task2 综合评分。
    有真值 → 加权 F1 × 100；无真值 → 分布分析 + 提示。
    """
    analysis = analyze_predictions(df)

    result: dict = {
        'distribution': analysis,
        'task2_score': None,  # type: ignore[dict-item]
    }

    f1_result = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt = pd.read_csv(ground_truth_path)
        f1_result = calc_f1_with_labels(df, gt)

    if f1_result and 'capital_f1' in f1_result:
        result['f1'] = f1_result
        cap_f1 = float(f1_result['capital_f1'])
        int_f1 = float(f1_result['intention_f1'])
        result['task2_score'] = round(cap_f1 * 0.7 + int_f1 * 0.3, 2)
    else:
        result['note'] = '无真实标签，Task2 分数无法计算。请提供 ground_truth CSV 以计算 F1。'

    return result
