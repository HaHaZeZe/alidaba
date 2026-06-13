"""
eval_core.py — 离线评估主入口
===========================
基于赛题评分标准:
  Task1 (40%): 交易模式识别 — 聚类质量评估
  Task2 (60%): 资金类型与交易意图识别 — 加权 F1 Score
  A/B 榜总分 = Task1分 × 0.4 + Task2分 × 0.6

模块结构:
  eval_utils.py  — 编码检测 / DTW / 价格曲线 / 常量
  eval_data.py   — 数据加载 / 对齐 / 特征矩阵
  eval_task1.py  — 聚类评估 4 项指标 + 综合评分
  eval_task2.py  — 分类评估 F1 + 分布分析
  eval_core.py   — 主入口 run()

用法:
  from eval_core import run
  result = run(features_dir, output_dir, raw_dir)
"""
from typing import Optional

import numpy as np

from eval_data import load_all_features, load_outputs, align_data, prepare_feature_matrix
from eval_task1 import calc_silhouette, calc_ch_index, calc_wasserstein, calc_dtw, compute_task1_score
from eval_task2 import compute_task2_score


def run(features_dir: str,
        output_dir: str,
        raw_dir: str,
        ground_truth_path: Optional[str] = None) -> dict:
    """
    离线评估主入口。

    参数:
        features_dir: data/features/ 目录路径
        output_dir:   output/ 目录路径
        raw_dir:      data/raw/100stock/ 目录路径
        ground_truth_path: (可选) 真实标签 CSV 路径

    返回:
        dict: 包含所有评估指标的字典
    """
    print("=" * 60)
    print("  离线评估开始")
    print("=" * 60)

    # --- Step 1: 加载数据 ---
    print("\n[1/5] 加载数据...")
    df_features = load_all_features(features_dir)
    df_pattern, df_predict = load_outputs(output_dir)
    df = align_data(df_features, df_pattern, df_predict)

    # --- Step 2: 准备特征矩阵 ---
    print("\n[2/5] 准备特征矩阵...")
    _X_raw, X_scaled, feature_cols = prepare_feature_matrix(df)
    labels_raw = df['pattern_type'].values
    labels = np.asarray(labels_raw)
    unique_labels = np.unique(labels)
    print(f"  特征维度: {X_scaled.shape[1]} | 聚类数: {len(unique_labels)} | 样本数: {X_scaled.shape[0]}")
    counts = dict(zip(unique_labels, [int((labels == ul).sum()) for ul in unique_labels]))
    print(f"  各簇样本数: {counts}")

    # --- Step 3: Task1 评估 ---
    print("\n[3/5] Task1 — 聚类质量评估...")
    t1: dict = {}

    r_sil = calc_silhouette(X_scaled, labels)
    t1['silhouette'] = r_sil
    _print_metric("轮廓系数", r_sil, 'silhouette')

    r_ch = calc_ch_index(X_scaled, labels)
    t1['ch_index'] = r_ch
    _print_metric("CH指数", r_ch, 'ch_index')

    r_wass = calc_wasserstein(X_scaled, labels)
    t1['wasserstein'] = r_wass
    _print_metric("Wasserstein", r_wass, 'wasserstein')

    r_dtw = calc_dtw(df, raw_dir)
    t1['dtw'] = r_dtw
    _print_metric("DTW ratio", r_dtw, 'dtw_ratio')

    t1_score = compute_task1_score(t1)
    print(f"  => Task1 综合分: {t1_score['task1_score']:.2f} / 100")

    # --- Step 4: Task2 评估 ---
    print("\n[4/5] Task2 — 分类评估...")
    t2 = compute_task2_score(df, ground_truth_path)
    dist = t2['distribution']
    print(f"  资金类型分布: {dist['capital_distribution']}")
    print(f"  交易意图分布: {dist['intention_distribution']}")
    cap_ok = "✅" if dist['capital_type_合规'] else "❌"
    int_ok = "✅" if dist['capital_intention_合规'] else "❌"
    print(f"  合规校验: capital_type {cap_ok} | intention {int_ok}")

    if t2.get('f1'):
        f1r = t2['f1']
        print(f"  资金类型 F1: {f1r['capital_f1']:.2f}% "
              f"(P={f1r['capital_precision']:.2f}%, R={f1r['capital_recall']:.2f}%)")
        print(f"  交易意图 F1: {f1r['intention_f1']:.2f}% "
              f"(P={f1r['intention_precision']:.2f}%, R={f1r['intention_recall']:.2f}%)")
        ts = t2.get('task2_score')
        if ts is not None:
            print(f"  => Task2 综合分: {ts:.2f} / 100")
    else:
        note = t2.get('note', '无真实标签')
        print(f"  => Task2 综合分: N/A ({note})")

    # --- Step 5: 总分 ---
    print("\n[5/5] 汇总...")
    t1_s = float(t1_score['task1_score'])
    t2_s = float(t2.get('task2_score') or 0.0)
    overall = round(t1_s * 0.4 + t2_s * 0.6, 2)
    print(f"  A/B榜模拟总分 = {t1_s:.2f} × 0.4 + {t2_s:.2f} × 0.6 = {overall:.2f} / 100")

    print("\n" + "=" * 60)
    print("  评估完成")
    print("=" * 60)

    return {
        'task1': t1,
        'task1_score': t1_score,
        'task2': t2,
        'overall_score': overall,
        'n_samples': len(df),
        'n_features': X_scaled.shape[1],
        'n_clusters': len(unique_labels),
        'feature_cols': feature_cols,
    }


def _print_metric(name: str, result: dict, key: str) -> None:
    """统一打印指标结果"""
    val = result.get(key)
    if val is not None and not (isinstance(val, float) and np.isnan(val)):
        print(f"  {name}: {val:.4f}")
    else:
        reason = result.get('reason', '')
        print(f"  {name}: N/A ({reason})")
