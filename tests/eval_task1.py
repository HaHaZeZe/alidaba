"""
eval_task1.py — Task1 聚类质量评估
===============================
轮廓系数 / CH指数 / Wasserstein距离 / DTW时序距离 / 综合评分
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from eval_utils import N_INTERVALS, dtw_distance, load_price_curve


def calc_silhouette(X: np.ndarray, labels: np.ndarray) -> dict:
    """轮廓系数：衡量类内聚合度与类间区分度，[-1,1]"""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'silhouette': np.nan, 'status': 'SKIP', 'reason': '聚类数<2，无法计算'}
    score = float(silhouette_score(X, labels))
    return {
        'silhouette': score,
        'status': 'OK' if score > 0.2 else 'WARN',
        'benchmark': '>0.2 合格, >0.5 优秀'
    }


def calc_ch_index(X: np.ndarray, labels: np.ndarray) -> dict:
    """CH 指数：类间方差 / 类内方差，越高越好"""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'ch_index': np.nan, 'status': 'SKIP', 'reason': '聚类数<2，无法计算'}
    score = float(calinski_harabasz_score(X, labels))
    return {
        'ch_index': score,
        'status': 'OK' if score > 10 else 'WARN',
        'benchmark': '>10 合格, 越高越好'
    }


def calc_wasserstein(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Wasserstein 距离：逐特征维度计算簇间分布差异，取所有簇对的均值。
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return {'wasserstein': np.nan, 'status': 'SKIP', 'reason': '聚类数<2，无法计算'}

    n_features = X.shape[1]
    all_distances = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mask_i = labels == unique_labels[i]
            mask_j = labels == unique_labels[j]
            dim_dists = []
            for f in range(n_features):
                d = float(wasserstein_distance(X[mask_i, f], X[mask_j, f]))
                if np.isfinite(d):
                    dim_dists.append(d)
            if dim_dists:
                all_distances.append(float(np.mean(dim_dists)))

    if not all_distances:
        return {'wasserstein': np.nan, 'status': 'SKIP', 'reason': '无有效簇对'}

    mean_wass = float(np.mean(all_distances))
    return {
        'wasserstein': mean_wass,
        'wasserstein_std': float(np.std(all_distances)),
        'status': 'OK' if mean_wass > 0.1 else 'WARN',
        'benchmark': '>0.1 合格, 越大区分度越好'
    }


def calc_dtw(df: pd.DataFrame, raw_dir: str) -> dict:
    """
    DTW 时序距离：基于 5 分钟价格曲线，计算簇间/簇内 DTW 距离比。
    """
    unique_labels = np.asarray(df['pattern_type'].unique())
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return {'dtw_ratio': np.nan, 'status': 'SKIP', 'reason': '聚类数<2，无法计算'}

    # Step 1: 提取价格曲线
    print("  [DTW] 提取价格曲线...")
    curves: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        code = str(row['stock_code'])
        if code not in curves:
            curves[code] = load_price_curve(code, raw_dir)

    # 归一化到 [0,1]
    for code in list(curves.keys()):
        c = curves[code]
        c_min, c_max = c.min(), c.max()
        if c_max - c_min > 1e-8:
            curves[code] = (c - c_min) / (c_max - c_min)
        else:
            curves[code] = np.zeros(N_INTERVALS, dtype=np.float64)

    # Step 2: 簇内 DTW
    print("  [DTW] 计算簇内距离...")
    intra_dists: list[float] = []
    for label in unique_labels:
        cluster_codes = df[df['pattern_type'] == label]['stock_code'].tolist()
        if len(cluster_codes) < 2:
            continue
        dists: list[float] = []
        for i in range(len(cluster_codes)):
            for j in range(i + 1, len(cluster_codes)):
                ci = curves.get(str(cluster_codes[i]), np.zeros(N_INTERVALS))
                cj = curves.get(str(cluster_codes[j]), np.zeros(N_INTERVALS))
                if ci.sum() > 0 and cj.sum() > 0:
                    dists.append(dtw_distance(ci, cj))
        if dists:
            intra_dists.append(float(np.mean(dists)))
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 1.0

    # Step 3: 簇间 DTW
    print("  [DTW] 计算簇间距离...")
    inter_dists: list[float] = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            codes_i = df[df['pattern_type'] == unique_labels[i]]['stock_code'].tolist()
            codes_j = df[df['pattern_type'] == unique_labels[j]]['stock_code'].tolist()
            dists: list[float] = []
            for ci in codes_i:
                for cj in codes_j:
                    vi = curves.get(str(ci), np.zeros(N_INTERVALS))
                    vj = curves.get(str(cj), np.zeros(N_INTERVALS))
                    if vi.sum() > 0 and vj.sum() > 0:
                        dists.append(dtw_distance(vi, vj))
            if dists:
                inter_dists.append(float(np.mean(dists)))
    mean_inter = float(np.mean(inter_dists)) if inter_dists else 1.0

    dtw_ratio = mean_inter / (mean_intra + 1e-8)

    return {
        'dtw_ratio': dtw_ratio,
        'dtw_intra_mean': mean_intra,
        'dtw_inter_mean': mean_inter,
        'status': 'OK' if dtw_ratio > 1.5 else 'WARN',
        'benchmark': '>1.5 合格 (簇间/簇内), 越大区分度越好'
    }


def compute_task1_score(results: dict) -> dict:
    """
    Task1 综合评分（0-100）：四项指标各占 25%，归一化后加权。
    """
    metrics: dict[str, float] = {}
    weights: dict[str, float] = {}
    _nan = float('nan')

    # 轮廓系数: -1→0, 1→100
    sil = results.get('silhouette', {})
    s = sil.get('silhouette', _nan) if isinstance(sil, dict) else _nan
    if s is not None and not (isinstance(s, float) and np.isnan(s)):
        metrics['silhouette'] = max(0.0, min(100.0, (float(s) + 1.0) / 2.0 * 100.0))
        weights['silhouette'] = 0.25

    # CH 指数: log 尺度映射
    ch = results.get('ch_index', {})
    c = ch.get('ch_index', _nan) if isinstance(ch, dict) else _nan
    if c is not None and not (isinstance(c, float) and np.isnan(c)):
        metrics['ch_index'] = max(0.0, min(100.0, np.log1p(float(c)) / np.log1p(500) * 100.0))
        weights['ch_index'] = 0.25

    # Wasserstein: 标准化后在 0-3 之间
    wass = results.get('wasserstein', {})
    w = wass.get('wasserstein', _nan) if isinstance(wass, dict) else _nan
    if w is not None and not (isinstance(w, float) and np.isnan(w)):
        metrics['wasserstein'] = max(0.0, min(100.0, float(w) / 1.5 * 100.0))
        weights['wasserstein'] = 0.25

    # DTW ratio: >1 有效, >3 优秀
    dtw = results.get('dtw', {})
    d = dtw.get('dtw_ratio', _nan) if isinstance(dtw, dict) else _nan
    if d is not None and not (isinstance(d, float) and np.isnan(d)):
        metrics['dtw'] = max(0.0, min(100.0, (float(d) - 0.5) / 3.5 * 100.0))
        weights['dtw'] = 0.25

    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return {'task1_score': 0.0, 'task1_details': metrics}

    score = sum(metrics[k] * weights[k] / total_weight for k in metrics)
    return {
        'task1_score': round(score, 2),
        'task1_details': metrics
    }
