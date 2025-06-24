# spectrum_similarity.py

import re
import numpy as np
from typing import List, Tuple, Dict

def extract_block(record: str, tag: str) -> str:
    """SDF レコードから <tag>…</tag> のブロックを抜き出す"""
    pattern = rf'<{tag}>\s*(.*?)\s*(?=(<[^>]+>|$))'
    m = re.search(pattern, record, re.DOTALL)
    return m.group(1).strip() if m else ''

def parse_peaks(block: str) -> List[Tuple[float, float]]:
    """タグ内のテキストを (m/z, intensity) のリストにパースする"""
    peaks = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                peaks.append((mz, intensity))
            except ValueError:
                continue
    return peaks

def bin_peaks_to_vector(
    peaks: List[Tuple[float, float]],
    vector_size: int,
    bin_width: float = 1.0,
    weighted: bool = False,
    x: float = 1.0,
    y: float = 0.5
) -> np.ndarray:
    """
    (m/z, intensity) リストを固定長 numpy ベクトルにビニング。
    weighted=False : 単純強度加算
    weighted=True  : m/z^x * intensity^y を加算
    """
    vec = np.zeros(vector_size, dtype=float)
    for mz, intensity in peaks:
        idx = int(mz // bin_width)
        if 0 <= idx < vector_size:
            if weighted:
                vec[idx] += (mz**x) * (intensity**y)
            else:
                vec[idx] += intensity
    return vec

def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    normalize: bool = True
) -> float:
    """二つのベクトルのコサイン類似度を返す"""
    if normalize:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        vec1 = vec1 / norm1
        vec2 = vec2 / norm2
    return float(np.dot(vec1, vec2))

def compute_similarities(
    sdf_path: str,
    record_index: int = 0,
    mz_max: int = 2000,
    bin_width: float = 1.0,
    weighted_params: Tuple[float, float] = (1.0, 0.5)
) -> Dict[str, float]:
    """
    SDF ファイルを読み込み、指定レコードの実測／予測スペクトル間で
    - 単純コサイン類似度
    - 重み付きコサイン類似度
    を計算して返す。
    """
    # SDF を読み込み
    with open(sdf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    records = content.split('$$$$')

    # レコード抽出
    rec = records[record_index]
    meas_block = extract_block(rec, 'MASS SPECTRAL PEAKS')
    pred_block = extract_block(rec, 'PREDICTED SPECTRUM')

    # パース
    meas_peaks = parse_peaks(meas_block)
    pred_peaks = parse_peaks(pred_block)

    # ベクトル化
    vec_simple_meas = bin_peaks_to_vector(meas_peaks, mz_max, bin_width, weighted=False)
    vec_simple_pred = bin_peaks_to_vector(pred_peaks, mz_max, bin_width, weighted=False)
    x, y = weighted_params
    vec_w_meas = bin_peaks_to_vector(meas_peaks, mz_max, bin_width, weighted=True, x=x, y=y)
    vec_w_pred = bin_peaks_to_vector(pred_peaks, mz_max, bin_width, weighted=True, x=x, y=y)

    # 類似度計算
    sim_simple = cosine_similarity(vec_simple_meas, vec_simple_pred, normalize=True)
    sim_weighted = cosine_similarity(vec_w_meas, vec_w_pred, normalize=True)

    return {
        'cosine_similarity': sim_simple,
        'weighted_cosine_similarity': sim_weighted
    }
