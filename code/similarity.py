# sdfのパースを行い類似度を計算したのちグラフの描画をする

import re
import numpy as np
from typing import List, Tuple, Dict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

def spectrum_range(vector: List[Tuple[float, float]]) -> Tuple[float, float]:
    """スペクトルの m/z 範囲を返す"""
    if not vector:
        return (0.0, 0.0)
    mz_values = [mz for mz, _ in vector]
    return (min(mz_values), max(mz_values))

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
        'weighted_cosine_similarity': sim_weighted,
        'vec_meas': vec_simple_meas,
        'vec_pred': vec_simple_pred
    }

def plot_spectrum_comparison(
    sdf_path: str,
    record_index: int = 0,
    bin_width: float = 1.0,
    weighted_params: Tuple[float, float] = (1.0, 0.5),
    output_pdf: str = 'spectrum_comparison.pdf'
):
    """SDF ファイルからスペクトルを比較し、上下に伸びる stem プロットを PDF に保存"""
    # まずピークデータを直接取り出す
    with open(sdf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    records = content.split('$$$$')
    rec = records[record_index]
    meas_peaks = parse_peaks(extract_block(rec, 'MASS SPECTRAL PEAKS'))
    pred_peaks = parse_peaks(extract_block(rec, 'PREDICTED SPECTRUM'))

    # 最大 m/z (インデックス) を求める
    max_mz_meas = max((mz for mz, _ in meas_peaks), default=0.0)
    max_mz_pred = max((mz for mz, _ in pred_peaks), default=0.0)
    max_mz = max(max_mz_meas, max_mz_pred)
    max_idx = int(max_mz // bin_width) + 1

    # ベクトル化
    vec_meas = bin_peaks_to_vector(meas_peaks,  max_idx, bin_width, weighted=False)
    vec_pred = bin_peaks_to_vector(pred_peaks,  max_idx, bin_width, weighted=False)
    # 類似度だけ必要なら compute_similarities を呼び出してください

    sims = compute_similarities(
        sdf_path, record_index, mz_max=max_idx, bin_width=bin_width, weighted_params=weighted_params
    )

    cosine_similarity = sims['cosine_similarity']
    weighted_cosine_similarity = sims['weighted_cosine_similarity']

    # プロット
    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        # 実測スペクトルを上向き
        x = np.arange(max_idx) * bin_width
        ax.stem(x, 10*vec_meas, linefmt='C0-', markerfmt=' ', basefmt=" ", 
                label='True Mass Spectrum', use_line_collection=True)
        # 予測スペクトルを下向き（負の値にして描画）
        ax.stem(x, -1*vec_pred, linefmt='C1-', markerfmt=' ', basefmt=" ", 
                label='Predicted Mass Spectrum', use_line_collection=True)

        # 軸の調整
        ax.set_xlim(0, max_mz + bin_width)
        # y 軸は両方向にデータの最大から少し余裕を持たせる
        # ymax = max(vec_meas.max(), vec_pred.max())
        # ax.set_ylim(-1.1*ymax, 1.1*ymax)

        ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: f"{abs(y):.0f}")
    )

        ax.set_xlabel('mass/charge ratio')
        ax.set_ylabel('relative intensity')
        ax.set_title(f'Spectrum Comparison (Record {record_index})\n'
                     f'Cosine: {cosine_similarity:.4f} | Weighted: {weighted_cosine_similarity:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Spectrum comparison PDF saved as '{output_pdf}'")

# Example usage:
def example_usage():
    sdf_path = '../data/MoNA-GC-MS_filtered_ionization_output.sdf'
    plot_spectrum_comparison(sdf_path, record_index=0, output_pdf='spectrum_comparison_with_smiles.pdf')
    print("Spectrum comparison PDF created.")

if __name__ == "__main__":
    example_usage()