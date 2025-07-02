import re
import numpy as np
from typing import List, Tuple
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import textwrap

# 追加: COMMENT から SMILES/InChI を抜き出す正規表現
SMILES_RE = re.compile(r"\bsmiles\s*=\s*([^\s;]+)", re.IGNORECASE)
INCHI_RE  = re.compile(r"\bInChI\s*=\s*(InChI=[^\s;]+)", re.IGNORECASE)

def extract_block(record: str, tag: str) -> str:
    pattern = rf'<{tag}>\s*(.*?)\s*(?=(<[^>]+>|$))'
    m = re.search(pattern, record, re.DOTALL)
    return m.group(1).strip() if m else ''

def parse_peaks(block: str) -> List[Tuple[float, float]]:
    peaks = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                mz = float(parts[0]); intensity = float(parts[1])
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
    vec = np.zeros(vector_size, dtype=float)
    for mz, intensity in peaks:
        idx = int(mz // bin_width)
        if 0 <= idx < vector_size:
            vec[idx] += (mz**x)*(intensity**y) if weighted else intensity
    return vec

def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    normalize: bool = True
) -> float:
    if normalize:
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1==0 or n2==0: return 0.0
        vec1, vec2 = vec1/n1, vec2/n2
    return float(np.dot(vec1, vec2))

def compute_similarity_vectors(
    meas_peaks, pred_peaks,
    bin_width: float = 1.0,
    weighted_params: Tuple[float, float] = (1.0, 0.5)
):
    max_mz = max((mz for mz,_ in meas_peaks+pred_peaks), default=0.0)
    size = int(max_mz//bin_width) + 1
    vec_simple_meas = bin_peaks_to_vector(meas_peaks, size, bin_width, False)
    vec_simple_pred = bin_peaks_to_vector(pred_peaks, size, bin_width, False)
    x, y = weighted_params
    vec_w_meas = bin_peaks_to_vector(meas_peaks, size, bin_width, True, x, y)
    vec_w_pred = bin_peaks_to_vector(pred_peaks, size, bin_width, True, x, y)
    return (
        vec_simple_meas,
        vec_simple_pred,
        cosine_similarity(vec_simple_meas, vec_simple_pred),
        cosine_similarity(vec_w_meas,    vec_w_pred)
    )

def plot_all_spectra(
    sdf_path: str,
    output_pdf: str = 'spectrum_comparison_all.pdf',
    bin_width: float = 1.0,
    weighted_params: Tuple[float, float] = (1.0, 0.5)
):
    # SDF全レコード読み込み
    with open(sdf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    records = [r for r in content.split('$$$$') if r.strip()]

    with PdfPages(output_pdf) as pdf:
        for idx, rec in enumerate(records, start=1):
            # 実測 vs 予測ピーク
            meas = parse_peaks(extract_block(rec, 'MASS SPECTRAL PEAKS'))
            pred = parse_peaks(extract_block(rec, 'PREDICTED SPECTRUM'))

            # COMMENT から SMILES/InChI を抽出
            comment = extract_block(rec, 'COMMENT')
            smi_match  = SMILES_RE.search(comment)
            inchi_match= INCHI_RE.search(comment)
            smiles = smi_match.group(1)    if smi_match   else 'N/A'
            inchi  = inchi_match.group(1)  if inchi_match else 'N/A'

            vec_m, vec_p, sim_s, sim_w = compute_similarity_vectors(
                meas, pred, bin_width, weighted_params
            )

            size   = len(vec_m)
            x_axis = np.arange(size) * bin_width

            fig, ax = plt.subplots(figsize=(10,8))
            ax.stem(x_axis, 10*vec_m, linefmt='C0-', markerfmt=' ', basefmt=' ', use_line_collection=True)
            ax.stem(x_axis, -1*vec_p, linefmt='C1-', markerfmt=' ', basefmt=' ', use_line_collection=True)

            ax.set_xlim(0, size * bin_width)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{abs(y):.0f}")
            )
            ax.set_xlabel('mass/charge ratio')
            ax.set_ylabel('relative intensity')

            # タイトルに Record 番号＋SMILES＋InChI＋類似度
            # ax.set_title(
            #     f'Record {idx}: {smiles}\n{inchi}\n'
            #     f'Cosine: {sim_s:.4f} | Weighted: {sim_w:.4f}'
            # )
            wrapped_smiles = textwrap.fill(smiles, width=50)
            wrapped_inchi = textwrap.fill(inchi, width=50)
            wrapped = textwrap.fill(f'SMILES: {wrapped_smiles}\nInChI: {wrapped_inchi}', width=50)
            ax.set_title(
                f'Record {idx}: {wrapped}\nCosine: {sim_s:.4f} | Weighted: {sim_w:.4f}',
                wrap=True
            )

            ax.legend(['True Spectrum','Predicted Spectrum'], loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved multi-page PDF: '{output_pdf}'")

# === 実行例 ===
if __name__ == '__main__':
    # plot_all_spectra('../data/MoNA-GC-MS_filtered_ionization_output.sdf', '../data/comparison_with_smiles_inchi.pdf')
    plot_all_spectra('../data/MoNA_dedup_output.sdf', '../data/MoNA-dedup_all_with_smiles_inchi.pdf')
