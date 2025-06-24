import matplotlib.pyplot as plt
import re
from matplotlib.backends.backend_pdf import PdfPages

def extract_block(record, tag):
    pattern = rf'<{tag}>\s*(.*?)\s*(?=(<[^>]+>|$))'
    match = re.search(pattern, record, re.DOTALL)
    return match.group(1).strip() if match else ''

def parse_peaks(block):
    data = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                mz = float(parts[0])
                intensity = float(parts[1])
                data.append((mz, intensity))
            except ValueError:
                continue
    return data

# 入出力ファイル
input_sdf = 'data/sample_output.sdf'
output_pdf = 'data/spectrum_comparison.pdf'

# SDF を読み込んでレコードごとに分割
with open(input_sdf, 'r', encoding='utf-8') as f:
    content = f.read()
records = content.split('$$$$')

# スペクトル抽出
spectra_list = []
for rec in records:
    if rec.strip():
        meas_block = extract_block(rec, 'MASS SPECTRAL PEAKS')
        pred_block = extract_block(rec, 'PREDICTED SPECTRUM')
        measured = parse_peaks(meas_block)
        predicted = parse_peaks(pred_block)
        print(measured)
        if measured or predicted:
            spectra_list.append((measured, predicted))

# PdfPages でマルチページ PDF 作成
with PdfPages(output_pdf) as pdf:
    for idx, (measured, predicted) in enumerate(spectra_list, start=1):
        fig, ax = plt.subplots()

        # 実測スペクトルを青で描画
        if measured:
            mz_meas, int_meas = zip(*measured)
            ax.vlines(mz_meas, ymin=0, ymax=int_meas,
                      color='blue', label='Measured Spectrum')

        # 予測スペクトルをオレンジで下向きに描画
        if predicted:
            mz_pred, int_pred = zip(*predicted)
            int_pred_scaled = [-i * 0.1 for i in int_pred]
            ax.vlines(mz_pred, ymin=0, ymax=int_pred_scaled,
                      color='orange', label='Predicted Spectrum')

        ax.set_xlabel('m/z')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Compound {idx} Spectrum Comparison')
        ax.legend()
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved multi-page PDF: {output_pdf}")
