from similarity import compute_similarities
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# 1) 類似度を取得
sims = compute_similarities('../data/sample_output.sdf', record_index=0)

# 2) PDF にプロット
with PdfPages('spectrum_comparison.pdf') as pdf:
    # …プロット処理（stem プロットなど）…
    plt.figure(figsize=(8, 4))
    # ここでは可視化用にもう一度ベクトルを読み込んでも良いですし、
    # spectrum_similarity モジュール内に可視化用関数を追加してもOK。
    plt.title(f"Cosine: {sims['cosine_similarity']:.4f}  |  Weighted: {sims['weighted_cosine_similarity']:.4f}")
    # plt.stem(...)

    pdf.savefig()
    plt.close()
