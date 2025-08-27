#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rdkit import Chem
import matplotlib.pyplot as plt
import argparse
import sys

def read_predicted_spectrum(mol, tag='PREDICTED SPECTRUM'):
    """
    RDKit Mol から指定タグのスペクトル文字列を取り出し、
    [(mz, intensity), ...] のリストにパースして返す。
    """
    if not mol.HasProp(tag):
        raise KeyError(f"Mol にプロパティ '{tag}' が見つかりません。")
    spec_str = mol.GetProp(tag)
    peaks = []
    for line in spec_str.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            mz = float(parts[0])
            inten = float(parts[1])
            peaks.append((mz, inten))
        except ValueError:
            # ヘッダ行などが混入している場合は無視
            continue
    return peaks

def plot_spectrum(peaks, title='Predicted EI-MS Spectrum', out_png=None):
    """
    m/z–強度リストをバーグラフでプロット。
    """
    mz_vals, intensities = zip(*peaks)
    plt.figure(figsize=(10, 6))
    plt.bar(mz_vals, intensities, width=1.0, edgecolor='black')
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"スペクトル画像を保存しました: {out_png}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="SDF に格納された予測 EI-MS スペクトルをプロットする"
    )
    parser.add_argument('sdf_file', help='入力 SDF ファイルパス')
    parser.add_argument('--tag', default='PREDICTED SPECTRUM',
                        help='読み出すスペクトルプロパティ名（デフォルト: PREDICTED SPECTRUM）')
    parser.add_argument('--out', metavar='PNG', default=None,
                        help='出力 PNG ファイル（指定なしなら表示のみ）')
    args = parser.parse_args()

    # SDF を読み込み
    supplier = Chem.SDMolSupplier(args.sdf_file)
    if not supplier or supplier[0] is None:
        print(f"ERROR: SDF から分子を読み込めませんでした: {args.sdf_file}", file=sys.stderr)
        sys.exit(1)

    mol = supplier[0]
    try:
        peaks = read_predicted_spectrum(mol, tag=args.tag)
    except KeyError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    if not peaks:
        print("ERROR: スペクトルデータが見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    plot_spectrum(peaks, title=f"{args.tag} for {args.sdf_file}", out_png=args.out)

if __name__ == '__main__':
    main()
