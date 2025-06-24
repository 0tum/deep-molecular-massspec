from rdkit import Chem

# 入力／出力ファイル名
input_sdf = "../data/MoNA-export-GC-MS_Spectra.sdf"
output_sdf = "../data/MoNA-export-GC-MS_Spectra_filtered.sdf"

# SDF 読み込みつつオブジェクト化（失敗したエントリは None になる）
supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

# フィルタ結果を書き込む SDWriter を用意
writer = Chem.SDWriter(output_sdf)

mol_count = 0

for mol in supplier:
    if mol is None:
        # 読み込み失敗エントリはスキップ
        continue

    # プロパティ名はタグのまま（空白や大文字小文字も含めて一致させる）
    if mol.HasProp("INSTRUMENT TYPE") and mol.GetProp("INSTRUMENT TYPE") == "EI-B":
        writer.write(mol)
        mol_count += 1

writer.close()
print(f"Filtered SDF written to: {output_sdf}")
print(f"Total molecules processed: {mol_count}")
