from rdkit import Chem
import re

# 入力／出力ファイル名
input_sdf = "../data/MoNA-export-GC-MS_Spectra.sdf"
output_sdf = "../data/MoNA-export-GC-MS_Spectra-filtered_ionization.sdf"

# マッチするパターン（大文字小文字を無視）
pattern = re.compile(r'ionization (?:energy|potential)=\s*70\s*eV', re.IGNORECASE)

# SDF 読み込み
supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

# フィルタ結果を書き込む SDWriter
writer = Chem.SDWriter(output_sdf)

for mol in supplier:
    if mol is None:
        continue
    # COMMENT フィールドを持っているか
    if mol.HasProp("COMMENT"):
        comment = mol.GetProp("COMMENT")
        # pattern にマッチすれば書き出し
        if pattern.search(comment) and mol.HasProp("INSTRUMENT TYPE") and mol.GetProp("INSTRUMENT TYPE") == "EI-B":
            writer.write(mol)

writer.close()
print(f"Filtered SDF written to: {output_sdf}")
