from rdkit import Chem

# 入力／出力ファイル名
input_sdf = "../data/MoNA-GC-MS_filtered_ionization.sdf"

# SDF 読み込みつつオブジェクト化（失敗したエントリは None になる）
supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)

mol_count = 0

for mol in supplier:
    if mol is None:
        # 読み込み失敗エントリはスキップ
        continue

    mol_count += 1

print(f"Total molecules processed: {mol_count}")
