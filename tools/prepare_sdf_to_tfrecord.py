#!/usr/bin/env python
"""Normalize SDF fields and create TFRecords for training.

This script:
- Loads an input SDF (e.g., MoNA export) with RDKit.
- Normalizes key fields to match this repo expectations:
  - MASS SPECTRAL PEAKS: as lines of "mz intensity".
  - EXACT MASS: filled if missing using RDKit exact mass.
  - INCHIKEY: filled if missing if RDKit InChI is available; else leave blank.
  - SMILES: set to canonical SMILES.
- Optionally oversamples fluorine-containing molecules (F, atomic number 9) in train.
- Splits into train/test and writes TFRecords and a minimal dataset_config.json.

Usage example:
  python tools/prepare_sdf_to_tfrecord.py \
    --input_sdf=../sdf/data/sample.sdf \
    --output_dir=./prepared_mona \
    --train_ratio=0.9 \
    --oversample_f_factor=20
"""



import argparse
import json
import os
import random
import re
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import feature_utils
import mass_spec_constants as ms_constants
import parse_sdf_utils


def _find_peak_prop_name(mol: Chem.Mol) -> Optional[str]:
  # Preferred tag name in this repo
  if mol.HasProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS):
    return ms_constants.SDF_TAG_MASS_SPEC_PEAKS
  # Fallback: search props containing 'PEAK' (case-insensitive)
  for name in mol.GetPropNames():
    if re.search(r'peak', name, flags=re.IGNORECASE):
      return name
  return None


def _parse_peaks_generic(pk_str: str) -> Tuple[List[int], List[float]]:
  """Parse various peak formats into (mz, intensity) lists.

  Accepts lines separated by newline/semicolon, pairs separated by
  whitespace/colon/comma. Ignores non-numeric tokens.
  """
  parts = re.split(r'[\n;]+', pk_str.strip())
  mz_list: List[int] = []
  it_list: List[float] = []
  for p in parts:
    p = p.strip()
    if not p:
      continue
    # Replace common delimiters with space
    q = re.sub(r'[:,\t]+', ' ', p)
    toks = q.split()
    if len(toks) < 2:
      continue
    try:
      mz = int(round(float(toks[0])))
      it = float(toks[1])
    except Exception:
      # Try to find first two numeric tokens
      nums = []
      for t in toks:
        try:
          nums.append(float(t))
        except Exception:
          pass
      if len(nums) >= 2:
        mz = int(round(nums[0]))
        it = float(nums[1])
      else:
        continue
    if mz < 0:
      continue
    mz_list.append(mz)
    it_list.append(it)
  # Deduplicate by mz: keep max intensity
  d = {}
  for m, it in zip(mz_list, it_list):
    d[m] = max(it, d.get(m, 0.0))
  mz_sorted = sorted(d.keys())
  its_sorted = [d[m] for m in mz_sorted]
  return mz_sorted, its_sorted


def _normalize_mol_props(mol: Chem.Mol, max_peak_loc: int) -> bool:
  """Normalize props in-place. Returns True if peaks ok, else False."""
  peak_name = _find_peak_prop_name(mol)
  if not peak_name:
    return False
  pk_str = mol.GetProp(peak_name)
  # Try native parser first if already in expected format
  mz, it = None, None
  try:
    mz, it = feature_utils.parse_peaks(pk_str)
  except Exception:
    # Fallback generic parser
    mz, it = _parse_peaks_generic(pk_str)
  if not mz:
    return False
  # Clip to max_peak_loc
  mz2, it2 = [], []
  for m, v in zip(mz, it):
    if m < max_peak_loc:
      mz2.append(m)
      it2.append(v)
  if not mz2:
    return False
  dense = feature_utils.make_dense_mass_spectra(mz2, it2, max_peak_loc)
  # Normalize to integer-ish intensities like original dataset if needed
  # Here we keep original scale; downstream will reweight via hparams.
  spec_str = feature_utils.convert_spectrum_array_to_string(dense)
  mol.SetProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS, spec_str)

  # EXACT MASS
  try:
    _ = float(mol.GetProp(ms_constants.SDF_TAG_MOLECULE_MASS))
  except Exception:
    try:
      em = rdMolDescriptors.CalcExactMolWt(mol)
      mol.SetProp(ms_constants.SDF_TAG_MOLECULE_MASS, str(em))
    except Exception:
      pass

  # INCHIKEY (best-effort)
  if not mol.HasProp(ms_constants.SDF_TAG_INCHIKEY):
    try:
      # Requires RDKit built with InChI support
      from rdkit.Chem import inchi
      ik = inchi.MolToInchiKey(mol)
      mol.SetProp(ms_constants.SDF_TAG_INCHIKEY, ik)
    except Exception:
      # Fallback to SMILES hash
      smi = feature_utils.get_smiles_string(mol)
      mol.SetProp(ms_constants.SDF_TAG_INCHIKEY, str(hash(smi)))

  # NAME
  if not mol.HasProp('NAME'):
    mol.SetProp('NAME', feature_utils.get_smiles_string(mol))

  # SMILES
  mol.SetProp('SMILES', feature_utils.get_smiles_string(mol))

  return True


def _contains_fluorine(mol: Chem.Mol) -> bool:
  return any(at.GetAtomicNum() == 9 for at in mol.GetAtoms())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--input_sdf', required=True)
  ap.add_argument('--output_dir', required=True)
  ap.add_argument('--train_ratio', type=float, default=0.9)
  ap.add_argument('--max_atoms', type=int, default=ms_constants.MAX_ATOMS)
  ap.add_argument('--max_peak_loc', type=int, default=ms_constants.MAX_PEAK_LOC)
  ap.add_argument('--oversample_f_factor', type=int, default=0,
                  help='Oversample F-containing molecules in train by this factor')
  ap.add_argument('--seed', type=int, default=42)
  args = ap.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  suppl = Chem.SDMolSupplier(args.input_sdf)
  mols: List[Chem.Mol] = []
  for i, m in enumerate(suppl):
    if m is None:
      continue
    if not _normalize_mol_props(m, args.max_peak_loc):
      continue
    # Filter by parse_sdf_utils heuristics
    if not feature_utils.check_mol_has_non_empty_mass_spec_peak_tag(m):
      continue
    if feature_utils.get_largest_mass_spec_peak_loc(m) >= args.max_peak_loc:
      continue
    mols.append(m)

  if not mols:
    raise RuntimeError('No valid molecules parsed from input SDF.')

  # Split train/test
  random.seed(args.seed)
  random.shuffle(mols)
  n_train = int(len(mols) * args.train_ratio)
  train_mols = mols[:n_train]
  test_mols = mols[n_train:]

  # Oversample F in train if requested
  if args.oversample_f_factor and args.oversample_f_factor > 1:
    f_mols = [m for m in train_mols if _contains_fluorine(m)]
    train_mols = train_mols + f_mols * (args.oversample_f_factor - 1)
    random.shuffle(train_mols)

  # Write normalized SDFs (optional but useful)
  norm_train_sdf = os.path.join(args.output_dir, 'train_normalized.sdf')
  norm_test_sdf = os.path.join(args.output_dir, 'test_normalized.sdf')
  with Chem.SDWriter(norm_train_sdf) as w:
    for m in train_mols:
      w.write(m)
  with Chem.SDWriter(norm_test_sdf) as w:
    for m in test_mols:
      w.write(m)

  # Write TFRecords
  train_record = os.path.join(args.output_dir, 'train.tfrecord')
  test_record = os.path.join(args.output_dir, 'test.tfrecord')

  parse_sdf_utils.write_dicts_to_example(
      train_mols, train_record, args.max_atoms, args.max_peak_loc)
  parse_sdf_utils.write_info_file(train_mols, train_record)

  parse_sdf_utils.write_dicts_to_example(
      test_mols, test_record, args.max_atoms, args.max_peak_loc)
  parse_sdf_utils.write_info_file(test_mols, test_record)

  # Minimal dataset config
  cfg = {
      'SPECTRUM_PREDICTION_TRAIN': [os.path.basename(train_record)],
      'SPECTRUM_PREDICTION_TEST': [os.path.basename(test_record)],
  }
  with open(os.path.join(args.output_dir, 'dataset_config.json'), 'w') as f:
    json.dump(cfg, f)

  print('Prepared:', args.output_dir)
  print(' Train molecules:', len(train_mols))
  print(' Test molecules :', len(test_mols))


if __name__ == '__main__':
  main()

