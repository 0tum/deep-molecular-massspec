import matplotlib.pyplot as plt
import re

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

# Read file and split records
with open('../data/sample_output.sdf', 'r', encoding='utf-8') as f:
    content = f.read()
records = content.split('$$$$')

# Extract spectra
spectra_list = []
for rec in records:
    if rec.strip():
        meas_block = extract_block(rec, 'MASS SPECTRAL PEAKS')
        pred_block = extract_block(rec, 'PREDICTED SPECTRUM')
        measured = parse_peaks(meas_block)
        predicted = parse_peaks(pred_block)
        if measured or predicted:
            spectra_list.append((measured, predicted))

# Create stick plots with English internal labels
for idx, (measured, predicted) in enumerate(spectra_list, start=1):
    fig, ax = plt.subplots()
    
    if measured:
        mz_meas, int_meas = zip(*measured)
        ax.vlines(mz_meas, ymin=0, ymax=int_meas, color='blue', label='Measured Spectrum')
    if predicted:
        mz_pred, int_pred = zip(*predicted)
        int_pred_scaled = [i * (-0.1) for i in int_pred]
        ax.vlines(mz_pred, ymin=0, ymax=int_pred_scaled, color='orange', label='Predicted Spectrum')

    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Compound {idx} Spectrum Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'../data/spectrum_{idx}.png')
    plt.show()
