import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

filepath = 'd:/Skripsi/new_data/01_granger_causality/01_gc_analysis_optimized.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

for i in range(5, 12):
    if i < len(data.get('cells', [])):
        cell = data['cells'][i]
        cell_type = cell.get('cell_type')
        source = ''.join(cell.get('source', []))
        print(f"=== CELL {i} ({cell_type}) ===")
        print(source)
        print("================\n")
