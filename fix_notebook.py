import json

path = r"D:\Duy Toan\Python\DUT AI Club\Homework\MobileNet\mobilenet.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = cell['source']
    
    for i, line in enumerate(src):
        # Fix 1: early_stop=40 → 7 in PTMobileNetV3Small and V3Large
        if 'early_stop = 40' in line and '36 epochs' in line:
            src[i] = '        early_stop = 7\n'
            changes.append(f'early_stop fix: {line.strip()}')
        
        # Fix 2: PTMobileNetV3Large pretrained=True → weights=
        if 'mobilenet_v3_large(pretrained=True)' in line:
            src[i] = line.replace('pretrained=True', 'weights=torch_models.MobileNet_V3_Large_Weights.DEFAULT')
            changes.append(f'pretrained fix V3Large: {line.strip()}')
        
        # Fix 3: PT V3Large epochs=36→25 in outer __init__
        if 'num_classes=10, epochs=36' in line and 'batch_size=64, lr=1e-3' in line:
            src[i] = line.replace('epochs=36', 'epochs=25')
            changes.append(f'epochs=36→25: {line.strip()}')
        
        # Fix 4: _ckpt path for V3Large
        if 'ckpt_pt_v3l.pth' in line and '"cifar10_data"' in line and '/content/' not in line:
            src[i] = line.replace('"cifar10_data"', '"/content/cifar10_data"')
            changes.append(f'ckpt path fix V3L: {line.strip()}')
        
        # Fix 5: torch.load without weights_only for V3Large
        if 'torch.load(self._ckpt, map_location=DEVICE)' in line and 'weights_only' not in line:
            src[i] = line.replace('torch.load(self._ckpt, map_location=DEVICE)', 'torch.load(self._ckpt, map_location=DEVICE, weights_only=True)')
            changes.append(f'weights_only fix: {line.strip()}')
        
        # Fix 6: DEMO_EPOCHS = 36 → 25
        if 'DEMO_EPOCHS = 36' in line:
            src[i] = line.replace('DEMO_EPOCHS = 36', 'DEMO_EPOCHS = 25')
            changes.append(f'DEMO_EPOCHS fix: {line.strip()}')
        
        # Fix 7: "6 models" → "7 models"
        if '✓ Tất cả 6 models' in line:
            src[i] = line.replace('6 models', '7 models')
            changes.append(f'6→7 models fix')
        
        # Fix 8: axes.flatten() → np.array(axes).flatten() in Figure 3
        if 'axes_flat = axes.flatten() if n_models > 1 else [axes]' in line:
            src[i] = line.replace('axes.flatten()', 'np.array(axes).flatten()')
            changes.append(f'axes.flatten fix')
    
    # Fix 9: Clear stale error outputs from Cell 5.5
    if 'source' in cell and any('qwngzpzed5' in str(cell.get('id','')) for _ in [1]):
        if cell.get('outputs'):
            cell['outputs'] = []
            changes.append('Cleared stale output from Cell 5.5')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Changes made:')
for c in changes:
    print(' -', c)
print('Done.')
