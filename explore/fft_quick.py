import json
import os
from collections import Counter

file_path = 'jsons/datalist_panda_0.json'
with open(file_path, 'r') as file:
    data = json.load(file)

data_root = '/data/breast-cancer/PANDA/train_images_FFT_WSI/'
existing_data = {'training': [], 'validation': []}
for split, paths in data.items():
    for item in paths:
        file_name = item['image'][:-5] + '.npz'
        if os.path.exists(data_root + file_name):
            existing_data[split].append(item)

# labels = [item['label'] for item in existing_data['training']]
# label_counts = Counter(labels)

output_file_path = 'datalists/datalist_panda_fft.json'
with open(output_file_path, 'w') as output_file:
    json.dump(existing_data, output_file)

output_file_path = 'datalists/datalist_panda_fft_quick.json'
existing_data['training'] = existing_data['training'][:50]
existing_data['validation'] = existing_data['validation'][:25]
with open(output_file_path, 'w') as output_file:
    json.dump(existing_data, output_file)