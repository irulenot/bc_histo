import os
import random
import json

# Directory path
directory = '/data/breast-cancer/PANDA/train_images_FFT10000_WSI_grayscaled'
# Get all filenames in the directory
filenames = os.listdir(directory)
# Strip ".npz" suffix from each filename
filenames = [filename.replace('.npz', '') for filename in filenames]

json_file = '/home/anthony/bc_histo/datalists/datalist_panda_0.json'

# Load the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

for dataset in ['training', 'validation']:
    filtered_data = []
    for entry in data[dataset]:
        img = entry['image'].split('.')[0]
        if img in filenames:
            filtered_data.append(entry)
    data[dataset] = filtered_data

# Combine training and validation data into a single list
combined_data = data['training'] + data['validation']

# Shuffle the combined data
random.seed(42)
random.shuffle(combined_data)

# Calculate split index for training and validation
split_index = int(0.8 * len(combined_data))

# Distribute into training and validation sets
data['training'] = combined_data[:split_index]
data['validation'] = combined_data[split_index:]

json_file = '/home/anthony/bc_histo/datalists/datalist_panda_fft_10000.json'

# Save the data dictionary as JSON
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

# Restrict to max 16 elements in training and validation
data['training'] = data['training'][:16]
data['validation'] = data['validation'][:16]

# Path to save the JSON file
json_file = '/home/anthony/bc_histo/datalists/datalist_panda_fft_quick_10000.json'

# Save the data dictionary as JSON
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print('done')