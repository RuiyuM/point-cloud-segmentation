import pickle
import torch

file_path = '/people/cs/r/rxm210041/Desktop/test_3d_active/Annotator/logs/al/minknet/syn2poss/VCD_collect/stats/0/states_pre.pkl'

# Open the file and load its content
with open(file_path, 'rb') as f:
    data = pickle.load(f)
total = len(data)
more_than_one_class = 0

for item in data:
    tensor_counts = item[0].squeeze()
    nonzero_count = (tensor_counts > 0).sum().item()
    if nonzero_count > 1:
        more_than_one_class += 1
        print(item)

percentage = (more_than_one_class / total) * 100 if total > 0 else 0

print(f"\nCount with >1 non-zero classes: {more_than_one_class}/{total} ({percentage:.2f}%)")  # Be careful with large outputs
