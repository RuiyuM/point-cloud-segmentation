import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
# Define the file path
file_path = '/people/cs/r/rxm210041/Desktop/test_3d_active/Annotator/logs/al/minknet/syn2poss/VCD_ST/stats/4/states_pre.pkl'
file_path_2 = '/people/cs/r/rxm210041/Desktop/test_3d_active/Annotator/logs/al/minknet/syn2poss/VCD_ST/stats/4/states_true.pkl'


x = [4, 3, 2, 1, 0]
y = [0, 1, 2, 3, 4]
# Function to load the pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
for j in range(5):
    # Load the data
    file_path = f'/people/cs/r/rxm210041/Desktop/test_3d_active/Annotator/logs/al/minknet/syn2poss/VCD_ST_frequency_3/stats/{x[j]}/states_pre.pkl'
    file_path_2 = f'/people/cs/r/rxm210041/Desktop/test_3d_active/Annotator/logs/al/minknet/syn2poss/VCD_ST_reverse_logit_mean/stats/{x[j]}/states_true.pkl'
    loaded_data = load_pickle(file_path)
    loaded_data_2 = load_pickle(file_path_2)
    pre_selected_label_count = loaded_data[0][0]
    # You can now use the loaded_data as needed
    entropy_pre = []
    entropy_true = []
    count_sum_list = []
    for i in range(len(loaded_data)):
        pre_selected_label_count = loaded_data[i][0]
        pre_selected_label_count_t = loaded_data_2[i][0]
        probabilities = pre_selected_label_count.float() / pre_selected_label_count.sum()
        count_sum = pre_selected_label_count.sum()
        # Remove zero probabilities to avoid log(0) which is undefined
        probabilities = probabilities[probabilities > 0]

        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        entropy_pre.append(entropy)
        probabilities = pre_selected_label_count_t.float() / pre_selected_label_count_t.sum()
        count_sum = pre_selected_label_count_t.sum()
        # Remove zero probabilities to avoid log(0) which is undefined
        probabilities = probabilities[probabilities > 0]
        count_sum_list.append(count_sum / loaded_data_2[i][1])
        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        entropy_true.append(entropy)

    entropy_tensor = torch.stack(entropy_pre)

    # Calculate the mean
    mean_entropy = torch.mean(entropy_tensor)

    entropy_tensor_t = torch.stack(entropy_true)

    # Calculate the mean
    mean_entropy_t = torch.mean(entropy_tensor_t)

    count_tensor = torch.stack(count_sum_list)

    count_tensor_mean = torch.mean(count_tensor)


    #############
    count_sum_list = entropy_tensor.cpu().numpy()
    count_sum_list = np.sort(count_sum_list)
    # Create a plot
    bins = np.arange(0, 3.1, 0.1)  # Creates an array from 0 to 1 with a step of 0.1

    # Create the histogram
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.hist(count_sum_list, bins=bins, edgecolor='black', alpha=0.7)

    # Adding titles and labels
    plt.title(f'Histogram of pre_entropy {y[j]}')  # You can uncomment other titles as needed
    # plt.title('Histogram of true_entropy')  # Adjust title as needed
    # plt.title('Histogram of pre_entropy')  # Adjust title as needed
    plt.xlabel('Value Range')  # Label for the x-axis
    plt.ylabel('Frequency')  # Label for the y-axis
    plt.savefig(f'pre_entropy_kitti_{y[j]}.png')
    # Show the plot
    plt.show()
