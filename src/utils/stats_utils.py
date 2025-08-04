import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def cal_router_weight(**kargs):
    weights = kargs['weight']
    # set a static variable
    if not hasattr(cal_router_weight, 'weights_list'):
        cal_router_weight.weights_list = []
    # weights: (batch_size, num_classes)
    weights = weights.detach().cpu()
    cal_router_weight.weights_list.append(weights)
    # calculate the mean of weights_list
    weights_list = torch.concat(cal_router_weight.weights_list, dim=0)
    weights_mean = torch.mean(weights_list, dim=0)
    # to numpy
    weights_mean = weights_mean.numpy()
    return weights_mean


def save_tsne_tensor(dataset, model_name, vids, labels, tensors):
    # save vids: tensors dict to static data
    # Create a dictionary to store the data
    new_data = {}

    # Iterate through vids, labels, and tensors
    for vid, label, tensor in zip(vids, labels, tensors):
        # Store the data for each video
        new_data[vid] = {
            "label": label,
            "tensor": tensor.detach().cpu()
        }

    # Define the output file path
    output_path = Path(f"statis/data/tsne_tensors_{dataset}_{model_name}.pt")

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if output_path.exists():
        # Load existing data
        existing_data = torch.load(output_path)
        
        # Merge existing data with new data
        existing_data.update(new_data)
        
        # Save the merged data
        torch.save(existing_data, output_path)
    else:
        # Save the new data as a torch tensor
        torch.save(new_data, output_path)

    print(f"Data saved/updated in {output_path}")
    
def save_weight_tensor(dataset, vids, labels, tensors):
    # save vids: tensors dict to static data
    # Create a dictionary to store the data
    new_data = {}
    if dataset == 'MHClipEN':
        dataset_path = 'data/MultiHateClip/en'
    elif dataset == 'MHClipZH':
        dataset_path = 'data/MultiHateClip/zh'
    replace_dict = {
        'a': 0,
        't': 1,
        'v': 2,
        'at': 3,
        'av': 4,
        'tv': 5,
        'atv': 6
    }
        
    modal_df = pd.read_json(f'{dataset_path}/modal.jsonl', orient='records', lines=True)
    # Iterate through vids, labels, and tensors
    for vid, label, tensor in zip(vids, labels, tensors):
        if vid in modal_df['vid'].values:
        # Store the data for each video
            modal = modal_df[modal_df['vid'] == vid]['modal'].values[0]
            if modal != '':
                modal = replace_dict[modal]
                new_data[vid] = {
                "label": modal,
                "tensor": tensor.detach().cpu()
                }

    # Define the output file path
    output_path = Path(f"statis/data/weight_tensors_{dataset}.pt")

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if output_path.exists():
        # Load existing data
        existing_data = torch.load(output_path)
        
        # Merge existing data with new data
        existing_data.update(new_data)
        
        # Save the merged data
        torch.save(existing_data, output_path)
    else:
        # Save the new data as a torch tensor
        torch.save(new_data, output_path)

    print(f"Data saved/updated in {output_path}")