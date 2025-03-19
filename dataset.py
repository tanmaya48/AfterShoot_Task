import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import pickle

always_feature_columns = ['currTemp', 'currTint', 'aperture',
       'flashFired', 'focalLength', 'isoSpeedRating', 'shutterSpeed',
       'intensity', 'ev']


target_columns = ['Temperature_regr','Tint_regr']


class EditStyleDataset(Dataset):
    def __init__(self, table,embeddings):
        self.table = table
        self.embeddings_dict = embeddings
        self.table = table[table['image_id'].isin(embeddings.keys())]
        self.feature_columns = always_feature_columns + [col for col in table.columns.tolist() if 'camera_model_' in col]

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self,index):
        features = self.table.loc[index, self.feature_columns].to_numpy().astype(np.float32)
        image_id = self.table.iloc[index]['image_id']
        embedding = self.embeddings_dict[image_id].astype(np.float32)
        #
        target = self.table.loc[index, target_columns].to_numpy().astype(np.float32)
        #
        return torch.from_numpy(features),torch.from_numpy(embedding),torch.from_numpy(target)
 
