import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import pickle
import random
from constants import sliders_table_path
import random





always_feature_columns = ['currTemp', 'currTint', 'aperture',
       'flashFired', 'focalLength', 'isoSpeedRating', 'shutterSpeed',
       'intensity', 'ev']


target_columns = ['Temperature_regr','Tint_regr']



og_table = pd.read_csv(sliders_table_path)
ctemp_std = og_table['currTemp'].std()
ctint_std = og_table['currTint'].std()


def randomize_features(row):
    row = row.copy()
    row['currTemp'] += random.randint(-100,100)/ctemp_std
    row['currTint'] += random.randint(3,3)/ctint_std
    return row


class EditStyleDataset(Dataset):
    def __init__(self, table,embeddings,train=False):
        self.table = table
        self.embeddings_dict = embeddings
        self.table = table[table['image_id'].isin(embeddings.keys())]
        print(len(table))
        self.feature_columns = always_feature_columns + [col for col in table.columns.tolist() if 'camera_model_' in col]
        self.train = train

    def __len__(self):
        return len(self.table)
    
    def __getitem__(self,index):
        features = self.table.loc[index, self.feature_columns]
        if self.train:
            features = randomize_features(features) 
        features = features.to_numpy().astype(np.float32)
        image_id = self.table.iloc[index]['image_id']
        if type(self.embeddings_dict[image_id]) == list:
            embedding = random.choice(self.embeddings_dict[image_id]).astype(np.float32)
        else:
            embedding = self.embeddings_dict[image_id].astype(np.float32)
        #
        target = self.table.loc[index, target_columns].to_numpy().astype(np.float32)
        #
        return torch.from_numpy(features),torch.from_numpy(embedding),torch.from_numpy(target)
 
