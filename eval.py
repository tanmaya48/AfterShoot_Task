import torch
import numpy as np
from tqdm import tqdm

from model_structure import StyleModel
from dataset import EditStyleDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('best_model.pt',weights_only=False)
model.to(device)
model.eval()


import pandas as pd
import pickle

embeddings_path = 'profile_embeddings.pkl'
with open(embeddings_path,'rb') as file:
    embeddings_dict = pickle.load(file)


test_table = pd.read_csv('profile_test.csv')
test_dataset = EditStyleDataset(test_table,embeddings_dict)


df = pd.read_csv('profile.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df[['Tint']])


def post_process(temp,tint):
    temp = temp * 5000
    tint = scaler.inverse_transform([[tint]])[0]
    return temp,tint

pred_temps = []
pred_tints = []
gt_temps = []
gt_tints = []

for i,(features, embeddings, targets) in enumerate(test_dataset):
    features = features.to(device)
    embeddings = embeddings.to(device)
    with torch.no_grad():
        output = model(embeddings.unsqueeze(0),features.unsqueeze(0))
        temp,tint = output[0].cpu().numpy()
    temp,tint = post_process(temp,tint)
    real_temp,real_tint = test_table[['Temperature','Tint']].iloc[i].tolist()
    pred_temps.append(temp)
    gt_temps.append(real_temp)
    pred_tints.append(tint)
    gt_tints.append(real_tint)
    if i == len(test_dataset)-1:
        break
print(f"Number of samples in test set : {len(test_dataset)}")

from sklearn.metrics import mean_absolute_error, r2_score
print(f"Temperature-> MAE: {mean_absolute_error(gt_temps,pred_temps)}, R2: {r2_score(gt_temps,pred_temps)}")
print(f"Tint-> MAE: {mean_absolute_error(gt_tints,pred_tints)}, R2: {r2_score(gt_tints,pred_tints)}")
