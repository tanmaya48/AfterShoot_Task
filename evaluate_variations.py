import torch
import numpy as np
from tqdm import tqdm
import os
import cv2
import random

from model_structure import StyleModel
from dataset import EditStyleDataset

import torchvision.models as models
from constants import profile_images_dir,sliders_table_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('best_model.pt',weights_only=False)
model.to(device)
model.eval()


import pandas as pd

from get_embeddings import get_embeddings,preprocess

embedding_model = models.vit_h_14(weights='ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1')
embedding_model = embedding_model.to(device)






df = pd.read_csv('profile.csv')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df[['Tint']])


def post_process(temp,tint):
    temp = temp * 5000
    tint = scaler.inverse_transform([[tint]])[0]
    return temp,tint



def randomize_image(image):
    im_h,im_w,_ = image.shape
    ratio = random.uniform(0.5,1)
    w = ratio
    h = ratio
    x1 = int(random.uniform(0,im_w*(1-w)))
    y1 = int(random.uniform(0,im_h*(1-h)))
    x2 = x1 + int(im_w*w)
    y2 = y1 + int(im_h*h)
    return image[y1:y2,x1:x2,:]


def get_preds(model,embeddings,features):
    with torch.no_grad():
        output = model(embeddings.unsqueeze(0),features.unsqueeze(0))
        temp,tint = output[0].cpu().numpy()
    temp,tint = post_process(temp,tint)
    return temp,tint


def get_random_embeddings_set(embedding_model,image):
    images = [image]+[randomize_image(image) for i in range(4)] 
    
    embeddings = []
    for image_sample in images:
        image_tensor = preprocess(image_sample)
        with torch.no_grad():
            output = get_embeddings(embedding_model,image_tensor.to(device))
            embedding = output.flatten()
            embeddings.append(embedding)
    return embeddings



def get_random_features_set(feature_row):
    gt_features = feature_row.to_numpy().astype(np.float32)
    features_set = [ torch.from_numpy(gt_features).to(device) ]
    for i in range(4):
        row = feature_row.copy()
        row['currTemp'] += (random.randint(-50,50))/df['currTemp'].std()
        row['currTint'] += (random.randint(-1,1))/df['currTint'].std()
        row = row.to_numpy().astype(np.float32)
        row = torch.from_numpy(row)
        row = row.to(device)
        features_set.append(row)
    return features_set


pred_temps = []
pred_tints = []
gt_temps = []
gt_tints = []

image_dir = profile_images_dir


test_table = pd.read_csv('profile_test.csv')

from dataset import always_feature_columns
feature_columns = always_feature_columns + [col for col in test_table.columns.tolist() if 'camera_model_' in col]


temp_vars, temp_stds = [],[]
tint_vars, tint_stds = [],[]
for idx, row in tqdm(test_table.iterrows()):
    image_temps,image_tints = [],[]
    image_name = row['image_id']
    real_temp,real_tint = test_table[['Temperature','Tint']].iloc[idx].tolist()

    features = test_table.iloc[idx][feature_columns]
    features_set = get_random_features_set(features)

    image_path = os.path.join(image_dir, f'{image_name}.tif')
    image = cv2.imread(image_path)
    embeddings = get_random_embeddings_set(embedding_model,image)
    #
    #
    #for emb in embeddings:
    #    pred_temp,pred_tint = get_preds(model,emb,features_set[0])
    #    image_temps.append(pred_temp)
    #    image_tints.append(pred_tint)
    #
    #for feat in features_set:
    #    pred_temp,pred_tint = get_preds(model,embeddings[0],feat)
    #    image_temps.append(pred_temp)
    #    image_tints.append(pred_tint)
    #
    for i in range(5):
        pred_temp,pred_tint = get_preds(model,embeddings[i],features_set[i])
        image_temps.append(pred_temp)
        image_tints.append(pred_tint)
    image_temps = np.array(image_temps).flatten()
    image_tints = np.array(image_tints).flatten()
    #
    temp_stds.append(np.std(image_temps))
    temp_vars.append(np.var(image_temps))
    tint_stds.append(np.std(image_tints))
    tint_vars.append(np.var(image_tints))

print(np.mean(temp_stds))
print(np.mean(temp_vars))
print(np.mean(tint_stds))
print(np.mean(tint_vars))
