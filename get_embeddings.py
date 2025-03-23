import os
import cv2
import numpy as np
import torch
import pickle
import torchvision.models as models
from tqdm import tqdm
import pandas as pd

import random 

def randomize_image(image):
    im_h,im_w,_ = image.shape
    w = random.uniform(0.75,1)
    h = random.uniform(0.75,1)
    x1 = int(random.uniform(0,im_w*(1-w)))
    y1 = int(random.uniform(0,im_h*(1-h)))
    x2 = x1 + int(im_w*w)
    y2 = y1 + int(im_h*h)
    return image[y1:y2,x1:x2,:]


def preprocess(image):
    image = cv2.resize(image,(224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image * (1 / 255)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image - np.array(mean)
    image = image / np.array(std)
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, axis=0)
    image = np.float32(image)
    return torch.tensor(image)


def get_embeddings(model, x):
    # Pass the image through the model up to the penultimate layer
    feats = model._process_input(x)
    batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
    feats = torch.cat([batch_class_token, feats], dim=1)

    feats = model.encoder(feats)

    # We're only interested in the representation of the CLS token that we appended at position 0
    feats = feats[:, 0]
    return feats   # Return the embeddings before classification


from constants import profile_images_dir,sliders_table_path

def main(model,device,image_dir,train_images,val_images):
    pickle_filename = "train_embeddings.pkl"
    train_embeddings_dict = {}
    for image_name in tqdm(train_images):
        image_path = os.path.join(image_dir, f'{image_name}.tif')
        image = cv2.imread(image_path)
        images = [randomize_image(image) for i in range(7)] + [image]
        embeddings = []
        for image_sample in images:
            image_tensor = preprocess(image_sample)
            output = get_embeddings(model,image_tensor.to(device))
            embedding = output.cpu().numpy().flatten()
            embeddings.append(embedding)
        train_embeddings_dict[image_name] = embeddings
    with open(pickle_filename, 'wb') as f:
        pickle.dump(train_embeddings_dict, f)
    #
    #
    pickle_filename = "val_embeddings.pkl"
    val_embeddings_dict = {}
    for image_name in tqdm(val_images):
        image_path = os.path.join(image_dir, f'{image_name}.tif')
        image = cv2.imread(image_path)
        image_tensor = preprocess(image)
        output = get_embeddings(model,image_tensor.to(device))
        embedding = output.cpu().numpy().flatten()
        val_embeddings_dict[image_name] = [embedding]
    with open(pickle_filename, 'wb') as f:
        pickle.dump(val_embeddings_dict, f)




if __name__ == "__main__":
    device = 0
    model = models.vit_h_14(weights='ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1')
    model = model.to(device)
    model.eval()
    image_dir =profile_images_dir 
    train_images = pd.read_csv('profile_train.csv')['image_id'].tolist()
    val_images = pd.read_csv('profile_test.csv')['image_id'].tolist()
    with torch.no_grad():
        main(model,device,image_dir,train_images,val_images)
    
