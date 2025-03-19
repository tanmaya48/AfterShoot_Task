import os
import cv2
import numpy as np
import torch
import pickle
import torchvision.models as models
from tqdm import tqdm


def preprocess(image):
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

from constants import profile_images_dir,sliders_table_path
if __name__ == "__main__":
    device = 0
    model = torch.nn.Sequential(*list(models.resnet50().children())[:-1])
    model = model.to(device)
    model.eval()
    image_dir =profile_images_dir 
    pickle_filename = "profile_embeddings.pkl"
    embeddings_dict = {}
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        image = preprocess(image)
        with torch.no_grad():
            output = model(image.to(device))
            embedding = output.cpu().numpy().flatten()
        image_id = os.path.splitext(image_name)[0]
        embeddings_dict[image_id] = embedding
    with open(pickle_filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)

