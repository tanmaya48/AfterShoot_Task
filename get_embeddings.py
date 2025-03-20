import os
import cv2
import numpy as np
import torch
import pickle
import torchvision.models as models
from tqdm import tqdm


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
if __name__ == "__main__":
    device = 0
    #model = torch.nn.Sequential(*list(models.resnet50().children())[:-1])
    model = models.vit_b_16(pretrained=True)
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
            #output = model(image.to(device))
            #output = model.forward_features(image.to(device))
            output = get_embeddings(model,image.to(device))
            embedding = output.cpu().numpy().flatten()
        image_id = os.path.splitext(image_name)[0]
        embeddings_dict[image_id] = embedding
    with open(pickle_filename, 'wb') as f:
        pickle.dump(embeddings_dict, f)

