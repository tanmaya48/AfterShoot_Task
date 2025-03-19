import torch
from tqdm import tqdm

from model_structure import StyleModel
from dataset import EditStyleDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import pandas as pd
import pickle

import torch.nn as nn


# Optimizer
import torch.optim as optim



def train(model,train_dataloader,val_dataloader,optimizer,l1_loss):
    model.train()
    train_loss = 0.0
    print("training:")
    for (features, embeddings, targets) in tqdm(train_dataloader):
        features = features.to(device)
        embeddings = embeddings.to(device)
        #
        optimizer.zero_grad()
        #
        output = model(embeddings,features)
        loss = l1_loss(output.cpu(),targets)
        #
        train_loss += loss
        loss.backward()
        optimizer.step()
    #
    model.eval()
    print("validation")
    with torch.no_grad():
        val_loss = 0
        for (features, embeddings, targets) in tqdm(val_dataloader):
            features = features.to(device)
            embeddings = embeddings.to(device)
            #
            output = model(embeddings,features)
            loss = l1_loss(output.cpu(),targets)
            #
            val_loss += loss
    return train_loss,val_loss

if __name__ == '__main__':
    embeddings_path = 'profile_embeddings.pkl'
    with open(embeddings_path,'rb') as file:
        embeddings_dict = pickle.load(file)

    train_dataset = EditStyleDataset(pd.read_csv('profile_train.csv'),embeddings_dict)
    test_dataset = EditStyleDataset(pd.read_csv('profile_test.csv'),embeddings_dict)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=128,shuffle=True)

    model = StyleModel(2048,len(train_dataset.feature_columns))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    l1_loss = nn.L1Loss()

    #
    best_val_loss =float('inf') 
    best_epoch = 0
    patience = 30
    n_epochs = 1000
    for epoch in range(n_epochs):
        if epoch - best_epoch> patience:
            break
        print(f"epoch {epoch}")
        train_loss,val_loss = train(model,train_loader,test_loader,optimizer,l1_loss)
        print(f"train_loss : {train_loss}, val_loss : {val_loss}")
        if val_loss < best_val_loss:
            torch.save(model, "best_model.pt")
            best_val_loss = val_loss
            best_epoch = epoch

