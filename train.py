import os

import data_loader
from features_generator import FeaturesGenerator
from imageEmbedding import ImageEmbedding
from pyDeepInsight import ImageTransformer, Norm2Scaler

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import progress_bar
from archs.resnet import ResNet50

from archs.efficientnet import EfficientNetV2
from archs.vit import ViTModel, DeepViTModel, CaiTModel

import warnings; 
warnings.simplefilter('ignore')

from lion_pytorch import Lion

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, accuracy_score, precision_score, recall_score

from tqdm import tqdm

def train_test_model(X_train, X_test, y_train, y_test, epochs, idx=0):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    transform_train = transforms.Compose([ 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize(mean=[0.5], std=[0.5])
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize(mean=[0.5], std=[0.5])
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    X_train_tensor = torch.stack([transform_train(img) for img in X_train]).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    
    X_test_tensor = torch.stack([transform_test(img) for img in X_test]).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)

    batch_size = 128

    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = DeepViTModel()
    #model = ViTModel()
    #model = CaiTModel()

    #model = ResNet50(num_classes=5)
    #model = EfficientNetV2(num_classes=5)

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001) #Best

    #optimizer = Lion(model.parameters(), lr = 0.001, weight_decay = 1e-2)
    #optimizer = optim.AdamW(model.parameters(), lr=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    
    best_acc = 0
    best_train = 0
    best_loss = 0
    best_epoch = 0
    best_f1 = 0
    best_cm = 0

    train_data = pd.DataFrame()

    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        loop = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_description(f"Train [{epoch}/{epochs}]")
            loop.set_postfix(loss=(train_loss/(batch_idx+1)), acc=100.*correct/total)
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc = 100.*correct/total
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        y_true = []
        y_pred = []
        with torch.no_grad():
            loop = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                y_true += targets.tolist()
                y_pred += predicted.tolist()

                loop.set_description(f"Test [{epoch}/{epochs}]")
                loop.set_postfix(loss=((test_loss/(batch_idx+1))), acc=100.*correct/total)
                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)#, labels=list(data.labels[0]))

        # Compute the f1 score
        f1 = 100.*f1_score(y_true, y_pred, average='macro')

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
        #if f1 > best_f1:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, ('./checkpoint/ckpt' + str(idx) +'.pth'))
            best_acc = acc
            best_loss = (test_loss/(batch_idx+1))
            best_epoch = epoch
            best_train = train_acc
            best_f1 = f1
            best_cm = cm

        new_row = pd.DataFrame({
                    'train_acc': train_acc,
                    'test_acc': acc,
                    'test_f1': f1,
                    'epoch': epoch}, index=[0])

        train_data = pd.concat([train_data, new_row], ignore_index=True)
        #scheduler.step()

        #print("Learning Rate: ", optimizer.param_groups[-1]['lr'])

        print("Best Accuracy", best_acc, best_loss, best_epoch)
        print(f'Best F1 score: {best_f1:.4f}')

    print('Confusion matrix:')
    print('=================')
    print(np.array2string(best_cm, separator=', ', 
                           formatter={'int': lambda x: f'{x:5d}'},
                           prefix='[[', suffix=']]'))
    print('')

    print(f'Best F1 score: {best_f1:.4f}')

    train_data.to_csv("data/training/train_test_" + str(idx) + ".csv", index=True)

    return best_train, best_acc, best_epoch, best_f1, best_cm

def main():

    ### Loading Dataset Data into Pandas Dataframe ###
    
    ## Init the dataset Loader
    loader = data_loader.DataLoader("GeoLife Trajectories", "dataset/Data/")

    ## process the loaded dataset to generate the data to extract features
    ## only needed if the features are not extrated yet 
    #loader.loadData()

    ## save the processed data into a csv file
    #loader.save2csv("data/dataset.csv")

    ## load the dataset from the csv file if already save from previous stage
    loader.readcsv("data/dataset.csv")


    ### Extract features from the dataset ###

    ## create the generator for Feature extraction Module
    generator = FeaturesGenerator(loader.data)

    ## Extract basic features from the data loaded from the dataset
    #generator.extractBasicFeatures()

    ## Save Basic Features to a csv file
    #generator.save2csv("data/basic_features.csv")

    ## Extract Advanced Features from Basic features
    #generator.extractFeatures()

    ## Save Advanced Features to a csv file
    #generator.save2csv("data/full_features.csv")

    ## Load Basic and Advanced features from saved csv files
    generator.readcsv("data/basic_features.csv", "data/full_features.csv")

    ## Remove data with labels that we don't want to train and test
    #['walk', 'bike', 'car', 'taxi', 'bus', 'subway', 'train', 'boat', 'run', 'airplane', 'motorcycle']
    for l in ['run', 'motorcycle', 'boat', 'taxi', 'subway', 'airplane']: 
        generator.dataset.drop(generator.dataset[generator.dataset['label'] == l].index, inplace=True)
    
    generator.dataset = generator.dataset.reset_index(drop=True)
    generator.labels = list(generator.dataset['label'].unique())

    ## Print labels and number of sample in each label
    for label in generator.labels:
        print(label, len(generator.dataset.loc[generator.dataset['label'] == label]), 
            len(generator.features.loc[generator.features['label'] == label]))


    ### Train and Test with the Deepinsight image transformer and Deep learning models ###
    
    repeats = 20 #repeats for the cross-validation
    bacc = 0
    bf1 = 0
    idx = 0
    size = (64, 64) #images size for training the model

    ## nonlinear dimensionality reduction technique to identify the coordinates of features in images
    reducer = TSNE(verbose=2, n_components=2, perplexity=50,
                    metric='cosine',
                    init='random',
                    learning_rate='auto',
                    #n_iter = 1000,
                    n_jobs=-1)

    dados = pd.DataFrame()


    for i in range(0, repeats):

        ## Prepare the DeepInsight Module to generate images from features
        emb = ImageEmbedding(features=generator.dataset,#[features_selected+['label']], #generator=poly, 
                reducer=reducer, size=size)

        # Process features to create the training and testing sets to train a model
        emb.fit_embedding(False)

        # Split the dataset into Train and Test sets
        X_train, X_test, y_train, y_test = train_test_split(
        emb.X, emb.Y, test_size=0.2, stratify=emb.Y, random_state=0)

        train, acc, epoch, f1, cm = train_test_model(X_train, X_test, y_train, y_test, 300, i)

        if f1 > bf1:
            bf1 = f1
            bacc = acc
            idx = i
        with open('coords/coords'+str(i)+'.npy', 'wb') as f:
            np.save(f, emb.coords)#.to_numpy())

        new_row = pd.DataFrame({
                    'idx model': i,
                    'Model': "ViT",
                    'Classes': 5,
                    'size': 64,
                    'perplexity': 50,
                    'Train': train,
                    'Accuracy': acc,
                    'F1-Score': f1,
                    'Epochs': epoch}, index=[0])

        dados = pd.concat([dados, new_row], ignore_index=True)

        print("Model",i,train, acc, epoch)
        print('Confusion matrix:')
        print('=================')
        print(np.array2string(cm, separator=', ', 
                               formatter={'int': lambda x: f'{x:5d}'},
                               prefix='[[', suffix=']]'))
        print('')
        print(f'F1 score: {f1:.4f}')

        cm_dataframe = pd.DataFrame(cm)
        cm_dataframe.to_csv("data/confusion_matrix/train_test_"+str(i)+".csv", index=True)


    print("Model with best f1: ", bf1, idx)
    print("Model with best accuracy: ", bacc, idx)

    #Save the best data model of the training/testing repeatings 
    dados.to_csv("data/Deepinsight_ViT_test.csv", index=True)


    return 0

if __name__ == '__main__':
    main()