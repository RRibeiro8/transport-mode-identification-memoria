import numpy as np
import pandas as pd

from pyDeepInsight import ImageTransformer, Norm2Scaler

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



class ImageEmbedding:

    def __init__(self, features, reducer=None, size, generator=None):

        self.features = features
        self.generator = generator
        self.X = np.empty(0)
        self.Y = np.empty(0)
        self.reducer = reducer
        self.size = size
        self.it = ImageTransformer(feature_extractor=reducer, pixels=size, discretization="assignment")
        #self.coords = pd.DataFrame()
        self.coords = np.empty(0)
        self.num_classes = 0
        self.classes = []
        self.labels = pd.DataFrame()

    def label_encoder(self, Y):
        encoder = LabelEncoder()
        y = encoder.fit_transform(Y)
        print(y)
        self.num_classes = np.unique(y).size

        return y, encoder

    def featureExpansion(self, X):
        return self.generator.fit_transform(X)

    def fit_embeddingByClasses(self):

        ln = Norm2Scaler()
        self.classes = list(self.features['label'].unique())

        labels = pd.DataFrame()

        for label in self.classes:

            df = self.features.loc[(self.features['label'] == label)]

            X = np.array(df.drop(['label'], axis=1))
            dl = self.features['label']
            dl = dl.loc[(self.features['label'] == label)]

            labels = pd.concat([labels, dl], ignore_index=True)

            if self.generator:
                X = self.featureExpansion(X)
            self.it.fit(X, plot=False)

            X_norm = ln.fit_transform(X)

            X_img = self.it.transform(X_norm)

            coords = pd.DataFrame(self.it._coords, columns = ['x','y'])

            coords['label'] = label
            self.coords = pd.concat([self.coords, coords], ignore_index=True)

            print(self.X.shape, X_img.shape)

            if np.any(self.X):
                self.X = np.append(self.X, X_img, axis=0)
            else:
                self.X = X_img
        
        y = np.array(labels)
        self.Y = self.label_encoder(y)



    def fit_embedding(self, plot=False):

        ln = Norm2Scaler()
        #ln = MinMaxScaler(feature_range=(0, 255))
        self.classes =  list(self.features['label'].unique())

        X = np.array(self.features.drop(['label'], axis=1))
        y = np.array(self.features['label'])

        self.Y, encoder= self.label_encoder(y)

        self.labels = pd.concat([pd.DataFrame(self.Y), self.features['label']], ignore_index=True, axis=1).drop_duplicates().reset_index(drop=True)

        if self.generator:
            X = self.featureExpansion(X)
        self.it.fit(X, plot=plot)

        X_norm = ln.fit_transform(X)

        #print(X_norm)

        self.X = self.it.transform(X_norm)

        if plot:

            fig, ax = plt.subplots(1, 5, figsize=(15, 5))
            labels_list = []
            ly = encoder.inverse_transform(self.Y)
            counter = 0
            for i in range(500,len(self.X)):

                if len(labels_list) > self.num_classes:
                    break

                if ly[i] not in labels_list:
                    print(ly[i])
                    print(self.X[i].shape)
                    ax[counter].imshow(self.X[i])
                    ax[counter].title.set_text(f"{ly[i]}")
                    labels_list.append(ly[i])
                    counter = counter + 1

                

            plt.tight_layout()

            plt.show()

        self.coords = self.it._coords    