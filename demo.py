import os

import data_loader
from features_generator import FeaturesGenerator

from pyDeepInsight import Norm2Scaler

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

from archs.vit import DeepViTModel

def transform2Image(X: np.ndarray, size) -> np.ndarray:
	
	coords_file = 'pre-trained/BestDeepViT_coords.npy'
	np_coords = np.load(coords_file)

	img_coords = pd.DataFrame(np.vstack((
		np_coords.T,
		X
	)).T).groupby([0, 1], as_index=False).mean()

	img_list = []
	blank_mat = np.zeros(size)
	
	for z in range(2, img_coords.shape[1]):
		img_matrix = blank_mat.copy()
		img_matrix[img_coords[0].astype(int),
				   img_coords[1].astype(int)] = img_coords[z]
		
		img_list.append(img_matrix)

	
	img_matrices = np.array([np.repeat(m[:, :, np.newaxis], 3, axis=2)  for m in img_list])

	return img_matrices


def main():

	loader = data_loader.DataLoader("GeoLife Trajectories", "dataset/Data/")
	loader.readcsv("data/dataset.csv")

	generator = FeaturesGenerator(loader.data)
	generator.readcsv("data/basic_features.csv", "data/full_features.csv")


	for l in ['run', 'motorcycle', 'boat', 'taxi', 'subway', 'airplane']: 
		generator.dataset.drop(generator.dataset[generator.dataset['label'] == l].index, inplace=True)
	
	generator.dataset = generator.dataset.reset_index(drop=True)
	generator.labels = list(generator.dataset['label'].unique())

	size = (64, 64)

	### Fitting normalization from the trained data to transform new data to predict transport mode
	ln = Norm2Scaler()
	X_dataset = np.array(generator.dataset.drop(['label'], axis=1))
	ln = ln.fit(X_dataset)

	### Testing in data not trained ####

	data = pd.read_csv("data/unkown_data.csv")

	new_gen = FeaturesGenerator(data)
	new_gen.extractBasicFeatures()
	new_gen.extractFeatures()

	X = np.array(new_gen.dataset.drop(['label'], axis=1))
	X_norm = ln.transform(X)


	imgs =  transform2Image(X_norm, size)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	transform_test = transforms.Compose([
		transforms.ToTensor(),
	])

	X_tensor = torch.stack([transform_test(img) for img in imgs]).float().to(device)

	model = DeepViTModel()

	checkpoint = torch.load("pre-trained/BestDeepViT.pth")

	model.load_state_dict(checkpoint['model'], strict=False)
	model = model.to(device)

	model.eval()

	with torch.no_grad():

		output = model(X_tensor)

		prob = torch.nn.functional.softmax(output)

		top_p, top_class = prob.topk(3)

	print(top_p, top_class)

	return 0

if __name__ == '__main__':
	main()