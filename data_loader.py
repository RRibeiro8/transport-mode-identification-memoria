
import os
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from pykalman import KalmanFilter
from hampel import hampel
import matplotlib.pyplot as plt
import time

class DataLoader:

	def __init__(self, dataset, path):
		self.headers = ['traj_id', 'lat', 'long', 'altitude','datetime', 'label']
		self.path = path
		self.dataset = dataset
		self.labels = [] #List of Unique labels
		self.data = pd.DataFrame(columns = self.headers)

	def plotTrajectory(self, idx):

		trip = self.data[self.data.traj_id == idx][["long", "lat"]]
		label = list(self.data[self.data.traj_id == idx]['label'].unique())[0]
		plt.plot(trip.long, trip.lat, 'bo')
		plt.title(label)
		plt.show()

	def loadData(self):

		labels_path = "labels.txt"

		count_traj = 0
		traj_idx = 0
		for folder in tqdm(os.listdir(self.path)):

			subdir = self.path + folder + '/'
			tdir = subdir + 'Trajectory/'

			if os.path.isdir(tdir):

				if labels_path in os.listdir(subdir):
					df_labels = self.load_labels(subdir + labels_path).drop_duplicates().reset_index(drop=True)

					count_traj = count_traj + len(df_labels)

					df = pd.DataFrame()
					for file in os.listdir(tdir):
						df = pd.concat([df, self.load_files(tdir+file)], ignore_index=True)

					#print(df_labels)

					for i in range(len(df_labels)):

						new_df = pd.DataFrame(columns = ['traj_id', 'start_time', 'end_time', 'label'])

						st = df_labels.loc[i, 'start_time']
						et = df_labels.loc[i, 'end_time']
						label = df_labels.loc[i, 'label']

						new_df = df.loc[(df['datetime'] >= st) & (df['datetime'] <= et)]

						if len(new_df) > 2:

							new_df['label'] = label
							new_df['traj_id'] = traj_idx

							traj_idx = traj_idx + 1

							self.data = pd.concat([self.data, new_df], ignore_index=True)

						del new_df
					del df

		print("Total Trajectories: ", count_traj)
		print("Counted Trajectories: ", traj_idx)
		self.data = self.data.drop_duplicates().reset_index(drop=True)
		self.data['traj_id'] = self.data['traj_id'].astype('int')
		self.labels = list(self.data['label'].unique())
		

	def load_labels(self, filename):
		df = pd.read_csv(filename, sep='\t')
		df['start_time'] = df['Start Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
		df['end_time'] = df['End Time'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
		df['label'] = df['Transportation Mode']
		df = df.drop(['End Time', 'Start Time', 'Transportation Mode'], axis=1)
		return df

	# Read data from a file and convert to Dataframe with date, time and timestamp in single column
	def load_files(self, filename):
		df = pd.read_csv(filename, skiprows = 6, names=['lat', 'long', 'null', 
						'altitude','timestamp_float', 
						'date', 'time'])
		df['datetime'] = df.apply(lambda z: self.to_datetime(z.date + ' ' + z.time), axis=1)		
		df = df.drop(['null', 'timestamp_float', 'date', 'time'], axis=1)

		return df

	def to_datetime(self, string):
		return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

	def save2csv(self, filename):
		self.data.to_csv(filename, index=False)

	def readcsv(self, filename):
		self.data = pd.read_csv(filename)
		self.headers = self.data.head() 
		self.labels =  list(self.data['label'].unique())

	def dropLabels(self, labels):
		for l in labels:
			self.data.drop(self.data[self.data['label'] == l].index, inplace=True)

		self.data = self.data.reset_index(drop=True)
		self.labels = list(self.data['label'].unique())

	def changeLabels(self, label, newlabel):
		self.data.loc[(self.data['label'] == label), 'label'] = newlabel
		self.labels = list(self.data['label'].unique())

	def hampelFilter(self):

		if self.data.empty:
			print("You have to load the dataset first!")
		else:

			traj_idx =  list(self.data['traj_id'].unique()) 

			for traj in tqdm(traj_idx):

				traj_df = self.data.loc[(self.data['traj_id'] == traj)].reset_index(drop=True)
				label = list(traj_df['label'].unique())[0]

				longitude = traj_df['long'].squeeze()
				#print(len(longitude))
				latitude = traj_df['lat'].squeeze()
				#print(len(latitude))

				ts_lat = hampel(latitude, window_size=5, n=3, imputation=True)
				ts_long = hampel(longitude, window_size=5, n=3, imputation=True)
				
				#plt.figure(1)
				#plt.plot(longitude, latitude, 'bo', ts_long, ts_lat, 'r--')
				#plt.show()

				# print(measurements, smoothed_state_means)

				#print("Smothed: ", smoothed_state_means.shape)

				self.data.loc[(self.data['traj_id'] == traj), 'long'] = np.array(ts_long)
				self.data.loc[(self.data['traj_id'] == traj), 'lat'] = np.array(ts_lat)
				#print(self.data.loc[(self.data['traj_id'] == traj), 'lat'] )
				#print(self.data.loc[(self.data['traj_id'] == traj), 'long'])


	def kalmanFilter(self):

		if self.data.empty:
			print("You have to load the dataset first!")
		else:

			traj_idx =  list(self.data['traj_id'].unique()) 

			for traj in tqdm(traj_idx):

				traj_df = self.data.loc[(self.data['traj_id'] == traj)].reset_index(drop=True)
				label = list(traj_df['label'].unique())[0]

				#print(traj_df)

				points = []

				for i in range(len(traj_df)):
					points.append((traj_df.loc[i, 'long'], traj_df.loc[i, 'lat']))

				del traj_df

				measurements = np.asarray(points)

				#print("Init: ", measurements.shape)

				initial_state_mean = [measurements[0, 0],
									  0,
									  measurements[0, 1],
									  0]

				transition_matrix = [[1, 1, 0, 0],
									 [0, 1, 0, 0],
									 [0, 0, 1, 1],
									 [0, 0, 0, 1]]

				observation_matrix = [[1, 0, 0, 0],
									  [0, 0, 1, 0]]

				kf1 = KalmanFilter(transition_matrices = transition_matrix,
								  observation_matrices = observation_matrix,
								  initial_state_mean = initial_state_mean)
				
				kf1 = kf1.em(measurements, n_iter=5)
				#(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
				(smoothed_state_means, smoothed_state_covariances) = kf1.filter(measurements)

				# kf2 = KalmanFilter(transition_matrices = transition_matrix,
				#   observation_matrices = observation_matrix,
				#   initial_state_mean = initial_state_mean,
				#   observation_covariance = 10*kf1.observation_covariance,
				#   em_vars=['transition_covariance', 'initial_state_covariance'])

				# kf2 = kf2.em(measurements, n_iter=5)
				# #(smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
				# (smoothed_state_means, smoothed_state_covariances)  = kf2.filter(measurements)

				#plt.figure(1)
				#times = range(measurements.shape[0])
				#plt.plot(measurements[:, 0], measurements[:, 1], 'bo', smoothed_state_means[:, 0], smoothed_state_means[:, 2], 'r--')
				# 		 times, measurements[:, 1], 'ro',
				# 		 times, smoothed_state_means[:, 0], 'b--',
				# 		 times, smoothed_state_means[:, 2], 'r--',)
				#plt.show()

				self.data.loc[(self.data['traj_id'] == traj), 'long'] = smoothed_state_means[:, 0]
				self.data.loc[(self.data['traj_id'] == traj), 'lat'] = smoothed_state_means[:, 2]

