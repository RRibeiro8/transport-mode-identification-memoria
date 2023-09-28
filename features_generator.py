import os
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm

from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pyDeepInsight import Norm2Scaler

import tsfel

#import tsfel

def plotSubGraph(data, ax):

	colors_list = ['Red','Orange', 'Blue', 'Purple']

	labels =  list(data['label'].unique())
	total = len(data)
	stats = pd.DataFrame(columns = ["label", "ndata", "percentage"])

	for l in labels:
		count = len(data.loc[(data['label'] == l)])
		stats = pd.concat([stats, pd.DataFrame({"label": [l], "ndata": [count], "percentage": [round((count/total)*100,2)]  })], ignore_index=True)

	graph = ax.bar(stats['label'], stats['ndata'], color = colors_list)

	i = 0
	for p in graph:
		width = p.get_width()
		height = p.get_height()
		x, y = p.get_xy()
		ax.text(x+width/2,
				 y+height*1.01,
				 str(stats.percentage[i])+'%',
				 ha='center',
				 weight='bold')
		i+=1

	return ax

class FeaturesGenerator:

	def __init__(self, df):
		self.data = df
		self.headers = ["traj_id", "lat", "long", "haversine", "duration", "speed", "acceleration", "jerk", "bearing", "b_rate","br_rate","stop_ratio", "label"]
		self.features = pd.DataFrame(columns = self.headers)
		self.dataset = pd.DataFrame()
		self.labels = []


	def data_distribution(self):

		fig,(ax1, ax2, ax3) = plt.subplots(1, 3)
		fig.suptitle('Data Distribution')

		ax1 = plotSubGraph(self.data, ax1)
		ax2 = plotSubGraph(self.features, ax2)
		ax3 = plotSubGraph(self.dataset, ax3)

		plt.show()


	# def all_domain_features(self):

	# 	traj_idx =  list(self.features['traj_id'].unique())  

	# 	cfg = tsfel.get_features_by_domain()

	# 	for traj in tqdm(traj_idx):

	# 		traj_df = self.features.loc[(self.features['traj_id'] == traj)].reset_index(drop=True)
	# 		traj_df = traj_df.drop(['traj_id', 'label'], axis=1)
	# 		print(traj_df)

	# 		X = tsfel.time_series_features_extractor(cfg, traj_df)

	# 		print(X)

	def resample(self, mode=None, filename=None):

		if self.data.empty:
			print("You have to load the dataset first!")
		else:

			traj_idx =  list(self.data['traj_id'].unique()) 

			new_id = 0

			new_data = pd.DataFrame()

			for traj in tqdm(traj_idx):

				traj_df = self.data.loc[(self.data['traj_id'] == traj)].sort_values(by=['datetime']).reset_index(drop=True)
				label = list(traj_df['label'].unique())[0]

				new_sample = pd.DataFrame()

				tmp = 0

				# if mode == "bearing":

				# 	for i in range(1, len(traj_df)):
				# 		startPos = (traj_df.loc[i-1, 'lat'], traj_df.loc[i-1, 'long'])
				# 		endPos = (traj_df.loc[i, 'lat'], traj_df.loc[i, 'long'])

				# 		tmp = self.computeBearing(startPos, endPos)
				# 		if tmp < 0: tmp += 360

				# 		print(traj, tmp)

				if mode == "time":

					startTime = traj_df.loc[0, 'datetime']

					split_df = pd.DataFrame()

					for i in range(1, len(traj_df)):

						
						endTime = traj_df.loc[i, 'datetime']
						tmp = self.durationTime(self.to_datetime(startTime), self.to_datetime(endTime))

						if tmp >= 1200: #20 minutos

							split_df = traj_df.loc[(traj_df['datetime'] >= startTime) & (traj_df['datetime'] < endTime)]

							if len(split_df) > 3:

								split_df['label'] = label
								split_df['traj_id'] = new_id

								new_id = new_id + 1

								new_sample = pd.concat([new_sample, split_df], ignore_index=True)

							startTime = endTime

				elif mode == "point":

					split_df = pd.DataFrame()

					for i in range(0, len(traj_df)):

						if i % 20 == 0 and i != 0:
							split_df = traj_df.iloc[(i-20):i]
							split_df['label'] = label
							split_df['traj_id'] = new_id

							new_id = new_id + 1
							new_sample = pd.concat([new_sample, split_df], ignore_index=True)
				else:
					print("A mode is needed! (time or point)")

				new_data = pd.concat([new_data, new_sample], ignore_index=True)

			self.data = new_data.reset_index(drop=True)
			self.data['traj_id'] = self.data['traj_id'].astype('int')

			print(self.data)

			if filename:
				self.data.to_csv(filename, index=False)

				# for i in range(1, len(traj_df)):

				# 	startPos = (traj_df.loc[i-1, 'lat'], traj_df.loc[i-1, 'long'])
				# 	endPos = (traj_df.loc[i, 'lat'], traj_df.loc[i, 'long'])

				# 	if mode == "bearing":

				# 		tmp = self.computeBearing(startPos, endPos)

				# 	elif mode == "time":

				# 		startTime = self.to_datetime(traj_df.loc[i-1, 'datetime'])
				# 		endTime = self.to_datetime(traj_df.loc[i, 'datetime'])
				# 		tmp = tmp + self.durationTime(startTime, endTime)



				# 	elif mode == "point":

				# 		tmp = i

				# print(("Total " + mode), tmp)


					#new_df = df.loc[(df['datetime'] >= st) & (df['datetime'] <= et)]


	def extractBasicFeatures(self):

		traj_idx =  list(self.data['traj_id'].unique())  

		for traj in tqdm(traj_idx):

			traj_df = self.data.loc[(self.data['traj_id'] == traj)].sort_values(by=['datetime']).reset_index(drop=True)
			
			if 'label' in traj_df.columns:
				label = list(traj_df['label'].unique())[0]
			else:
				label = "unknown"

			if len(traj_df) >= 4: 

				first = ({
						'traj_id': [traj_df['traj_id'].iloc[0]], 
						'lat': [traj_df['lat'].iloc[0]], 
						'long': [traj_df['long'].iloc[0]], 
						'haversine': [0.0], 
						'duration': [0.0],
						'speed': [0.0], 
						'acceleration': [0.0],
						'stop_ratio': [0.0],
						'jerk': [0.0],
						'bearing': [0.0],
						'b_rate': [0.0],
						'br_rate': [0.0],
						'label': [label]
				})

				features = pd.DataFrame(first)

				#features = pd.DataFrame()

				if label not in self.labels:
					self.labels.append(label)

				cout_points = 1
				stops = 0

				for idx in range(1, len(traj_df)):

					startPos = (traj_df.loc[idx-1, 'lat'], traj_df.loc[idx-1, 'long'])
					endPos = (traj_df.loc[idx, 'lat'], traj_df.loc[idx, 'long'])
					dist = self.haversineDistance(startPos, endPos)

					bearing = self.computeBearing(startPos, endPos)

					startTime = self.to_datetime(traj_df.loc[idx-1, 'datetime'])
					endTime = self.to_datetime(traj_df.loc[idx, 'datetime'])
					duration = self.durationTime(startTime, endTime)

					speed = 0
					if duration > 0:
						speed = self.computeSpeed(dist, duration)

					if speed < 0.6:
						stops = stops + 1

					#print(label, dist, duration, speed)
					stop_ratio = 0
					if stops > 0:
						stop_ratio = stops / idx

					new_row = pd.DataFrame({
								'traj_id': traj_df['traj_id'].iloc[idx],
								'lat': traj_df['lat'].iloc[idx], 
								'long': traj_df['long'].iloc[idx], 
								'haversine': dist, 
								'duration': duration, 
								'speed': speed,  
								'stop_ratio': stop_ratio,
								'acceleration': 0.0,
								'jerk': 0.0,
								'bearing': bearing,
								'b_rate': 0.0,
								'br_rate': 0.0,
								'label':label}, index=[0])

					features = pd.concat([features, new_row], ignore_index=True)

					cout_points = cout_points + 1

				for idx in range(1, len(features)):

					startSpeed = features.loc[idx-1, "speed"]
					endSpeed = features.loc[idx, "speed"]
					duration = features.loc[idx, "duration"]

					acc = 0
					if duration > 0:
						acc = self.computeAcceleration(startSpeed, endSpeed, duration)

					features.loc[idx, "acceleration"] = acc

					startAcc = features.loc[idx-1, "acceleration"]
					endAcc = features.loc[idx, "acceleration"]
					#duration = features.loc[idx, "duration"]
					jerk = 0
					if duration > 0:
						jerk = self.computeJerk(startAcc, endAcc, duration)
					features.loc[idx, "jerk"] = jerk

					startB = features.loc[idx-1, "bearing"]
					endB = features.loc[idx, "bearing"]

					bearing_rate = 0
					if duration > 0:
						bearing_rate = self.computeBearingRate(startB, endB, duration)
					features.loc[idx, "b_rate"] = bearing_rate

					startBr = features.loc[idx-1, "b_rate"]
					endBr = features.loc[idx, "b_rate"]

					br_rate = 0
					if duration > 0:
						br_rate = self.computeRateOfBearingRate(startBr, endBr, duration)
					features.loc[idx, "br_rate"] = br_rate

				self.features = pd.concat([self.features, features.iloc[1:]], ignore_index=True)
				del features

	def Normalize_features_col(self):

		#scaler = MinMaxScaler(feature_range=(0, 1))
		ln = Norm2Scaler()
		features = pd.DataFrame()

		traj_idx =  list(self.features['traj_id'].unique()) 
		#self.features.iloc[:,2:-1] = ln.fit_transform(self.features.iloc[:,2:-1].to_numpy())
		
		for traj in tqdm(traj_idx):

			traj_df = self.features.loc[(self.features['traj_id'] == traj)].reset_index(drop=True)
			label = list(traj_df['label'].unique())[0]

			traj_df.iloc[:,2:-1] = ln.fit_transform(traj_df.iloc[:,2:-1].to_numpy())

			#traj_df.iloc[:,2:-1] = scaler.fit_transform(traj_df.iloc[:,2:-1].to_numpy())

			features = pd.concat([features, traj_df], ignore_index=True)

		self.features = features

	def tsfel_features(self, signal, name):

		fs = 50
		features = {
			(name + "abs_energy"): [tsfel.feature_extraction.features.abs_energy(signal)],
			(name + "auc"): [tsfel.feature_extraction.features.auc(signal, fs)],
			(name + "autocorr"): [tsfel.feature_extraction.features.autocorr(signal)],
			(name + "entropy"): [tsfel.feature_extraction.features.entropy(signal, prob='standard')],
			(name + "f_frequency"): [tsfel.feature_extraction.features.fundamental_frequency(signal, fs)],
			(name + "hr_energy"): [tsfel.feature_extraction.features.human_range_energy(signal, fs)],
			(name + "kurtosis"): [tsfel.feature_extraction.features.kurtosis(signal)],
			(name + "max_freq"): [tsfel.feature_extraction.features.max_frequency(signal, fs)],
			(name + "mean_abs_deviation"): [tsfel.feature_extraction.features.mean_abs_deviation(signal)],
			(name + "mean_abs_diff"): [tsfel.feature_extraction.features.mean_abs_diff(signal)],
			(name + "mean_diff"): [tsfel.feature_extraction.features.mean_diff(signal)],
			(name + "median_abs_deviation"): [tsfel.feature_extraction.features.median_abs_deviation(signal)],
			(name + "median_abs_diff"): [tsfel.feature_extraction.features.median_abs_diff(signal)],
			(name + "median_diff"): [tsfel.feature_extraction.features.median_diff(signal)],
			(name + "median_frequency"): [tsfel.feature_extraction.features.median_frequency(signal, fs)],
			(name + "negative_turning"): [tsfel.feature_extraction.features.negative_turning(signal)],
			(name + "pk_pk_distance"): [tsfel.feature_extraction.features.pk_pk_distance(signal)],
			(name + "positive_turning"): [tsfel.feature_extraction.features.positive_turning(signal)],
			(name + "power_bandwidth"): [tsfel.feature_extraction.features.power_bandwidth(signal, fs)],
			(name + "rms"): [tsfel.feature_extraction.features.rms(signal)],
			(name + "skewness"): [tsfel.feature_extraction.features.skewness(signal)],
			(name + "slope"): [tsfel.feature_extraction.features.slope(signal)],
			(name + "spectral_centroid"): [tsfel.feature_extraction.features.spectral_centroid(signal, fs)],
			(name + "spectral_decrease"): [tsfel.feature_extraction.features.spectral_decrease(signal, fs)],
			(name + "spectral_distance"): [tsfel.feature_extraction.features.spectral_distance(signal, fs)],
			(name + "spectral_entropy"): [tsfel.feature_extraction.features.spectral_entropy(signal, fs)],
			(name + "spectral_kurtosis"): [tsfel.feature_extraction.features.spectral_kurtosis(signal, fs)],
			(name + "spectral_positive_turning"): [tsfel.feature_extraction.features.spectral_positive_turning(signal, fs)],
			(name + "spectral_roll_off"): [tsfel.feature_extraction.features.spectral_roll_off(signal, fs)],
			(name + "spectral_roll_on"): [tsfel.feature_extraction.features.spectral_roll_on(signal, fs)],
			(name + "spectral_skewness"): [tsfel.feature_extraction.features.spectral_skewness(signal, fs)],
			(name + "spectral_slope"): [tsfel.feature_extraction.features.spectral_slope(signal, fs)],
			(name + "spectral_spread"): [tsfel.feature_extraction.features.spectral_spread(signal, fs)],
			(name + "spectral_variation"): [tsfel.feature_extraction.features.spectral_variation(signal, fs)],
			(name + "sum_abs_diff"): [tsfel.feature_extraction.features.sum_abs_diff(signal)],
			(name + "total_energy"): [tsfel.feature_extraction.features.total_energy(signal, fs)],
			(name + "wavelet_abs_mean"): [np.average(list(tsfel.feature_extraction.features.wavelet_abs_mean(signal)))],
			(name + "wavelet_energy"): [np.average(list(tsfel.feature_extraction.features.wavelet_energy(signal)))],
			(name + "wavelet_entropy"): [tsfel.feature_extraction.features.wavelet_entropy(signal)],
			(name + "wavelet_std"): [np.average(list(tsfel.feature_extraction.features.wavelet_std(signal)))],
			(name + "wavelet_var"): [np.average(list(tsfel.feature_extraction.features.wavelet_var(signal)))],
			}

		return features

	def extractFeatures(self):

		if self.features.empty:
			print("You have to extract basic features first!")
		else:

			traj_idx =  list(self.features['traj_id'].unique()) 
			headers = ["harversine", "h_min", "h_max", "h_avg", "h_median", "h_std", "h_quantile", "h_25", "h_50", "h_75", "h_80", "h_90", 
					"straight", "ratio", 
					"vel_min", "vel_max", "vel_avg", "vel_median", "vel_std", "vel_quantile", "vel_25", "vel_50", "vel_75", "vel_80", "vel_90",
					"vel_top1", "vel_top2", "vel_top3", 
					"acc_min", "acc_max", "acc_avg", "acc_median", "acc_std", "acc_quantile", "acc_25", "acc_50", "acc_75", "acc_80", "acc_90",
					"acc_top1", "acc_top2", "acc_top3",
					"jerk_min", "jerk_max", "jerk_avg", "jerk_median", "jerk_std", "jerk_quantile", "jerk_25", "jerk_50", "jerk_75", "jerk_80", "jerk_90",
					"jerk_top1", "jerk_top2", "jerk_top3",
					"bearing_min", "bearing_max", "bearing_avg", "bearing_median", "bearing_std", "bearing_quantile", "bearing_25", "bearing_50", "bearing_75", "bearing_80", "bearing_90",
					"bearing_top1", "bearing_top2", "bearing_top3",
					"br_min", "br_max", "br_avg", "br_median", "br_std", "br_quantile", "br_25", "br_50", "br_75", "br_80", "br_90",
					"br_top1", "br_top2", "br_top3",
					"brr_min", "brr_max", "brr_avg", "brr_median", "brr_std", "brr_quantile", "brr_25", "brr_50", "brr_75", "brr_80", "brr_90",
					"brr_top1", "brr_top2", "brr_top3",
					"duration", "d_min", "d_max", "d_avg", "d_median", "d_std", "d_quantile", "d_25", "d_50", "d_75", "d_80", "d_90",
					"area", "stop_ratio","label"]
			features = pd.DataFrame()

			for traj in tqdm(traj_idx):

				all_features = {}

				traj_df = self.features.loc[(self.features['traj_id'] == traj)].reset_index(drop=True)
				
				
				if 'label' in traj_df.columns:
					label = list(traj_df['label'].unique())[0]
				else:
					label = "unknown"

				polygon = self.computePolygon(traj_df)
				area = polygon.area

				straight = self.computeLineDist(traj_df)

				dh = traj_df['haversine'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")

				fdict = self.tsfel_features(traj_df['haversine'], "h_")
				all_features.update(fdict)

				harvesine = traj_df['haversine'].sum()

				ratio = float(harvesine/straight)

				ds = traj_df['speed'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_speed = traj_df['speed'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['speed'], "vel_")
				all_features.update(fdict) 

				da = traj_df['acceleration'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_acc = traj_df['acceleration'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['acceleration'], "acc_")
				all_features.update(fdict)

				dj = traj_df['jerk'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_jerk = traj_df['jerk'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['jerk'], "jerk_")
				all_features.update(fdict)

				db = traj_df['bearing'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_bearing = traj_df['bearing'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['bearing'], "bearing_")
				all_features.update(fdict)

				dbr = traj_df['b_rate'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_br = traj_df['b_rate'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['b_rate'], "b_rate_")
				all_features.update(fdict)

				dbrr = traj_df['br_rate'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				top_brr = traj_df['br_rate'].nlargest(3).values.flatten()

				fdict = self.tsfel_features(traj_df['br_rate'], "br_rate_")
				all_features.update(fdict)

				dd = traj_df['duration'].describe(percentiles=[.25, .5, .75, .8, .9], include="all")
				sr = traj_df['stop_ratio'].iat[-1]

				fdict = {
					"harversine": harvesine,
					"h_min": dh["min"],  
					"h_max": dh["max"], 
					"h_avg": dh["mean"], 
					"h_median": traj_df['haversine'].median(), 
					"h_std": dh["std"], 
					"h_quantile": traj_df['haversine'].quantile(0.85),
					"h_25": dh["25%"],
					"h_50": dh["50%"],
					"h_75": dh["75%"],
					"h_80": dh["80%"],
					"h_90": dh["90%"],
					"straight": straight, 
					"ratio":ratio, 
					"vel_min": ds["min"],
					"vel_max": ds["max"], 
					"vel_avg": ds["mean"], 
					"vel_median": traj_df['speed'].median(), 
					"vel_std": ds["std"], 
					"vel_quantile": traj_df['speed'].quantile(0.85),
					"vel_25": ds["25%"],
					"vel_50": ds["50%"],
					"vel_75": ds["75%"],
					"vel_80": ds["80%"],
					"vel_90": ds["90%"],
					"vel_top1": top_speed[0],
					"vel_top2": top_speed[1],
					"vel_top3": top_speed[2],
					"acc_min": da["min"],
					"acc_max": da["max"], 
					"acc_avg": da["mean"], 
					"acc_median": da.median(), 
					"acc_std": da["std"], 
					"acc_quantile": da.quantile(0.85),
					"acc_25": da["25%"],
					"acc_50": da["50%"],
					"acc_75": da["75%"],
					"acc_80": da["80%"],
					"acc_90": da["90%"],
					"acc_top1": top_acc[0],
					"acc_top2": top_acc[1],
					"acc_top3": top_acc[2],
					"jerk_min": dj["min"],
					"jerk_max": dj["max"], 
					"jerk_avg": dj["mean"], 
					"jerk_median": traj_df['jerk'].median(), 
					"jerk_std": dj["std"], 
					"jerk_quantile": traj_df['jerk'].quantile(0.85),
					"jerk_25": dj["25%"],
					"jerk_50": dj["50%"],
					"jerk_75": dj["75%"],
					"jerk_80": dj["80%"],
					"jerk_90": dj["90%"],
					"jerk_top1": top_jerk[0],
					"jerk_top2": top_jerk[1],
					"jerk_top3": top_jerk[2],
					"bearing_min": db["min"],
					"bearing_max": db["max"], 
					"bearing_avg": db["mean"], 
					"bearing_median": traj_df['bearing'].median(), 
					"bearing_std": db["std"], 
					"bearing_quantile": traj_df['bearing'].quantile(0.85),
					"bearing_25": db["25%"],
					"bearing_50": db["50%"],
					"bearing_75": db["75%"],
					"bearing_80": db["80%"],
					"bearing_90": db["90%"],
					"bearing_top1": top_bearing[0],
					"bearing_top2": top_bearing[1],
					"bearing_top3": top_bearing[2],
					"br_min": dbr["min"],
					"br_max": dbr["max"], 
					"br_avg": dbr["mean"], 
					"br_median": traj_df['b_rate'].median(), 
					"br_std": dbr["std"], 
					"br_quantile": traj_df['b_rate'].quantile(0.85),
					"br_25": dbr["25%"],
					"br_50": dbr["50%"],
					"br_75": dbr["75%"],
					"br_80": dbr["80%"],
					"br_90": dbr["90%"],
					"br_top1": top_br[0],
					"br_top2": top_br[1],
					"br_top3": top_br[2],
					"brr_min": dbrr["min"],
					"brr_max": dbrr["max"], 
					"brr_avg": dbrr["mean"], 
					"brr_median": traj_df['br_rate'].median(), 
					"brr_std": dbrr["std"], 
					"brr_quantile": traj_df['br_rate'].quantile(0.85),
					"brr_25": dbrr["25%"],
					"brr_50": dbrr["50%"],
					"brr_75": dbrr["75%"],
					"brr_80": dbrr["80%"],
					"brr_90": dbrr["90%"],
					"brr_top1": top_brr[0],
					"brr_top2": top_brr[1],
					"brr_top3": top_brr[2],
					"duration": dd.sum(), 
					"d_min": dd["min"],
					"d_max": dd["max"], 
					"d_avg": dd["mean"], 
					"d_median": traj_df['duration'].median(), 
					"d_std": dd["std"], 
					"d_quantile": traj_df['duration'].quantile(0.85),
					"d_25": dd["25%"],
					"d_50": dd["50%"],
					"d_75": dd["75%"],
					"d_80": dd["80%"],
					"d_90": dd["90%"],
					"area": area, 
					"stop_ratio": sr,
					"label": label}

				all_features.update(fdict)
				#print(len(all_features))
				new_row = pd.DataFrame(all_features, index=[0])

				#print(new_row)

				new_row.replace([np.inf, -np.inf], np.nan, inplace=True)

				if(not new_row.isnull().values.any()):					
					features = pd.concat([features, new_row], ignore_index=True)

				del new_row

			self.dataset = features

	def extractFeaturesOLD(self):

		if self.features.empty:
			print("You have to extract basic features first!")
		else:

			traj_idx =  list(self.features['traj_id'].unique()) 
			headers = ["harversine", "h_min", "h_max", "h_avg", "h_median", "h_std", "h_quantile", 
					"straight", "ratio", 
					"vel_min", "vel_max", "vel_avg", "vel_median", "vel_std", "vel_quantile", 
					"vel_top1", "vel_top2", "vel_top3", 
					"acc_min", "acc_max", "acc_avg", "acc_median", "acc_std", "acc_quantile",
					"acc_top1", "acc_top2", "acc_top3",
					"jerk_min", "jerk_max", "jerk_avg", "jerk_median", "jerk_std", "jerk_quantile",
					"jerk_top1", "jerk_top2", "jerk_top3",
					"bearing_min", "bearing_max", "bearing_avg", "bearing_median", "bearing_std", "bearing_quantile",
					"bearing_top1", "bearing_top2", "bearing_top3",
					"br_min", "br_max", "br_avg", "br_median", "br_std", "br_quantile",
					"br_top1", "br_top2", "br_top3",
					"brr_min", "brr_max", "brr_avg", "brr_median", "brr_std", "brr_quantile",
					"brr_top1", "brr_top2", "brr_top3",
					"duration", "d_min", "d_max", "d_avg", "d_median", "d_std", "d_quantile",
					"area", "stop_ratio","label"]
			features = pd.DataFrame(columns = headers)

			

			for traj in tqdm(traj_idx):

				traj_df = self.features.loc[(self.features['traj_id'] == traj)].reset_index(drop=True)
				label = list(traj_df['label'].unique())[0]

				polygon = self.computePolygon(traj_df)
				area = polygon.area

				straight = self.computeLineDist(traj_df)

				dh = traj_df['haversine']

				harvesine = dh.sum()

				ratio = float(harvesine/straight)

				ds = traj_df['speed']
				top_speed = ds.nlargest(3).values.flatten()

				da = traj_df['acceleration']
				top_acc = da.nlargest(3).values.flatten()

				dj = traj_df['jerk']
				top_jerk = dj.nlargest(3).values.flatten()

				db = traj_df['bearing']
				top_bearing = db.nlargest(3).values.flatten()

				dbr = traj_df['b_rate']
				top_br = dbr.nlargest(3).values.flatten()

				dbrr = traj_df['br_rate']
				top_brr = dbrr.nlargest(3).values.flatten()


				dd = traj_df['duration']
				sr = traj_df['stop_ratio'].iat[-1]

				new_row = pd.DataFrame({
									"harversine": harvesine,
									"h_min": dh.min(),  
									"h_max": dh.max(), 
									"h_avg": dh.mean(), 
									"h_median": dh.median(), 
									"h_std": dh.std(), 
									"h_quantile": dh.quantile(0.85),
									"straight": straight, 
									"ratio":ratio, 
									"vel_min": ds.min(),
									"vel_max": ds.max(), 
									"vel_avg": ds.mean(), 
									"vel_median": ds.median(), 
									"vel_std": ds.std(), 
									"vel_quantile": ds.quantile(0.85),
									"vel_top1": top_speed[0],
									"vel_top2": top_speed[1],
									"vel_top3": top_speed[2],
									"acc_min": da.min(),
									"acc_max": da.max(), 
									"acc_avg": da.mean(), 
									"acc_median": da.median(), 
									"acc_std": da.std(), 
									"acc_quantile": da.quantile(0.85),
									"acc_top1": top_acc[0],
									"acc_top2": top_acc[1],
									"acc_top3": top_acc[2],
									"jerk_min": dj.min(),
									"jerk_max": dj.max(), 
									"jerk_avg": dj.mean(), 
									"jerk_median": dj.median(), 
									"jerk_std": dj.std(), 
									"jerk_quantile": dj.quantile(0.85),
									"jerk_top1": top_jerk[0],
									"jerk_top2": top_jerk[1],
									"jerk_top3": top_jerk[2],
									"bearing_min": db.min(),
									"bearing_max": db.max(), 
									"bearing_avg": db.mean(), 
									"bearing_median": db.median(), 
									"bearing_std": db.std(), 
									"bearing_quantile": db.quantile(0.85),
									"bearing_top1": top_bearing[0],
									"bearing_top2": top_bearing[1],
									"bearing_top3": top_bearing[2],
									"br_min": dbr.min(),
									"br_max": dbr.max(), 
									"br_avg": dbr.mean(), 
									"br_median": dbr.median(), 
									"br_std": dbr.std(), 
									"br_quantile": dbr.quantile(0.85),
									"br_top1": top_br[0],
									"br_top2": top_br[1],
									"br_top3": top_br[2],
									"brr_min": dbrr.min(),
									"brr_max": dbrr.max(), 
									"brr_avg": dbrr.mean(), 
									"brr_median": dbrr.median(), 
									"brr_std": dbrr.std(), 
									"brr_quantile": dbrr.quantile(0.85),
									"brr_top1": top_brr[0],
									"brr_top2": top_brr[1],
									"brr_top3": top_brr[2],
									"duration": dd.sum(), 
									"duration": dd.sum(), 
									"d_min": dh.min(),
									"d_max": dh.max(), 
									"d_avg": dd.mean(), 
									"d_median": dd.median(), 
									"d_std": dd.std(), 
									"d_quantile": dd.quantile(0.85),
									"area": area, 
									"stop_ratio": sr,
									"label": label}, index=[0])

				new_row.replace([np.inf, -np.inf], np.nan, inplace=True)

				if(not new_row.isnull().values.any()):					
					features = pd.concat([features, new_row], ignore_index=True)

				del new_row

			self.dataset = features


	def computeBearing(self, startPos, endPos):
		long1, lat1, long2, lat2 = map(np.radians, [startPos[1], startPos[0], endPos[1], endPos[0]])
		dLon = (long2 - long1)
		x = np.cos(lat2) * np.sin(dLon)
		y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
		brng = np.arctan2(x,y)
		brng = np.degrees(brng)

		return brng

	def computeBearingRate(self, startBearing, endBearing, duration):
		return (endBearing - startBearing) / duration

	def computeRateOfBearingRate(self, startBearing, endBearing, duration):
		return (endBearing - startBearing) / duration

	def computeLineDist(self, df):
		start  = np.array([df['lat'].iloc[0], df['long'].iloc[0]])
		end = np.array([df['lat'].iloc[-1], df['long'].iloc[-1]])
		ds = end - start
		return np.sqrt(np.dot(ds,ds))

					
	def computePolygon(self, points):

		bounds = [points['lat'].min(), points['long'].min(), points['lat'].max(), points['long'].max()]

		coords = ((bounds[0], bounds[3]), (bounds[2], bounds[3]),
				(bounds[2], bounds[1]), (bounds[0], bounds[1])) 
		
		return Polygon(coords)

	def computeSpeed(self, distance, duration):
		return (distance / duration)

	def computeAcceleration(self, startSpeed, endSpeed, duration):
		return (endSpeed - startSpeed) / duration

	def computeJerk(self, startAcc, endAcc, duration):
		return (endAcc - startAcc) / duration

	def durationTime(self, startTime, endTime):
		return (endTime - startTime).total_seconds()

	def haversineDistance(self, startPos, endPos):
		"""
		Calculate the great circle distance between two points
		on the earth (specified in decimal degrees)
		All args must be of equal length.    
		"""

		radius = self.earthRad(endPos[0])

		lon1, lat1, lon2, lat2 = map(np.radians, [startPos[1], startPos[0], endPos[1], endPos[0]])

		dlon = lon2 - lon1
		dlat = lat2 - lat1

		a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
		c = 2 * np.arcsin(np.sqrt(a))
		distance = radius * c
		return distance

	def earthRad(self, lat):
		'''
		Calculates the Earth's radius (in m) at a given latitude using an ellipsoidal model. Major/minor axes from NASA
		'''
		a = 6378137
		b = 6356752
		lat = np.radians(lat)
		g = (a**2*np.cos(lat))**2 + (b**2*np.sin(lat))**2
		f = (a*np.cos(lat))**2 + (b*np.sin(lat))**2
		radius = np.sqrt(g/f)
		return radius


	def save2csv(self, filename):
		if self.dataset.empty:
			self.features.to_csv(filename, index=False)
		else:
			self.dataset.to_csv(filename, index=False)

	def readcsv(self, filename, filename2=None):

		if filename2:
			self.dataset = pd.read_csv(filename2)
		self.features = pd.read_csv(filename)
		self.headers = self.features.head() 
		self.labels =  list(self.features['label'].unique())

	def to_datetime(self, string):
		return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


	def random_split(self, n):	

		new_features = pd.DataFrame()
		for l in self.labels:

			df = self.features.loc[(self.features['label'] == l)]
			df =  df.sample(n=n, ignore_index=True)
			new_features = pd.concat([new_features, df], ignore_index=True)

		self.features = new_features.reset_index(drop=True)

	def dropFeatures(self, features):
		for f in features:
			self.dataset = self.dataset.drop([f], axis=1)
		self.dataset = self.dataset.reset_index(drop=True)

	def createSamplesCSV(self, n, filename):	

		new_features = pd.DataFrame()
		for l in self.labels:

			df = self.dataset.loc[(self.dataset['label'] == l)]
			df =  df.sample(n=n, ignore_index=True)
			new_features = pd.concat([new_features, df], ignore_index=True)

		new_features.to_csv(filename, index=False)