import numpy as np
import csv
from visual import save_mean, save_mean_2d, euclidean_distance, euclidean_data


all_output = csv.reader(open("./means/handIDall.csv", "r"), delimiter=",")
all_output = list(all_output)
all_output = np.array(all_output).astype("float")

means = csv.reader(open("./means/handIDmean.csv", "r"), delimiter=",")
means = list(means)
means = np.array(means).astype("float")

avg_distance = np.zeros(means.shape[0])
max_distance = np.zeros(means.shape[0])

average_individual = []

k = 0
for i, output in enumerate(all_output):
	# check if to switch
	config = [158, 326, 481, 642, 800]
		
	mean = means[k,:]
	distance = output - mean
	average_individual.append(distance)
	
	if i == config[k] - 1:
		avg_distance[k], max_distance[k] = euclidean_data(average_individual)
		k += 1
		average_individual = []

avg_distance[k], max_distance[k] = euclidean_data(average_individual, axis=0)
save_mean_2d(avg_distance, "average_distance")
save_mean_2d(max_distance, "maximum_distance")
print("mean saved")