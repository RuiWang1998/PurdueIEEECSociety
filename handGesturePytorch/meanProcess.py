import numpy as np
import csv
from visual import save_mean, save_mean_2d

def euclidean_distance(matrix):
	
	euclidean_distance_vector = np.sum(np.square(matrix), axis=1)

	return euclidean_distance_vector

def euclidean_data(matrix):

	euclidean_avg = np.mean(euclidean_distance(np.asarray(matrix)))
	euclidean_max = np.max(euclidean_distance(np.asarray(matrix)))

	return euclidean_avg, euclidean_max

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
	if i % 160 == 0 and i != 0:
		avg_distance[k], max_distance[k] = euclidean_data(average_individual)
		k += 1
		average_individual = []
		
	mean = means[k,:]
	distance = output - mean
	average_individual.append(distance)

avg_distance[k], max_distance[k] = euclidean_data(average_individual)
save_mean_2d(avg_distance, "average_distance")
save_mean_2d(max_distance, "maximum_distance")
print("mean saved")