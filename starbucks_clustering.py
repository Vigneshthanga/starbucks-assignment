from sklearn.cluster import AgglomerativeClustering
import math
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

points = np.array([[1,5], [1,7], [2,6], [2,9], [3,6], [3,8], [3,8], [3,9], [4,8], [4,8], [3,3], [5,4], [7,2], [5,7], [4,5], [6,9], [7,3], [7,9], [8,1], [8,7]])

#This is the condensed matrix of distance between the points:
'''dc = pdist(points)
sf = squareform(dc)
print('This is the distance condensed matrix:')
#print(dc)

print(sf)
'''

for i in range(0, len(points)):
	P = points[i]
	j = i
	while (j<len(points)):
		Q = points[j]
		#print(str(P)+" "+str(Q))
		print(round(math.sqrt(math.pow((P[0]-Q[0]),2) + math.pow((P[1]-Q[1]),2)),2))
		j+=1



x = np.array([3,2,5,6,5,12,10,8,7,2,5])
y = np.array([7,6,8,6,5,8,6,4,3,2,2])

plt.scatter(x, y)
fig = plt.gcf()
fig.canvas.set_window_title('Scatter Plot of points')
#plt.show()

#Fitting the data in Agg Clustering, can use this in future.
clustering = AgglomerativeClustering().fit(points)

AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='single', memory=None, n_clusters=1,
                        pooling_func='deprecated')

S= hierarchy.linkage(dc, 'single')
sdn = hierarchy.dendrogram(S)
fig = plt.gcf()
fig.canvas.set_window_title('Single-Linkage Clustering')
#plt.show()

A = hierarchy.linkage(dc, 'average')
adn = hierarchy.dendrogram(A)
fig = plt.gcf()
fig.canvas.set_window_title('Average Linkage Clustering')
#plt.show()

C = hierarchy.linkage(dc, 'complete')
cdn = hierarchy.dendrogram(C)
fig = plt.gcf()
fig.canvas.set_window_title('Complete Linkage Clustering')
#plt.show()

hierarchy.set_link_color_palette(None)  # reset to default after use
