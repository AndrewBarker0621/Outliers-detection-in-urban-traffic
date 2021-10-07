import nmf
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_clusters = 300
X = nmf.W

kmeans = KMeans(n_clusters=n_clusters)
#cluttering
kmeans.fit(X)
print("-------------------")
#kmeans.score() cost of fitting
print("kmeans: k = {}, cost = {}".format(n_clusters, int(kmeans.score(X))))

ax = plt.figure().add_subplot(111, projection = '3d')
labels = kmeans.labels_
count = 0
#center point of clusters
centers = kmeans.cluster_centers_
#sample data
for c in range(n_clusters):
    cluster = X[labels == c]
    count += cluster.shape[0]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2])
#plot
print(count)
ax.scatter(centers[:,  0], centers[:,  1], centers[:,  2], c = 'k', marker = '+')
plt.show()