import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X,_=make_blobs(n_samples=1500,centers=4,n_features=3,cluster_std=1.4,random_state=42)
X=StandardScaler().fit_transform(X); m=KMeans(n_clusters=4,n_init=20,random_state=42); labels=m.fit_predict(X)
print({'silhouette':silhouette_score(X,labels),'clusters':len(set(labels))})
