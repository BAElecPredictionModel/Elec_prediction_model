import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans

from umap import UMAP

class ClusterPattern:
    def __init__(self, data: pd.DataFrame):
        # Save input as class variables
        self.data = data
        self.timeIndex = data.index

        # Pre-declare the variables for future functions
        self.labels = None
        self.num_of_labels = None

    # Dimension reduction for clustering
    def dim_reduction(self, clf_method=None) -> np.array:
        # Convert DataFrame to Numpy array
        arr_raw = self.data.values

        # Dimension reduction
        self.arr_tsne = TSNE().fit_transform(arr_raw)
        self.arr_umap = UMAP().fit_transform(arr_raw)

        if not clf_method:
            return arr_raw
        elif clf_method == 'tsne':
            return self.arr_tsne
        elif clf_method == 'umap':
            return self.arr_umap


    # Clustering: Generate pattern labels
    def clustering(self, data_arr: np.array, eps=10) -> pd.DataFrame:
        labels = DBSCAN(eps=eps).fit_predict(data_arr)
        # labels = KMeans(n_clusters=2).fit_predict(data_arr)
        labels = pd.DataFrame(labels, index=self.timeIndex, columns=['label'])
        
        self.labels = labels
        self.num_of_labels = labels.max() + 1

        return labels
