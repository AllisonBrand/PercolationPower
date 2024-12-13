import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.colors import ListedColormap

viridis = mpl.colormaps['viridis'].resampled(8)

class ClusterFind:

    def __init__(self, occupied=np.array([])):
        # When a cluster is merged, the label specified by key
        # falls under the label specied by value. 
        self.occupied = occupied
        self.label_relations = {}
        self.label = np.array([])
        self.cluster_count = 0

    def find_clusters(self, occupied=None):
        '''
        occupied is an ndarray of booleans.
        True means a site is occupied. False means a site is empty.
        Cubic Lattice
        
        Implements Hoshen-Kopelman Algorithm'''
        def visited_neighbors(index):
            '''For a square or cubic lattice, 
            yields indicies of visited neighbors, 
            assuming a raster scan.'''
            for dim in range(len(index)):
                if index[dim] > 0:
                    neighbor = np.array(index)
                    neighbor[dim] -= 1
                    yield tuple(neighbor)
        # occupied = np.array(occupied, dtype=bool)
        if occupied is None:
            occupied = self.occupied
        else:
            self.occupied = occupied
        # Initialize label_relations to empty, so reusing an instance of ClusterFind for a 
        # different arrangement of occupied and empty sites doesn't fail. 
        self.label_relations = {} 
        shape = np.shape(occupied)
        # Will contain numeric labels, such that all sites in a cluster share the same
        # label, and each label is unique to a cluster.
        # Cluster labels count up from 1
        self.label = np.full(shape, 0)
        label = self.label

        # visited = np.full(shape, False, dtype=bool)
        count = 0 # Count of cluster labels used
        for index in np.ndindex(shape):
            # visited[index] = True
            if occupied[index]:
                # Check the visited neighbors to see if they are 
                # occupied and already assigned a cluster
                for neighbor in visited_neighbors(index):
                    if occupied[neighbor]:
                        if not label[index]:
                            label[index] = label[neighbor]
                        elif label[index] != label[neighbor]:
                            # Already took on the label of one occupied neighbor, 
                            # and this occupied neighbor has a different label.
                            # a.k.a. the current site is merging two clusters
                            label[index] = self.unify(
                                 label[index], label[neighbor])
                # If none of the visited neighbors were occupied, then
                # we never gave this site a label and
                # we need a new cluster label for it. 
                if not label[index]:
                    self.cluster_count +=1
                    count += 1
                    label[index] = count

        self.clean()
        return label

        # stack = []
        # # Breadth First Search


    def unify(self, label_1, label_2):
        '''Merge the clusters represented by label_1 and label_2'''
        # Go with the smaller label, 
        # checking if the labels are already superseded:
        label = min(self.true_label(label_1), 
                    self.true_label(label_2))
        if label != label_1:
            self.relabel(label_1, label)
        if label != label_2:
            self.relabel(label_2, label)
        return label

    def true_label(self, label):
        '''If the label has been superseded, return the 
        label it is superseded by.'''
        # There may be a chain of succession in the labels
        # Find last label in the chain.
        while label in self.label_relations:
            label = self.label_relations[label] 
        return label
    
    def relabel(self, old_label, new_label):
        ''' Make new_label superseed old_label, and if old_label was already superseded,
        go down the chain of label succession pointing each one to the new_label.
        Update cluster_count if needed.'''
        while old_label in self.label_relations:
            # Label that old_label used to point to:
            next_label = self.label_relations[old_label]
            # Point old_label to new_label
            self.label_relations[old_label] = new_label
            # next_label, and any labels it points too, all need to be pointed to new_label
            old_label = next_label

        # If a label is being superseded for the first time,
        # the number of unique clusters falls by 1.
        if old_label != new_label:
            # At this point, we know that (old_label not in self.label_relations), 
            # a.k.a. it hasn't already been superseded.
            # old_label is being superseded for the first time.
            self.cluster_count -=1
            self.label_relations[old_label] = new_label



    def clean(self, fix_labels=True):
        '''Raster scan over the lattice, overwritting labels that have been 
        superseded. If fix_labels is True (the default), additionlly ensures that the labeling doesn't skip over numbers.'''
        # There are self.cluster_count unique clusters. 
        # The initial labeling can skip over numbers because some clusters are merged.
        # These initial labels are the return values from self.true_label(self.label[index]).
        count = 1 # Provides the new, consecutive labels
        mapping = {}
        if fix_labels:
            for index in np.ndindex(*np.shape(self.occupied)):
                if self.label[index] == 0:
                    continue
                initial = self.true_label(self.label[index])
                if initial not in mapping:
                    mapping[initial] = count
                    count += 1
                self.label[index] = mapping[initial]
        else: 
            for index in np.ndindex(*np.shape(self.occupied)):
                self.label[index] = self.true_label(self.label[index])

    def show2D(self):
        '''Visualize the clusters identified for a 2D matrix'''
        gist_ncar = mpl.colormaps['gist_ncar']
        newcolors = gist_ncar(np.linspace(0, 1, 256))
        white = np.array([1, 1, 1, 1])
        newcolors[0] = white
        newcmp = ListedColormap(newcolors)
        # Code below taken from https://stackoverflow.com/questions/76710406/hoshen-kopelman-algorithm-for-cluster-detection
        plt.matshow(self.label, cmap = newcmp)
        rows, cols = np.shape(self.label)
        for i in range(rows):
            for j in range(cols):
                c = self.label[i, j]
                plt.text(j, i, str(c), va='center', ha='center', color='white')
        plt.show()
         

if __name__ == "__main__":
    rng = np.random.default_rng()
    occupied = rng.integers(2, size=(50, 50)) # ones and zeros
    cf = ClusterFind(occupied)
    cf.find_clusters()
    cf.show2D()

