from directed_tree import Node
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings

'''Newman Ziff algorithm for site percolation. O(n) 

Supports calculation of spanning cluster statistics: 
   fraction of occupied sites in spanning cluster, and number of spanning clusters.'''


class SitePercolate():
    def __init__(self, shape, stats={}, show=False):
        '''shape is the shape of the lattice. The lattice can be square (2D), or cubic (3D).
        If 'spanning cluster' in stats, efficiently updates spanning cluster 
        statistics each time a site is occupied.
        '''
        self.lattice_shape = shape
        self.num_sites = np.prod(shape)
        self.show = show
        if isinstance(stats, str):
            self.stats = {stats}
        else:
            self.stats = set(stats)
        # Initializes the lattice and various data maintained on the percolating system
        self.reset() 
        
    def reset(self):
        '''Clears all occupied sites and resets various data maintained on the percolating system.'''
        shape = self.lattice_shape
        self.lattice = np.full(shape, None, dtype=Node) # None means a site is empty.
        self.num_occupied = 0 # Number of sites occupied
        if self.stats: # Define quantities useful for calculating stats:
            # If a merge occurs when the new site is added, this will contain
            # the cluster grown by the merge, and the clusters it consumed.
            self.merge_info = {"grown":None,
                               "consumed":[]}
        if "spanning cluster" in self.stats:
            self.spanning_clusters = set()

            self.boundary_touching = np.empty(len(shape), dtype=dict)
            for axis in range(len(shape)):
                # Can't use np.full, because the entries must be distinct objects
                self.boundary_touching[axis] = {'left': set(),
                                                'right': set()}

    def run(self, show=False):
        '''One run of the Newman Ziff Algorithm.
        Sites are occupied in random order.
        
        If show is True, displays a visualization (only implemented for 2D lattice)'''
        self.reset()
        # Random permutation of indicies:
        self.rng = rng = np.random.default_rng()
        permutation = list(np.ndindex(self.lattice.shape))
        rng.shuffle(permutation)

        stats = []
        self.show = show
        if show:
            self.display_handles = self.setup_show2D()

        for index in permutation:
            self.update(index)
            # Must be called after new site is occupied with self.update():
            if self.stats: stats.append(self.update_stats(index))
            if show and not self.lattice[index].is_root(): self.update_show2D()
        
        return stats


    def update(self, index):
        '''Occupies the site in the lattice specifed by index, growing or merging clusters as needed.

        Occupy a site by filling it with a Node. 
        If it connects to a cluster, it will be child of the cluster's root node.
        If it connects multiple clusters, they will be merged under the largest cluster.
        If there are no occupied neighbors, it forms a new cluster.'''
        self.num_occupied += 1
        neighbor_clusters = set() # set of neighboring clusters' root nodes
        for i in self.neighbor_ind(index): 
            neighbor = self.lattice[i]
            if neighbor is not None: # neighbor site is occupied.
                cluster_root = self.find_root(neighbor)
                neighbor_clusters.add(cluster_root)
        if len(neighbor_clusters) == 0:
                # This site is a new cluster, of size 1
                self.lattice[index] = Node(data=1)
        else:
            largest_cluster = np.max(list(neighbor_clusters))
            self.lattice[index] = Node(parent=largest_cluster)
            largest_cluster.data +=1
            # Merge clusters if needed
            self.merge(neighbor_clusters, largest_cluster)
        
        self.stats_updated = False # Stats no longer reflect current configuration


    def update_stats(self, new_site_index):
        '''Update stats given location of latest occupied site'''
        if self.stats_updated:
            raise warnings.warn('Not ready to update stats, need to occupy new site first. No action taken.', category=RuntimeWarning)
        results = {}
        if "spanning cluster" in self.stats:
            self.__update_spanning(new_site_index)
            results['spanning'] = self.spanning_stats()

        self.stats_updated = True
        return results


    def spanning_stats(self):
        '''Returns a dict containing stats on the spanning cluster(s): 
        {'num': number of spanning clusters,
         'P_infinity': fraction of the occupied sites that are part of a spanning cluster'}

        If "spanning cluster" was given in self.stats,
        this is effiently updated at each time a site is added. 
        If not, spanning cluster(s) are computed from scratch.

        A spanning cluster is defined as a cluster that extends from one boundary of the
        system to the opposite boundary. '''
        if "spanning cluster" not in self.stats: 
            # Computes spanning clusters from scratch, iterating over the boundaries of the lattice
            self.spanning_clusters = self.__find_spanning()

        P_infty = 0
        for span_clust in self.spanning_clusters: P_infty += span_clust.data
        P_infty /= self.num_occupied

        stats = {'num': len(self.spanning_clusters), # = deepcopy(self.spanning_clusters)
                 'P_infinity': P_infty}
        return stats

    def __update_spanning(self, new_site_index):
        '''Efficiently updates SitePercolate.spanning_clusters given location of next occupied site.
         Returns SitePercolate.spanning_clusters: a set of spanning clusters' root nodes. 

        A spanning cluster is defined as a cluster that extends from one boundary of the
        system to the opposite boundary. '''
        if self.stats_updated:
            raise warnings.warn('Not ready to update spanning cluster stats, need to occupy new site first. No action taken.', category=RuntimeWarning)

        # First update self.boundary_touching.
        self.__update_bound_touching(new_site_index)

        # A new site will only change the spanning clusters if merges with at least one
        # cluster. (Makes the reasonable assumption that the lattice has more than one site.)
        if self.lattice[new_site_index].is_root():
            return self.spanning_clusters
        # If the cluster that grew in the merge is spanning, add it to self.spanning_clusters.
        # self.spanning_clusters is a set, which automatically prevents duplicates.
        if self.spans(self.merge_info['grown']):
            self.spanning_clusters.add(self.merge_info['grown'])
            # Make spanning clusters white:
            if self.show: self.cluster_colors[self.merge_info['grown']] = 1.0
        # If any spanning clusters were consumed in a merge, remove them from self.spanning_clusters
        self.spanning_clusters -= self.merge_info['consumed']
        

        return self.spanning_clusters

    def __update_bound_touching(self, new_site_index):
        '''Updates self.boundary_touching given the new occupied site index.'''
        # new site might form a new cluster on the boundary, 
        #                connect an existing cluster to the boundary,
        #                cause a merge that consumes sites touching the boundary
        if self.merge_occurred(new_site_index):
            # If merge consumed boundary-touching clusters, the grown cluster now touches those boundaries
            for axis in range(len(self.lattice.shape)):
                for side in ['left', 'right']:
                    if self.merge_info['consumed'].intersection(self.boundary_touching[axis][side]):
                        # The merge consumed clusters touching the this boundary 
                        self.boundary_touching[axis][side].add(self.merge_info['grown'])
                        self.boundary_touching[axis][side] -= self.merge_info['consumed']
                            
        if self.on_boundary(new_site_index):
            cluster = self.find_root(new_site_index)
            for axis, lattice_len in enumerate(self.lattice.shape):
                if new_site_index[axis] == 0:
                    self.boundary_touching[axis]['left'].add(cluster)
                if new_site_index[axis] == lattice_len-1:
                    self.boundary_touching[axis]['right'].add(cluster)

    def spans(self, cluster):
        '''Returns true if the cluster is spanning, false otherwise.
        If  "spanning cluster" in self.stats, makes use of previous iteration's computation.
        Otherwise, ineffienciently iterates over the full boundaries of the lattice.
        '''
        if  "spanning cluster" in self.stats:
            for axis in range(len(self.lattice.shape)):
                if cluster in self.boundary_touching[axis]['left'] and \
                   cluster in self.boundary_touching[axis]['right']:
                      return True
            return False
        else: 
            # This could be done more efficiently, i.e. without finding all of the boundary touching clusters,
            # since we know the one cluster we are interested in.
            return cluster in self.__find_spanning()

    
    
    def merge_occurred(self, new_site_index):
        '''Returns True if a merge occurred when new_site_index
         was occupied, False otherwise. Clusters only grow through merges,
         whether that means a single site was added to it, or multiple. '''
        if self.lattice[new_site_index].is_root():
            return False
        else:
            return True

    def on_boundary(self, index):
        '''Returns True if index is on the boundary of the lattice, False otherwise. '''
        if 0 in index: # left boundary
            return True
        for i, l in zip(index, self.lattice.shape):
            if i == l-1: # right boundary
                return True

    def __find_spanning(self):
        '''Identifies spanning clusters by iterating along the boundaries of the lattice. 
        Returns a set of spanning clusters' root nodes, and overwrites SitePercolate.spanning_clusters
        with the same.
        A spanning cluster is defined as a cluster that extends from one boundary of the
        system to the opposite boundary. '''
        spanning_clusters = set()
        for axis in range(len(self.lattice.shape)):
            # Check for clusters that extend over the whole system in the direction of axis
            left_clusters = set() # Clusters that touch the left boundary
            left_indicies = self.boundary_ind(axis, left_side=True)
            for index in left_indicies:
                if self.occupied(index):
                    cluster = self.find_root(self.lattice[index])
                    left_clusters.add(cluster)

            right_clusters = set() # Clusters the touch the right boundary
            right_indicies = self.boundary_ind(axis, left_side=False)
            for index in right_indicies:
                if self.occupied(index):
                    cluster = self.find_root(self.lattice[index])
                    right_clusters.add(cluster)
            spanning_clusters |= left_clusters.intersection(right_clusters)
        return spanning_clusters

    def boundary_ind(self, axis, left_side=True):
        '''Yields each index along the left or right boundary of the given axis.
        For instance, if the lattice is 5 by 6 by 7, and axis is 1, then the left boundary 
        indices all have 0 in axis 1: 
        (0, 0, 0), (0, 0, 1), ... (0, 0, 6), (1, 0, 0), ... (4, 0, 6).
        The right boundary indicies all have 6-1=5 in axis 1.
        By default, yields indicies of left boundary (left_side=True). If left_side is False,
        yields the indicies of right boundary.
        '''
        dims = len(self.lattice.shape)
        boundary_shape = list(self.lattice.shape)
        del boundary_shape[axis]

        if left_side:
            boundary_index = np.zeros(dims)
        else: # right side
             boundary_index = np.full(dims, self.lattice.shape[axis]-1)

        for index in np.ndindex(boundary_shape):
            for i in range(dims-1): # dims-1 = dimensionality of a boundary
                if i < axis:
                    boundary_index[i] = index[i]
                else:
                    boundary_index[i +1] = index[i]
            yield tuple(boundary_index)

            
        

    def neighbor_ind(self, index): 
        '''Yields the index of each neighbor.'''
        if isinstance(index, int): # 1D lattice
             if index > 0:
                  yield index - 1
             if index+1 < len(self.lattice):
                  yield index + 1
             return
        for dim in range(len(index)): # ND lattice 
                if index[dim] > 0:
                    neighbor_index = tuple(
                         [index[d]-1 if d == dim else index[d] for d in range(len(index))])
                    yield neighbor_index
                if index[dim]+1 < self.lattice.shape[dim]:
                    neighbor_index = tuple(
                         [index[d]+1 if d == dim else index[d] for d in range(len(index))])
                    yield neighbor_index

    def find_root(self, site):
        '''Given a site (Node or index), trace the cluster tree up to the root node,
        which identifes the cluster. Return this root node and make all nodes along
        the path we followed become children of the root node, reducing future iterations.'''
        try:
            if isinstance(site, tuple):
                site = self.lattice[site]
            assert isinstance(site, Node)
        except:
            print(site)
            print(type(site))
            raise
        path = [] # nodes encountered on the way to finding root node
        current_node = site
        # Find root node
        while not current_node.is_root():
            path.append(current_node)
            current_node = current_node.parent
        root = current_node
        # Make all nodes encountered on the way to finding the root be
        # children of the root.
        for node in path:
             node.parent = root
        return root

    def merge(self, clusters, largest_cluster):
        '''Merge smaller clusters into largest cluster.
        clusters is a collection of cluster root nodes
        largest_cluster is the root of the cluster they will be merged under.
        '''
        if not isinstance(clusters, set):
            # Can't have duplicates, or they will be double counted
            clusters = set(clusters) 
        clusters.remove(largest_cluster)

        for cluster in clusters:
            # Two clusters are merged by making the root of the smaller cluster 
            # a child of the larger cluster's root.
            cluster.parent = largest_cluster
            # Size of smaller cluster is now added to size of large cluster.
            largest_cluster.data += cluster.data
            # print('Newly Merged Cluster Size: ', largest_cluster.data)
        
        if self.stats:
            self.merge_info['grown'] = largest_cluster
            self.merge_info['consumed'] = clusters

        
    
    def occupied(self, index):
        '''Returns True if the site at index is occupied, False if not.'''
        return self.lattice[index] is not None
    
    def cluster_size(self, node):
        '''Return size of cluster containing the given node.'''
        return self.find_root(node).data
    # def show(self):
    #     ''''''
    #     if len(self.lattice.shape) == 1: # 1D
    #          self.show1D()
    #     elif len(self.lattice.shape) == 2: # 2D
    #          self.show2D()
    #     else:
    #          return NotImplemented
    
    # def show2D(self):

    def setup_show2D(self):
        assert len(self.lattice.shape) == 2
        # Make colormap:
        gist_ncar = mpl.colormaps['gist_ncar']
        black = np.array([0.0, 0.0, 0.0, 1.0]) 
        white = np.array([1.0, 1.0, 1.0, 1.0]) 
        newcolors = np.full((256, 4), black)
        newcolors[1:-1] = gist_ncar(np.linspace(0.09, 0.9, 254))
        newcolors[-1] = white
        # starts with black at newcolors[0], 
        # then jumps up to brighter colors for newcolors[1:]
        newcmp = ListedColormap(newcolors)
        fig, ax = plt.subplots() 
        im = ax.imshow(np.zeros(self.lattice.shape), cmap = newcmp, vmin=0, vmax=1)
        # plt.colorbar(im, ax=ax)
        fig.draw_artist(im)
        plt.show(block = False)
        # To be filled with mapping from clusters to colors
        # as the run progresses
        self.cluster_colors = {} 
        return fig, im

    # def show_run2D(self, dt=0.01, save=False): # NOT DONE
    #     '''Visualize the process of one run, for a 2D lattice. 
    #     Each frame stays up for duration dt. 
    #     If save is True, save as a gif (default is False), '''
    def update_show2D(self):
        fig, im = self.display_handles
        im_data = np.zeros(self.lattice.shape)
        # Iterates through the entire lattice every frame 
        for index in np.ndindex(self.lattice.shape):
            if self.occupied(index):
                cluster = self.find_root(self.lattice[index])
                if cluster not in self.cluster_colors:
                    self.cluster_colors[cluster] = self.rng.random()
                im_data[index] = self.cluster_colors[cluster]
        im.set_data(im_data)  # Update data
        fig.draw_artist(im)
        # fig.canvas.draw_idle()
        plt.pause(0.001)  # Pause to allow the user to see it
        
        
if __name__ == "__main__":
    lattice_shape = (50, 50)
    sp = SitePercolate(lattice_shape , stats = "spanning cluster")
    stats = sp.run(show=True)
    # Number of sites
    N = sp.num_sites
    # Number of spanning clusters
    num_spanning = np.empty(len(stats), dtype=int)
    # Total number of sites in spanning cluster(s), divided by number of occupied sites
    P_infty = np.empty(len(stats))

    for i, res in enumerate(stats):
        spanning_stats = res['spanning']
        num_spanning[i] = spanning_stats['num']
        P_infty[i] = spanning_stats['P_infinity']
    p = np.arange(N) / N # Occupation Ratio
    fig, axes = plt.subplots(2, 1)
    plt.suptitle(f"Lattice Shape: {lattice_shape}")
    plt.tight_layout()
    axes[0].set_title(r'$P_{\infty}$')
    axes[0].plot(p, P_infty)
    axes[1].set_title('# spanning')
    axes[1].plot(p, num_spanning)
    plt.show()