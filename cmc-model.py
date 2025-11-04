import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class ConnectedMixtureComponents:
    """
    Connected Mixture Components (CMC)
    Probabilistic feature extractor for time-series data.
    """

    def __init__(self, n_components=4, window_size=168, step_size=24, epsilon=0.5, random_state=42):
        self.n_components = n_components
        self.window_size = window_size
        self.step_size = step_size
        self.epsilon = epsilon
        self.random_state = random_state
        self.connected_components = []
        self.features_ = None

    def _fit_gmm(self, X):
        gmm = GaussianMixture(n_components=self.n_components,
                              covariance_type='diag',
                              random_state=self.random_state)
        gmm.fit(X)
        return gmm.means_, np.sqrt(gmm.covariances_), gmm.weights_

    def fit_transform(self, X):
        n = X.shape[0]
        windows = [
            X[i:i + self.window_size]
            for i in range(0, n - self.window_size, self.step_size)
        ]
        gmms = [self._fit_gmm(w) for w in windows]
        self.connected_components = self._connect_components(gmms)
        self.features_ = self._build_features(self.connected_components)
        return self.features_

    def _connect_components(self, gmms):
        connections = []
        prev_means, prev_stds, prev_weights = gmms[0]
        for t in range(1, len(gmms)):
            curr_means, curr_stds, curr_weights = gmms[t]
            dist = cdist(prev_means, curr_means)
            links = dist < self.epsilon
            connections.append((t, links))
            prev_means, prev_stds, prev_weights = curr_means, curr_stds, curr_weights
        return connections

    def _build_features(self, connections):
        # Example placeholder: number of active connections per window
        feats = np.array([np.sum(c[1]) for c in connections])
        return feats.reshape(-1, 1)

    def get_component_stats(self):
        stats = {
            'num_connections': [np.sum(c[1]) for c in self.connected_components],
            'time_index': [c[0] for c in self.connected_components],
        }
        return stats

    def plot_component_stats(self):
        stats = self.get_component_stats()
        plt.figure(figsize=(10, 4))
        plt.plot(stats['time_index'], stats['num_connections'], label='Active Connections')
        plt.xlabel("Window Index")
        plt.ylabel("Connected Components")
        plt.legend()
        plt.tight_layout()
        plt.show()
