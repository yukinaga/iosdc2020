import numpy as np


class TorusNetwork():
    def __init__(self, n_h, n_w, n_connect):
        n_neuron = n_h * n_w  # Number of neurons in a network
        self.params = (n_h, n_w, n_neuron, n_connect)

        self.connect_ids = None  # Indices of presynaptic neurons
        self.w = None  # Weight
        self.b = None  # Bias
        self.y = None  # Output of neurons
        self.proj = None  # Is projection neurons
        self.proj_ids = None  # Indices of projection neurons
        self.proj_to_ids = None  # Indices of destination of projection
        self.inhib = None  # Is inhibitory neurons
        self.desire_ids = None  # Indices of desire neurons
        self.sn_sy = None  # Synapse sensitivity
        self.emotion = 0

    def connect(self, proj_ratio, sigma_inter):
        n_h, n_w, n_neuron, n_connect = self.params

        # Random choise of projection neurons
        n_proj = int(proj_ratio * n_neuron)
        rand_ids = np.random.permutation(np.arange(n_neuron))
        self.proj_ids = rand_ids[:n_proj]
        self.proj_to_ids = np.random.permutation(self.proj_ids)
        self.proj = np.zeros(n_neuron, dtype=np.bool)
        self.proj[self.proj_ids] = True

        # X-coordinate of interneurons
        inter_dist_x = np.random.randn(n_neuron, n_connect) * sigma_inter
        inter_dist_x = np.where(
            inter_dist_x < 0, inter_dist_x-0.5, inter_dist_x+0.5).astype(np.int32)
        x_connect = np.zeros((n_neuron, n_connect), dtype=np.int32)
        x_connect += np.arange(n_neuron).reshape(-1, 1)
        x_connect %= n_w
        x_connect += inter_dist_x
        x_connect = np.where(x_connect < 0, x_connect+n_w, x_connect)
        x_connect = np.where(x_connect >= n_w, x_connect-n_w, x_connect)

        # Y-coordinate of interneurons
        inter_dist_y = np.random.randn(n_neuron, n_connect) * sigma_inter
        inter_dist_y = np.where(
            inter_dist_y < 0, inter_dist_y-0.5, inter_dist_y+0.5).astype(np.int32)
        y_connect = np.zeros((n_neuron, n_connect), dtype=np.int32)
        y_connect += np.arange(n_neuron).reshape(-1, 1)
        y_connect //= n_w
        y_connect += inter_dist_y
        y_connect = np.where(y_connect < 0, y_connect+n_h, y_connect)
        y_connect = np.where(y_connect >= n_h, y_connect-n_h, y_connect)

        # Indices of connection
        self.connect_ids = x_connect + n_w * y_connect

    def initialize_network(self, inhib_ratio, w_mu, w_sigma):
        n_h, n_w, n_neuron, n_connect = self.params

        # Random choise of inhibitory neurons
        n_inhib = int(inhib_ratio * n_neuron)
        rand_ids = np.random.permutation(np.arange(n_neuron))
        inhib_ids = rand_ids[:n_inhib]
        self.inhib = np.zeros(n_neuron, dtype=np.bool)
        self.inhib[inhib_ids] = True

        # Initialize weight and bias
        self.w = np.random.randn(n_neuron, n_connect) * w_sigma + w_mu
        # self.w = np.where(np.isin(self.connect_ids, inhib_ids), -self.w, self.w)
        self.w = np.where(self.inhib[self.connect_ids], -self.w, self.w)
        self.w /= np.sum(self.w, axis=1, keepdims=True)
        self.b = np.zeros(n_neuron)

        # Initialize output
        self.y = np.random.randint(0, 2, n_neuron)

    def initialize_desire(self, desire_ratio):
        n_h, n_w, n_neuron, n_connect = self.params

        self.sn_sy = np.zeros((n_neuron, n_connect))

        # Random choise of desire neurons
        n_desire = int(desire_ratio * n_neuron)
        rand_ids = np.random.permutation(np.arange(n_neuron))
        self.desire_ids = rand_ids[:n_desire]
        self.emotion = 0

    def forward(self, delta_b, mu_u, excite_ratio, ramda_w, decay_ratio, desire_excite_ratio):
        n_h, n_w, n_neuron, n_connect = self.params
        n_excite = int(excite_ratio * n_neuron)

        # Forward calculation of neurons
        self.y[self.proj_to_ids] = self.y[self.proj_ids]  # Projection
        x = self.y[self.connect_ids]
        u = np.sum(self.w*x, axis=1) - self.b
        larger_ids = np.argpartition(-u, n_excite)[:n_excite]
        self.y[:] = 0
        self.y[larger_ids] = 1

        # Homeostasis
        self.b = np.where(self.y, self.b+delta_b, self.b-delta_b)
        self.b -= np.mean(self.b)

        # Desire
        self.sn_sy *= decay_ratio
        self.sn_sy = np.where(self.y.reshape(-1, 1)*x, 1, self.sn_sy)
        n_desire_excite = int(desire_excite_ratio * len(self.desire_ids))
        # self.emotion = 0
        # if self.y[self.desire_ids].sum() >= n_desire_excite:
        #     self.emotion = 1

        # Hebbian learning
        self.w += ramda_w*self.emotion*self.sn_sy * \
            self.y.reshape(-1, 1)*x*np.where(self.w < 0, -1, 1)
        self.w /= np.sum(self.w, axis=1, keepdims=True)  # Synaptic scaling
