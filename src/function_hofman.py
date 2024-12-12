import numpy as np
from scipy.special import comb, digamma, betaln, gammaln
from scipy.sparse import lil_matrix
import numba
from matplotlib import pyplot as plt
from tqdm import trange

class NetworkModuleInference:
    def __init__(self, A, K_values, opts=None, net0=None):
        self.A = lil_matrix(A, dtype='b').tocsr() if not isinstance(A, lil_matrix) else A
        self.K_values = K_values if isinstance(K_values, (list, np.ndarray)) else [K_values]
        self.opts = opts or {}
        self.net0 = net0 or {}

    @staticmethod
    def init(N, K):
        Q = np.random.random([N, K])
        return Q / Q.sum(axis=1, keepdims=True)

    @staticmethod
    @numba.njit
    def estep_numba(rows, cols, Q, JL, JG, lnpi, n):
        for i, j in zip(rows, cols):
            Q[i] = np.exp(JL * Q[j] + JG + lnpi)
            sum_q = Q[i].sum()
            Q[i] /= sum_q + 1e-5

    def safe_digamma(x):
        """Calculate digamma, ensuring the input is strictly positive."""
        return digamma(np.clip(x, 1e-4, np.inf))

    def safe_betaln(a, b):
        """Calculate betaln, ensuring that a and b are strictly positive."""
        a_safe = np.clip(a, 1e-4, np.inf)
        b_safe = np.clip(b, 1e-4, np.inf)
        return betaln(a_safe, b_safe)

    def safe_gammaln(a):
        """Calculate gammaln, ensuring that a is strictly positive."""
        a_safe = np.clip(a, 1e-4, np.inf)
        return gammaln(a_safe)
    
    def learn(self, K, max_iter=20, verbose=False):
        N = self.A.shape[0]
        M = 0.5 * self.A.sum()
        C = comb(N, 2)

        ap0, bp0, am0, bm0 = self.net0.get('ap0', N * 2), self.net0.get('bp0', 1), self.net0.get('am0', 1), self.net0.get('bm0', 2)
        a0 = np.ones([1, K])

        Q = self.net0.get('Q0', self.init(N, K))
        ap, bp, am, bm, a = ap0, bp0, am0, bm0, a0
        n = Q.sum(axis=0)
        rows, cols = self.A.nonzero()

        F = []
        TOL_DF = self.opts.get('TOL_DF', 1e-2)
        VERBOSE = self.opts.get('VERBOSE', 0)

        for i in trange(max_iter) if verbose else range(max_iter):
            psiap, psibp = safe_digamma(ap), safe_digamma(bp)
            psiam, psibm = safe_digamma(am), safe_digamma(bm)
            psip, psim = safe_digamma(ap + bp), safe_digamma(am + bm)

            JL, JG = psiap - psibp - psiam + psibm, psibm - psim - psibp + psip
            lnpi = safe_digamma(a) - safe_digamma(a.sum())

            self.estep_numba(rows, cols, Q, JL, JG, lnpi, n)

            n = Q.sum(axis=0)
            npp = 0.5 * (Q.T @ self.A @ Q).diagonal().sum()
            npm = 0.5 * np.trace(Q.T @ (N * n - Q)) - npp
            nmp, nmm = M - npp, C - M - npm

            ap, bp = npp + ap0, npm + bp0
            am, bm = nmp + am0, nmm + bm0
            a = n + a0

            Q[Q < 1e-323] = 1e-323
            F_value = safe_betaln(ap, bp) - safe_betaln(ap0, bp0) + safe_betaln(am, bm) - safe_betaln(am0, bm0)
            F_value += np.sum(safe_gammaln(a)) - safe_gammaln(a.sum()) - np.sum(np.multiply(Q, np.log(Q)))
            F.append(-F_value)


            if i > 1 and abs(F[-1] - F[-2]) < TOL_DF:
                break

        return {
            'F': F[-1],
            'F_iter': F,
            'Q': Q,
            'K': K,
            'ap': ap,
            'bp': bp,
            'am': am,
            'bm': bm,
            'a': a
        }

    def learn_restart(self):
        NUM_RESTARTS = self.opts.get('NUM_RESTARTS', 20)
        results = []
        F_K = []

        for K in self.K_values:
            best_net = None
            best_F = -np.inf

            for _ in range(NUM_RESTARTS):
                net = self.learn(K, verbose=self.opts.get('VERBOSE', False))
                if net['F'] > best_F:
                    best_net = net
                    best_F = net['F']

            results.append(best_net)
            F_K.append(best_F)

        best_idx = np.argmax(F_K)
        return results[best_idx], results

    def plot_results(self, net, nets_by_K):
        K_values = [net['K'] for net in nets_by_K]
        F_values = [net['F'] for net in nets_by_K]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(K_values, F_values, 'b^-')
        plt.plot(net['K'], net['F'], 'ro', label='Optimal K')
        plt.legend()
        plt.title('Complexity Control')
        plt.xlabel('K')
        plt.ylabel('F')
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.imshow(net['Q'], interpolation='nearest', aspect='auto')
        plt.title('Q Matrix')
        plt.xlabel('K')
        plt.ylabel('N')

        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(net['F_iter']) + 1), net['F_iter'], 'bo-')
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('F')
        plt.grid()

        plt.show()

N, K = 238, 12
tp, tm = 0.05, 0.01
graph = nx.read_gml("../data/sp_school_day_2.gml")
A = nx.to_numpy_array(graph)

opts = {'NUM_RESTARTS': 5, 'VERBOSE': 1}
inference = NetworkModuleInference(A, 12, opts=opts)
best_net, nets_by_K = inference.learn_restart()
inference.plot_results(best_net, nets_by_K)

