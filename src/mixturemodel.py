import numpy as np
import time
from .viz import *
import torch.multiprocessing as mp
from itertools import permutations
from .utils import convert_to_ranks
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import random
import torch
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans


def worker(seed, model, results, init, max_it, tolerance):
    torch.manual_seed(seed)

    if init == "Kmeans" or init == "K-means":
        tau_init = model._init_K_means()
    elif init == "Spectral":
        tau_init = model._init_spectral()
    elif init == "Sparse" or init == "RandomSparse":
        tau_init = model._init_tau_sparse()
    else:
        tau_init = model._init_tau()

    alpha, pi, tau, logs_like = model.em(tau=tau_init, max_it=max_it, tolerance=tolerance, verbose=False, log_path=True)

    alpha_cpu = torch.tensor(alpha).clone()
    pi_cpu = torch.tensor(pi).clone()
    tau_cpu = tau.clone().cpu()

    likelihood = model._likelihood(
        alpha_cpu.to(model.device),
        pi_cpu.to(model.device),
        tau_cpu.to(model.device)
    )

    results.append({
        'alpha': alpha_cpu,
        'pi': pi_cpu,
        'tau': tau_cpu,
        'likelihood': likelihood.item(),
        'logs_like': logs_like
    })


class MixtureModel():
    def __init__(self, X, n, k, device='cpu'):
        self.device = torch.device(device)
        self.N = n
        self.K = k
        self.logs_like = []
        self.X = torch.tensor(X, dtype=torch.float, device=self.device)
        self.tau = self._init_tau()
    
    def _init_tau(self):
        tau = torch.rand(self.N, self.K, device=self.device)
        tau /= tau.sum(dim=1, keepdim=True)
        return tau

    def _init_tau_sparse(self, sparse=1):

        if sparse > self.K:
            raise ValueError("Pas de valeurs non nulles supérieurs au nombre de classe (baisser sparse).")

        indices = torch.stack([torch.randperm(self.K, device=self.device)[:sparse] for _ in range(self.N)])

        values = torch.rand((self.N, sparse), device=self.device)
        values /= values.sum(dim=1, keepdim=True)

        tau = torch.zeros((self.N, self.K), device=self.device)

        row_indices = torch.arange(self.N, device=self.device).repeat_interleave(sparse)
        col_indices = indices.flatten()
        tau[row_indices, col_indices] = values.flatten()
        
        return tau

    def laplacian_matrix(self):
        adj_matrix = self.X.cpu().numpy().copy()
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        return degree_matrix - adj_matrix

    def spectral_clustering(self):
        L = self.laplacian_matrix()
        k = self.K
        eigvals, eigvecs = np.linalg.eigh(L)

        eigvecs_k = eigvecs[:, :k]

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(eigvecs_k)

        return kmeans.labels_

    def _init_spectral(self):
        tau = torch.zeros(self.N, self.K, device=self.device)
        labels = self.spectral_clustering()
        for i in range(self.N):
            tau[i,labels[i]] = 1
        return tau

    def _init_K_means(self, max_iter=5):
        adj_matrix = self.X.cpu().numpy().copy()
        centroids = []
        first_centroid = random.randint(0, self.N - 1)
        centroids.append(first_centroid)
        k = self.K

        #We choose one after another the other centroids by choosing the node thax maximize the minimum distance between the already chosen centroids
        for _ in range(1, k):
            distances_to_centroids = np.min(euclidean_distances(adj_matrix)[:, centroids], axis=1)
            farthest_node = np.argmax(distances_to_centroids)
            centroids.append(farthest_node)
            
        labels = np.zeros(self.N)
        
        for _ in range(max_iter):

            distances = euclidean_distances(adj_matrix)
            for i in range(self.N):
                min_dist = float('inf')
                closest_centroid = -1
                for c in centroids:
                    dist = distances[i, c]
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid = c
                labels[i] = closest_centroid

            new_centroids = []
            for c in range(k):
                cluster_nodes = [i for i in range(self.N) if labels[i] == c]
                if cluster_nodes:
                    new_centroid = np.mean(adj_matrix[cluster_nodes], axis=0)
                    closest_node = np.argmin(np.sum((adj_matrix - new_centroid) ** 2, axis=1))
                    new_centroids.append(closest_node)
                else:

                    new_centroids.append(random.randint(0, self.N - 1))

            if np.array_equal(centroids, new_centroids):
                break
            centroids = new_centroids
        tau = torch.zeros(self.N, k)

        labels = convert_to_ranks(labels)
        for i in range(self.N):
            tau[i] = torch.zeros(k, dtype=torch.float)
            tau[i][int(labels[i])] = 1
        return tau.to(self.device)

    def comp_alpha_pi(self, tau):

        mask = 1 - torch.eye(self.N, device=tau.device)  # Diagonal mask to exclude self-loops

        numerator = torch.einsum('iq,jl,ij->lq', tau, tau, self.X.to(tau.device) * mask)
        denominator = torch.einsum('iq,jl,ij->lq', tau, tau, mask.float())

        pi = numerator / denominator
        alpha = tau.mean(dim=0)

        return alpha, pi  # (K), (KxK)

    def compute_tau(self, alpha, pi, tau, epsilon=1e-5):
        N, K = tau.size()

        # clip sur pi pour éviter les valeurs proches de 0 ou 1, qui peuvent poser problème lors du log.
        pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)

        X_expanded = self.X.unsqueeze(-1).unsqueeze(-1)  # (N, N, 1, 1)
        pi_expanded = pi.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        log_b = X_expanded * torch.log(pi_expanded) + (1 - X_expanded) * torch.log(1 - pi_expanded)

        tau_expanded = tau.unsqueeze(0).unsqueeze(-1)  # (1, N, K, 1)
        log_b_tau = log_b * tau_expanded  # (N, N, K, K)

        mask = 1 - torch.eye(N, device=self.device).unsqueeze(-1).unsqueeze(-1)  # (N, N, 1, 1)
        masked_log_b_tau = log_b_tau * mask

        sum_jl = torch.einsum('ijnq->iq', masked_log_b_tau)  # (N, K)

        log_alpha = torch.log(alpha + epsilon).unsqueeze(0)  # (1, K)

        log_tau = log_alpha + sum_jl
        log_tau = log_tau - (log_tau + epsilon).logsumexp(dim=1, keepdim=True)  # Normalisation en espace logarithmique

        tau_new = torch.exp(log_tau)  # Retour à l'espace des probabilités

        # Normalisation par ligne pour s'assurer que la somme de chaque ligne de tau est égale à 1
        tau_new = tau_new / tau_new.sum(dim=1, keepdim=True)

        tau_new = torch.nan_to_num(tau_new, nan=epsilon)

        return tau_new

    def _Q(self, alpha, pi, tau):

        term1 = torch.einsum('iq,q->', tau, alpha)

        theta = torch.einsum('iq,jl->ijql', tau, tau)

        log_b = self.X.unsqueeze(-1).unsqueeze(-1) * torch.log(pi) + (1 - self.X.unsqueeze(-1).unsqueeze(-1)) * torch.log(1 - pi)

        mask = torch.triu(torch.ones(self.N, self.N), diagonal=1).to(self.device)
        masked_theta = theta * mask.unsqueeze(-1).unsqueeze(-1)

        term2 = torch.einsum('ijql,ijql->', masked_theta, log_b)

        return term1 + term2


    def _likelihood(self, alpha, pi, tau, epsilon=1e-5):
        pi = torch.clamp(pi, min=epsilon, max=1 - epsilon)
        term1 = torch.einsum('iq,q->', tau, torch.log(alpha))

        X_unsqueezed = self.X.unsqueeze(-1).unsqueeze(-1).to(tau.device)  # (N, N, 1, 1)
        log_b = (
            X_unsqueezed * torch.log(pi) + (1 - X_unsqueezed) * torch.log(1 - pi)
        )
        term2 = 0.5 * torch.einsum('iq,jl,ijql->', tau, tau, log_b)

        epsilon = 1e-10
        term3 = -torch.einsum('iq,iq->', tau, torch.log(tau + epsilon))

        result = term1 + term2 + term3
        return result#/self.N

    def _fixed_point_algorithm(self, alpha, pi, tau_initial, tol=1e-6, max_iter=100):
        tau = tau_initial.clone()
        for i in range(max_iter):
            tau_new = self.compute_tau(alpha, pi, tau)
            diff = torch.norm(tau_new - tau, p='fro')
            if diff < tol:
                break
            tau = tau_new

        return tau

    def em(self, max_it=50, tolerance=1e-10, upd_params=True, verbose=True,
           tau=None, log_path=False, max_it_fp=50):
        if tau == None:
            tau = self.tau
        tau = tau.to(self.device)
        self.X = self.X.to(self.device)
        start_time = time.time()
        logs_like = []
        prev_value = -float('inf')
        for _ in trange(max_it) if verbose else range(max_it):
            tau = torch.clamp(tau, min=1e-5, max=1 - 1e-5)
            alpha, pi = self.comp_alpha_pi(tau)  # M step
            if torch.isnan(alpha).any():
                print("Nans in alpha")
            if torch.isnan(pi).any():
                print("Nans in pi")
            tau = self._fixed_point_algorithm(alpha, pi, tau, tolerance, max_it_fp)  # E step
            tau = torch.clamp(tau, min=1e-5, max=1 - 1e-5)
            if torch.isnan(tau).any():
                print("Nans in tau")
            likeli = self._likelihood(alpha, pi, tau)  # Compute likelihood
            logs_like.append(likeli.item())
            if abs(likeli - prev_value) < tolerance:
                break
            prev_value = likeli

        self.time_passed = time.time() - start_time
        alpha, pi, tau = alpha.cpu(), pi.cpu(), tau.cpu()
        self.X = self.X.cpu()
        if upd_params:
            self.alpha, self.pi, self.tau = alpha.cpu().numpy(), pi.cpu().numpy(), tau

        if log_path:
            return (alpha.cpu().numpy(), pi.cpu().numpy(), tau, logs_like)

        return alpha.numpy(), pi.numpy(), tau

    def _ICL(self, tau, Q, init=False):
        if init:
            self.__init__(self.X.cpu().numpy(), self.N, Q, device=self.device)
            self.em()

        alpha, pi = self.comp_alpha_pi(tau)
        n = self.N
        term = (1 / 4) * Q * (Q + 1) * np.log((n * (n - 1)) / 2) + ((Q - 1) / 2) * np.log(n)
        return self._likelihood(alpha, pi, tau).item() - term
    
    def _BIC(self, tau, Q):
        alpha, pi = self.comp_alpha_pi(tau)
        return -2*self._likelihood(alpha, pi, tau) + (Q*(Q+3))* np.log(self.N*(self.N-1)/2)

    def _AIC(self, tau, Q):
        alpha, pi = self.comp_alpha_pi(tau)
        return -2*self._likelihood(alpha, pi, tau) + Q*(Q+3)        

    def discrete_distribution(self, x):
        cum_dist = torch.tensor(x.cumsum(0), dtype=torch.float)
        rand_val = np.random.rand()
        return torch.searchsorted(cum_dist, rand_val).item()

    def plot_from_tau(self, tau, determinist=True):
        G = nx.from_numpy_array(self.X.cpu().numpy().astype(int))
        #tau = self.tau.cpu().numpy()
        classes = [i for i in range(self.K)]
        for i, x in enumerate(tau):
            G.nodes[i]['cluster'] = np.argmax(x) if determinist else self.discrete_distribution(x)
        colors = {classe: plt.cm.tab10(i) for i, classe in enumerate(classes)}
        node_colors = [colors[G.nodes[i]['cluster']] for i in range(len(tau))]
        pos = nx.spring_layout(G)  
        plt.figure(figsize=(8, 6))        
        nx.draw(G, pos, with_labels=False, node_size=100, node_color=node_colors, font_size=15, font_weight='bold', edge_color='gray')
        plt.show()

    def loss(self, G, determinist = True):
        #if determinist : return the number of mismatch
        #else : return the sum of 1 - the probability of correct match 
        perm = list(permutations([i for i in range(self.K)]))
        minS = self.N
        minp = perm[0]
        
        for p in list(perm): 
            S = 0
            if determinist:
                for i, x in enumerate(self.tau):
                    if np.argmax(x.cpu()) != p[G.nodes[i]['block']]:
                        S += 1
            else:
                for i, x in enumerate(self.tau):
                    S += (1-x[p[G.nodes[i]['block']]])
            minS = min(S,minS)
            minp = p

        return minS

    def plot_preds_adjancy(self, tau):
        """
        Visualise une matrice d'adjacence regroupée selon les classes des prédites.

        adj (np.ndarray) : La matrice d'agency carrée (N x N).

        """

        classes = np.array(tau.argmax(dim=1).cpu())

        sorted_indices = np.argsort(classes)

        adj_sorted = self.X[np.ix_(sorted_indices, sorted_indices)].cpu().numpy()

        unique_classes, counts = np.unique(classes, return_counts=True)
        cumulative_indices = np.cumsum(counts)

        plt.figure(figsize=(8, 8))
        plt.imshow(-adj_sorted, cmap="gray", extent=(0, adj_sorted.shape[1],
                   adj_sorted.shape[0], 0))
        plt.axis("off")

        for index in cumulative_indices:
            plt.axhline(y=index, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=index, color='red', linestyle='-', linewidth=1)

        plt.show()

    def em_parallelised(self, num_inits=5, max_it=50, tolerance=1e-10,
                        upd_params=True, return_params=False, init=""):

        results = []
        seeds = range(num_inits)
        m_it = [max_it] * num_inits
        tols = [tolerance] * num_inits
        inits = [init] * num_inits
        with ThreadPoolExecutor(max_workers=num_inits) as executor:
            results = list(executor.map(self.single_em_run, seeds, m_it, tols, inits))

        best_res = max(results, key=lambda x: x['likelihood'])
        if upd_params:
            self.alpha = best_res["alpha"]
            self.pi = best_res["pi"]
            self.tau = best_res["tau"]
            self.logs_like = best_res["logs_like"]
        
        if return_params:
            return best_res

        return best_res["tau"]

    def single_em_run(self, seed=42, max_it=50, tolerance=1e-10, init="", verbose=False):
        torch.manual_seed(seed)

        if init == "Kmeans" or init == "K-means":
            tau_init = self._init_K_means()
        
        elif init == "Spectral":
            tau_init = self._init_spectral()
        
        elif init == "RandomSparse" or init == "Sparse":
            tau_init = self._init_tau_sparse()

        else:
            tau_init = self._init_tau()
        alpha, pi, tau, logs_like = self.em(tau=tau_init, max_it=max_it, tolerance=tolerance,
                                            verbose=verbose, log_path=True)
        likelihood = self._likelihood(torch.tensor(alpha, device=self.device), 
                                      torch.tensor(pi, device=self.device), 
                                      tau)
        return {
            'alpha': alpha,
            'pi': pi,
            'tau': tau,
            'likelihood': likelihood.item(),
            'logs_like': logs_like
        }

    def full_proc(self, list_K=[8], n_parralels=20, max_it=30, criterion="ICL", init=""):
        all_results = {}
        for K in tqdm(list_K):
            self.K = K
            all_results[K] = self.em_parallelised_2(num_inits=n_parralels, max_it=max_it, 
                                                    upd_params=False, return_params=True, init=init)
            self.X = self.X.to('cpu')
            all_results[K]["ICL"] = self._ICL(all_results[K]["tau"], K)
            all_results[K]["BIC"] = self._BIC(all_results[K]["tau"], K)
            all_results[K]["AIC"] = self._AIC(all_results[K]["tau"], K)
        return all_results

    def em_parallelised_2(self, num_inits=5, max_it=50, tolerance=1e-10, 
                          upd_params=True, return_params=False, init="", verbose=True):
        manager = mp.Manager()
        results = manager.list()

        processes = []
        for seed in trange(num_inits) if verbose else range(num_inits):
            p = mp.Process(target=worker, args=(seed, self, results, init, max_it, tolerance))
            p.start()
            processes.append(p)

        for p in tqdm(processes) if verbose else processes:
            p.join()
        self.all_res = results

        best_res = max(results, key=lambda x: x['likelihood'])
        if upd_params:
            self.alpha = best_res["alpha"]
            self.pi = best_res["pi"]
            self.tau = best_res["tau"]
            self.logs_like = best_res["logs_like"]

        if return_params:
            return best_res

        return best_res["tau"]

    def plot_logs_path(self, title="", max_l=1000):
        plt.figure(figsize=(12, 6))
        for idx in self.all_res:
            plt.step(range(len(idx["logs_like"][:max_l])), idx["logs_like"][:max_l], where="post")
        plt.title(title)
        plt.ylabel("Log-Likelihood")
        plt.grid(True)
        plt.show()
