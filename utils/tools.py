import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import scipy.sparse as sp


def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot

def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    return ri, f_beta


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    purity_list = []
    ri_list = []
    f_measure_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
        purity_score = accuracy(y, y_pred)
        ri, f_measure = get_rand_index_and_f_measure(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
        purity_list.append(purity_score)
        ri_list.append(ri)
        f_measure_list.append(f_measure)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list), \
           np.mean(purity_list), np.std(purity_list), np.mean(ri_list), np.std(ri_list), np.mean(
        f_measure_list), np.std(f_measure_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    repeat = 20
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)
    print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))
    print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01])]))

    print('\nK-means test')
    nmi_mean, nmi_std, ari_mean, ari_std,purity_mean,purity_std,ri_mean,ri_std,f_measure_mean,f_measure_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))
    print('purity:{:.6f}~{:.6f}'.format(purity_mean, purity_std))
    print('ri:{:.6f}~{:.6f}'.format(ri_mean, ri_std))
    print('f_measure:{:.6f}~{:.6f}'.format(f_measure_mean, f_measure_std))

    macro_mean = [x for (x, y) in svm_macro_f1_list]
    micro_mean = [x for (x, y) in svm_micro_f1_list]
    return np.array(macro_mean), np.array(micro_mean), nmi_mean,  ari_mean, purity_mean,ri_mean,f_measure_mean,


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):

    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        # g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        # result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g.to(device))
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list

def normalized_adj(adj):
    # D^-1 * A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print('generate single-normalized adjacency matrix.')
    return norm_adj.tocoo()

def array2ricci(adj_array):
    array = sp.csr_matrix(adj_array)
    ricci_array = array*array*array
    ricci_array = normalized_adj(ricci_array)
    return ricci_array

def select_top_k(adj,k):
    adj = adj.todense()
    s_d=np.argsort(adj[4278:6359,0:4278],axis=1)
    s_a = np.argsort(adj[6359:,0:4278],axis=1)
    news_d = s_d[:, 0:4278 - k]
    news_a = s_a[:, 0:4278 - k]
    for i in range(news_d.shape[0]):
        for j in range(news_d.shape[1]):
            adj[i+4278, news_d[i,j]] = 0
            adj[news_d[i,j],i+4278] = 0
    for i in range(news_a.shape[0]):
        for j in range(news_a.shape[1]):
            adj[i+6359, news_a[i,j]] = 0
            adj[news_a[i,j],i+6359] = 0
    adj[adj > 0] = 1
    return adj


def parse_mask(indices_list, type_mask, num_classes, src_type, rate, device):
    """
    This function can be implemented in a simpler way
    """
    nodes = set()
    for k in range(len(indices_list)):
        indices = indices_list[k].data.cpu().numpy()
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                nodes.add(indices[i, j])
    nodes = [x for x in sorted(nodes)]

    bound = [0]
    for i in range(num_classes):
        bound.append(np.where(type_mask == i)[0][-1]+1)

    mask_list = []
    for i in range(num_classes):
        mask_list.append(np.array(sorted(search(nodes, bound[i], bound[i+1]-1))))

    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(mask_list[src_type].shape[0]), test_size=rate)

    for i in range(num_classes):
        mask_list[i] = torch.LongTensor(mask_list[i]).to(device)
    feat_keep_idx = torch.LongTensor(feat_keep_idx).to(device)
    feat_drop_idx = torch.LongTensor(feat_drop_idx).to(device)
    return mask_list, feat_keep_idx, feat_drop_idx


def search(lst, m, n):

    def search_upper_bound(lst, key):
        low = 0
        high = len(lst) - 1
        if key > lst[high]:
            return []
        if key <= lst[low]:
            return lst
        while low < high:
            mid = int((low + high+1) / 2)
            if lst[mid] < key:
                low = mid
            else:
                high = mid - 1
        if lst[low] <= key:
            return lst[low+1:]

    def search_lower_bound(lst, key):
        low = 0
        high = len(lst) - 1
        if key <= lst[low]:
            return []
        if key >= lst[high]:
            return lst
        while low < high:
            mid = int((low + high) / 2)
            if key < lst[mid]:
                high = mid
            else:
                low = mid + 1
        if key <= lst[low]:
            return lst[:low]
    return list(set(search_upper_bound(lst, m)) & set(search_lower_bound(lst, n)))


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
