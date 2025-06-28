import pickle
import time
import argparse
import random
from sklearn.model_selection import train_test_split
import dgl
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc
from model.MAGNN_nc_high import MAGNN_nc_AC
from architect_full_batch import Architect
from utils.higher_order import embedding_adj_imdb


ap = argparse.ArgumentParser(description='MAGNN-AC testing for the ACM dataset')
ap.add_argument('--layers', type=int, default=3, help='Number of layers. Default is 2.')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
ap.add_argument('--batch-size', type=int, default=16, help='Batch size. Default is 8.')
ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--save-postfix', default='IMDB', help='Postfix for the saved model and result. Default is DBLP.')
ap.add_argument('--feats-opt', type=str, default='011', help='010 means 1 type nodes use our processed feature.')
ap.add_argument('--feats-drop-rate', type=float, default=0.01, help='The ratio of attributes to be dropped.')
ap.add_argument('--loss-lambda', type=float, default=0.2, help='Coefficient lambda to balance loss.')
ap.add_argument('--cuda', action='store_true', default=True, help='Using GPU or not.')
ap.add_argument('--w_lr', type=float, default=0.05, help='learning rate of HGNN parameter')
ap.add_argument('--w_decay', type=float, default=0.0005, help='weight decay of HGNN parameter')
ap.add_argument('--comp_lr',type=float,default=0.1,help='learning rate of completion parameter')
ap.add_argument('--comp_decay',type = float,default=0.01,help='weight decay of completion parameter')
ap.add_argument('--w_momentum',type=float,default=0.9)
ap.add_argument('--dropout_rate', type=float, default=0.3)
ap.add_argument('--topk', type=int, default=25)
args = ap.parse_args()
print(args)
num_layers = args.layers
hidden_dim = args.hidden_dim
num_heads = args.num_heads
attn_vec_dim = args.attn_vec_dim
rnn_type = args.rnn_type
num_epochs = args.epoch
patience = args.patience
repeat = args.repeat
save_postfix = args.save_postfix
feats_opt = args.feats_opt
feats_drop_rate = args.feats_drop_rate
loss_lambda = args.loss_lambda
is_cuda = args.cuda
dataset=args.save_postfix
w_lr = args.w_lr
w_decay = args.w_decay
comp_lr = args.comp_lr
comp_decay = args.comp_decay
w_momentum = args.w_momentum
dropout_rate = args.dropout_rate
topk = args.topk
with torch.autograd.set_detect_anomaly(True):
    # random seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        print('Using CUDA')
        torch.cuda.manual_seed(seed)

    feats_opt = list(feats_opt)
    feats_opt = list(map(int, feats_opt))
    print('feats_opt: {}'.format(feats_opt))

    # 0 for papers, 1 for authors, 2 for subjects   0-PA 1-AP 2-PS 3-SP
    etypes_lists = [[[0, 1], [2, 3]],
                    [[1, 0], [1, 2, 3, 0]],
                    [[3, 2], [3, 0, 1, 2]]]
    num_metapaths = [2, 2, 2]
    num_edge_type = 4
    src_node_type = 0

    # Params
    out_dim = 3

    device = torch.device('cuda:0' if is_cuda else 'cpu')

    # load data
    nx_G_lists, edge_metapath_indices_lists, features_list, emb, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()
    higher_order_matrix = embedding_adj_imdb(emb, topk)


    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    in_dims = [features.shape[1] for features in features_list]
    emb_dim = emb.shape[1]
    emb = torch.FloatTensor(emb).to(device)
    higher_order_matrix = torch.FloatTensor(higher_order_matrix).to(device)
    adjM = torch.FloatTensor(adjM).to(device)
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for
                                   indices_list in edge_metapath_indices_lists]
    labels = torch.LongTensor(labels).to(device)
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    mask_list = []
    for i in range(out_dim):
        mask_list.append(np.where(type_mask == i)[0])
    for i in range(out_dim):
        mask_list[i] = torch.LongTensor(mask_list[i]).to(device)
    feat_keep_idx, feat_drop_idx = train_test_split(np.arange(features_list[0].shape[0]), test_size=feats_drop_rate)
    feat_keep_idx = torch.LongTensor(feat_keep_idx).to(device)
    feat_drop_idx = torch.LongTensor(feat_drop_idx).to(device)

    print('data load finish')

    svm_macro_avg = np.zeros((7, ), dtype=np.float)
    svm_micro_avg = np.zeros((7, ), dtype=np.float)
    nmi_mean_list = []
    ari_mean_list = []
    purity_mean_list = []
    ri_mean_list = []
    f_measure_mean_list = []

    embedding_list=[]
    print('start train with repeat = {}\n'.format(repeat))
    for cur_repeat in range(repeat):
        print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
        net = MAGNN_nc_AC(num_layers, num_metapaths, num_edge_type, etypes_lists, in_dims, emb_dim, hidden_dim,
                          out_dim, num_heads, attn_vec_dim, higher_order_matrix,rnn_type, dropout_rate, is_cuda, feats_opt)

        net.to(device)
        #两个优化器 一个优化HGNN参数 一个优化补全参数
        w_optimizer = torch.optim.Adam(net.w_parameter(), lr=w_lr, weight_decay=w_decay)
        comp_optimizer = torch.optim.Adam(net.comp_parameter(), lr=comp_lr, weight_decay=comp_decay)
        target_node_indices = np.where(type_mask == 0)[0]
        print('model init finish\n')

        # training loop
        print('training...')
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))

        architect = Architect(net,w_momentum,w_decay,)
        for epoch in range(num_epochs):
            # training
            t = time.time()
            net.train()

            val_input1 = (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type)
            val_input2 = (g_lists, type_mask, edge_metapath_indices_lists)
            train_input1 = (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type)
            train_input2 = (g_lists, type_mask, edge_metapath_indices_lists)
            #先更新补全参数
            comp_optimizer.zero_grad()
            architect.unrolled_backward(train_input1, train_input2, val_input1, val_input2, w_lr, w_optimizer, labels,train_idx,val_idx)
            comp_optimizer.step()
            #计算loss，更新HGNN参数
            logits, _, _ = net(
                (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (g_lists, type_mask, edge_metapath_indices_lists))
            logp = F.log_softmax(logits, 1)
            loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
            train_loss = loss_classification
            # train_loss_avg += loss_classification.item()
            # auto grad
            w_optimizer.zero_grad()
            train_loss.backward()
            w_optimizer.step()
            print('\ttrain_loss: {:.6f} '.format(
                train_loss.item()))

            # train_loss_avg /= train_idx_generator.num_iterations()
            train_time = time.time() - t

            # validation
            t = time.time()
            net.eval()
            with torch.no_grad():
                logits, _, _ = net(
                    (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                    (g_lists, type_mask, edge_metapath_indices_lists))
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            #     val_loss_avg += val_loss.item()
            # val_loss_avg /= val_idx_generator.num_iterations()

            val_time = time.time() - t
            print('\tval_loss: {:.6f} '.format(
                val_loss.item()))
            print('\ttrain time: {} | val time: {}'.format(train_time, val_time))
            # early stopping
            early_stopping(val_loss.item(), net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        print('\ntesting...')
        # test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_embeddings = []
        with torch.no_grad():

            _, embeddings, _= net(
                (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (g_lists, type_mask, edge_metapath_indices_lists))
            svm_macro, svm_micro, nmi_mean, ari_mean, purity_mean, ri_mean, f_measure_mean, = evaluate_results_nc(
                embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
            embedding_list.append(embeddings[test_idx].cpu().numpy())
            svm_macro_avg = svm_macro_avg + svm_macro
            svm_micro_avg = svm_micro_avg + svm_micro
            nmi_mean_list.append(nmi_mean)

            ari_mean_list.append(ari_mean)

            purity_mean_list.append(purity_mean)

            ri_mean_list.append(ri_mean)

            f_measure_mean_list.append(f_measure_mean)


    svm_macro_avg = svm_macro_avg / repeat
    svm_micro_avg = svm_micro_avg / repeat
    with open(f'embedding_{save_postfix}.pkl', 'wb') as tf:
        pickle.dump(embedding_list, tf)
    print('---\nThe average of {} results:'.format(repeat))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))

