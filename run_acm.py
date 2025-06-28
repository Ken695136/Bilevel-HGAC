import time
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.data import load_ACM_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch, parse_mask
from model.MAGNN_nc_mb_remove_dual import MAGNN_nc_mb_AC
from architect import Architect
from utils.higher_order import embedding_adj


ap = argparse.ArgumentParser(description='MAGNN-AC testing for the ACM dataset')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is DBLP.')
ap.add_argument('--feats-opt', type=str, default='011', help='010 means 1 type nodes use our processed feature.')
ap.add_argument('--feats-drop-rate', type=float, default=0.01, help='The ratio of attributes to be dropped.')
ap.add_argument('--loss-lambda', type=float, default=0.2, help='Coefficient lambda to balance loss.')
ap.add_argument('--cuda', action='store_true', default=True, help='Using GPU or not.')
ap.add_argument('--w_lr',type=float,default=0.005,help='learning rate of hgnn parameter')
ap.add_argument('--w_decay',type = float,default=0.01,help='weight decay of hgnn parameter')
ap.add_argument('--comp_lr',type=float,default=0.01,help='learning rate of completion parameter')
ap.add_argument('--comp_decay',type = float,default=0.1,help='weight decay of completion parameter')
ap.add_argument('--w_momentum',type=float,default=0.9)
ap.add_argument('--dropout_rate',type=float,default=0.0)
ap.add_argument("--topk",type=int,default=5)
# ap.add_argument('--cof_dual', type=float, default=0.5)
args = ap.parse_args()
print(args)

hidden_dim = args.hidden_dim
num_heads = args.num_heads
attn_vec_dim = args.attn_vec_dim
rnn_type = args.rnn_type
num_epochs = args.epoch
patience = args.patience
batch_size = args.batch_size
neighbor_samples = args.samples
repeat = args.repeat
save_postfix = args.save_postfix
feats_opt = args.feats_opt
feats_drop_rate = args.feats_drop_rate
loss_lambda = args.loss_lambda
is_cuda = args.cuda
dataset=args.save_postfix

w_momentum = args.w_momentum
topk = args.topk
w_lr = args.w_lr
w_decay = args.w_decay
comp_decay = args.comp_decay
comp_lr = args.comp_lr


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
    etypes_list = [[0, 1], [2, 3]]
    num_metapaths = 2
    num_edge_type = 4
    src_node_type = 0

    # Params
    out_dim = 3
    dropout_rate = args.dropout_rate
    device = torch.device('cuda:0' if is_cuda else 'cpu')

    # load data
    adjlists, edge_metapath_indices_list, features_list, emb, adjM, type_mask, labels, train_val_test_idx = load_ACM_data()

    higher_order_matrix = embedding_adj(emb, topk)

    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    in_dims = [features.shape[1] for features in features_list]
    emb_dim = emb.shape[1]
    emb = torch.FloatTensor(emb).to(device)
    higher_order_matrix = torch.FloatTensor(higher_order_matrix).to(device)
    adjM = torch.FloatTensor(adjM).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    print('data load finish')

    svm_macro_avg = np.zeros((7, ), dtype=np.float)
    svm_micro_avg = np.zeros((7, ), dtype=np.float)

    embedding_list=[]
    print('start train with repeat = {}\n'.format(repeat))
    for cur_repeat in range(repeat):
        print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
        net = MAGNN_nc_mb_AC(num_metapaths, num_edge_type, etypes_list, in_dims, emb_dim, hidden_dim, out_dim,
                             num_heads, attn_vec_dim, higher_order_matrix, rnn_type, dropout_rate, is_cuda, feats_opt)

        net.to(device)
        w_optimizer = torch.optim.Adam(net.w_parameter(), lr=w_lr, weight_decay=w_decay)
        comp_optimizer = torch.optim.Adam(net.comp_parameter(), lr=comp_lr, weight_decay=comp_decay)
        print('model init finish\n')

        # training loop
        print('training...')
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        architect = Architect(net,w_momentum,w_decay,)
        for epoch in range(num_epochs):
            # training
            t = time.time()
            net.train()
            train_loss_avg = 0
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                val_idx_batch = val_idx_generator.next()
                val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                val_mask_list, val_feat_keep_idx, val_feat_drop_idx = parse_mask(
                    indices_list=val_indices_list, type_mask=type_mask, num_classes=out_dim,
                    src_type=src_node_type, rate=feats_drop_rate, device=device)

                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
                train_mask_list, train_feat_keep_idx, train_feat_drop_idx = parse_mask(
                    indices_list=train_indices_list, type_mask=type_mask, num_classes=out_dim,
                    src_type=src_node_type, rate=feats_drop_rate, device=device)

                val_input1 = (adjM, features_list, emb, val_mask_list, val_feat_keep_idx, val_feat_drop_idx, src_node_type)
                val_input2 = (val_g_list, type_mask, val_indices_list, val_idx_batch_mapped_list)
                train_input1 = (
                adjM, features_list, emb, train_mask_list, train_feat_keep_idx, train_feat_drop_idx, src_node_type)
                train_input2 = (train_g_list, type_mask, train_indices_list, train_idx_batch_mapped_list)
                comp_optimizer.zero_grad()
                architect.unrolled_backward(train_input1, train_input2, val_input1, val_input2, w_lr, w_optimizer, labels,
                                            train_idx_batch
                                            , val_idx_batch)

                comp_optimizer.step()
                logits, _, _ = net(
                    (adjM, features_list, emb, train_mask_list, train_feat_keep_idx, train_feat_drop_idx, src_node_type),
                    (train_g_list, type_mask, train_indices_list, train_idx_batch_mapped_list))


                logp = F.log_softmax(logits, 1)
                loss_classification = F.nll_loss(logp, labels[train_idx_batch])
                train_loss = loss_classification
                train_loss_avg += loss_classification.item()
                # auto grad
                w_optimizer.zero_grad()
                train_loss.backward()
                w_optimizer.step()

            train_loss_avg /= train_idx_generator.num_iterations()
            train_time = time.time() - t

            # validation
            t = time.time()
            net.eval()
            val_logp = []
            val_loss_avg = 0
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                        indices_list=val_indices_list, type_mask=type_mask, num_classes=out_dim,
                        src_type=src_node_type, rate=feats_drop_rate, device=device)

                    logits, _, _ = net(
                        (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                        (val_g_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp, labels[val_idx_batch])
                    val_loss_avg += val_loss.item()
                val_loss_avg /= val_idx_generator.num_iterations()

            val_time = time.time() - t
            print(
                'Epoch {:05d} | Train_Loss {:.4f} | Train_Time(s) {:.4f} | Val_Loss {:.4f} | Val_Time(s) {:.4f}'.format(
                    epoch, train_loss_avg, train_time, val_loss_avg, val_time))

            # early stopping
            early_stopping(val_loss_avg, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        print('\ntesting...')
        test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_embeddings = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, test_idx_batch, device, neighbor_samples)
                mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                    indices_list=test_indices_list, type_mask=type_mask, num_classes=out_dim,
                    src_type=src_node_type, rate=feats_drop_rate, device=device)
                _, embeddings, _ = net(
                    (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                    (test_g_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
                test_embeddings.append(embeddings)

            test_embeddings = torch.cat(test_embeddings, 0)
            embeddings = test_embeddings.detach().cpu().numpy()
            svm_macro, svm_micro, nmi_mean, ari_mean, purity_mean, ri_mean, f_measure_mean, = evaluate_results_nc(
                embeddings, labels[test_idx].cpu().numpy(), num_classes=out_dim)
            embedding_list.append(test_embeddings.cpu().numpy())

            svm_macro_avg = svm_macro_avg + svm_macro
            svm_micro_avg = svm_micro_avg + svm_micro

    svm_macro_avg = svm_macro_avg / repeat
    svm_micro_avg = svm_micro_avg / repeat

    print('---\nThe average of {} results:'.format(repeat))
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))

