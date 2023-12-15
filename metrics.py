import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import csv

def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res

def mrr_at_k(hit,k):
    assert k >= 1
    res = 0
    hit = np.asarray(hit)[:k]
    for i in range(k):
        if hit[i]==1:
            res = 1/(i+1)
            break
    return res

def mean_average_mrr(rs,k):
    #rs is a set of results, and k is the length of each result, i.e., how many items in each recomendation list.
    return np.mean([mrr_at_k(r,k) for r in rs])


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if hit[k]]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))

def mean_average_precision(rs,k):
    #rs is a set of results, and k is the length of each result, i.e., how many items in each recomendation list.
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r,k) for r in rs])


#def dcg_at_k(rel, k):
#    """
#    calculate discounted cumulative gain (dcg)
#    rel: list, element is positive real values, can be binary
#    """
#    rel = np.asfarray(rel)[:k]
#    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
#    return dcg

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.


def Fone(pre, rec):
    if np.mean(pre) + np.mean(rec) > 0:
        return (2.0 * np.mean(pre) * np.mean(rec)) / (np.mean(pre) + np.mean(rec))
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks, filename):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        if filename:
            aa = np.array(rank_indices, dtype=np.int)
            newfilename = "{}_{}.csv".format(filename,k)
            np.savetxt(newfilename, aa[:, :k], delimiter=",", fmt='%4d')        
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = np.mean(precision_at_k_batch(binary_hit, k))
        metrics_dict[k]['recall']    = np.mean(recall_at_k_batch(binary_hit, k))
        metrics_dict[k]['fone']    = Fone(metrics_dict[k]['precision'], metrics_dict[k]['recall'])
        metrics_dict[k]['ndcg']      = np.mean(ndcg_at_k_batch(binary_hit, k))
        metrics_dict[k]['map']      = mean_average_precision(binary_hit, k)
        metrics_dict[k]['mrr']      = mean_average_mrr(binary_hit, k)
    return metrics_dict


