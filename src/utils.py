import os
import pickle
from collections import defaultdict
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_auc_score




def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)
        wf.close()

def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as Rf:
        return pickle.load(Rf,encoding='bytes')

def get_data(pairs, paths_dict, p):
    paths_list = []
    label_list = []
    users = []
    items = []
    for pair in pairs:
        if len(paths_dict[(pair[0], pair[1])]):
            paths = paths_dict[(pair[0], pair[1])]

            if len(paths) >= p:
                indices = np.random.choice(len(paths), p, replace=False)
            else:
                indices = np.random.choice(len(paths), p, replace=True)

            paths_list.append([paths[i] for i in indices])
        else:
            paths_list.append([])

        label_list.append(pair[2])
        users.append(pair[0])
        items.append(pair[1])
    return paths_list, label_list, users, items

# metric

def cal_group_auc(model, pairs, paths_dict, args, relation_dict):
    """Calculate group auc"""
    model.eval()
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    group_score1 = defaultdict(lambda: [])
    group_truth1 = defaultdict(lambda: [])
    score_truth = defaultdict(lambda: [])
    pred_label = []
    paths_list, true_label, users, items = get_data(pairs, paths_dict, args.p)
    for i in range(0, len(pairs), args.batch_size):
        predicts = model(paths_list[i: i + args.batch_size], relation_dict)
        pred_label.extend(predicts.cpu().detach().numpy().tolist())
    for i in range(len(users)):
        user_id = users[i]
        score = pred_label[i]
        truth = true_label[i]
        group_score1[user_id].append(score)
        group_truth1[user_id].append(truth)
        key_list = list(group_score1.keys())
    for i in range(len(key_list)):
        user_id = key_list[i]
        for j in range(len(group_score1[user_id])):
            score_truth[group_score1[user_id][j]] = group_truth1[user_id][j]
        score_truth = dict([(m, n) for m, n in sorted(score_truth.items())])
        score_list = list(score_truth.keys())
        truth_list = list(score_truth.values())
        for k in range(len(score_list)):
            group_score[user_id].append(score_list[k])
            group_truth[user_id].append(truth_list[k])
        score_truth.clear()
    group_flag = defaultdict(lambda: False)

    for user_id in set(users):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            
    if impression_total != 0:
        group_auc = float(total_auc) / impression_total
        
    else:
        group_auc = 0
    group_auc = round(group_auc, 4)
    return group_auc


def group_auc(model, rec, paths_dict, relation_dict, args):
    model.eval()
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for user in rec:
        pairs = [(user, item, 0) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, args.p)
        items1 = set(items)
        items = list(items1)
        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()
        item_scores = {items[i]: predict_list[i] for i in range(len(items))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: args.topk]
        if items[-1] in item_list:
            _[item_list.index(items[-1])] = 1

        for item in item_list:
            group_score[user].append(item_scores[item])
        group_truth[user] = _[:args.topk]

    group_flag = defaultdict(lambda: False)

    for user_id in rec:
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0

    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    if impression_total != 0:
        group_auc = float(total_auc) / impression_total
        
    else:
        group_auc = 0
    group_auc = round(group_auc, 4)
    return group_auc

def get_hit(gt_item, pred_items):

    if gt_item in pred_items:
        return 1
    else:
        return 0


def get_ndcg(gt_item, pred_items):

    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    else:
        return 0

def eval_topk(model, rec, paths_dict, relation_dict, p, topk ):
    HR, NDCG = [], []
    model.eval()
    for user in rec:

        pairs = [(user, item, -1) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, p)

        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()

        item_scores = {items[i]: predict_list[i] for i in range(len(pairs))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]
        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    return np.mean(HR), np.mean(NDCG)


def eval_topk1(model, rec, paths_dict, relation_dict, p):
    precision_list = []
    model.eval()
    for user in rec:
        pairs = [(user, item, -1) for item in rec[user]]
        paths_list, _, users, items = get_data(pairs, paths_dict, p)
        predict_list = model(paths_list, relation_dict).cpu().detach().numpy().tolist()
        item_scores = {items[i]: predict_list[i] for i in range(len(pairs))}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        precision_list.append([len({items[-1]}.intersection(item_list[: k])) / k for k in [1, 2, 3, 4, 5, 10, 20]])##.intersection返回多个集合（集合的数量大于等于2）的交集，
    model.train()
    return np.array(precision_list).mean(axis=0).tolist()


def eval_ctr(model, pairs, paths_dict, args, relation_dict):

    model.eval()
    pred_label = []
    paths_list, true_label, users, items = get_data(pairs, paths_dict, args.p)
    for i in range(0, len(pairs), args.batch_size):
        predicts = model(paths_list[i: i+args.batch_size], relation_dict)
        pred_label.extend(predicts.cpu().detach().numpy().tolist())
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return auc, acc
