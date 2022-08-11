import json
import copy
import os
import random

import numpy as np
from builtins import dict
from functools import partial
from scipy.ndimage import filters
from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.spice.spice import Spice


def split_pred_parts(pred):
    pred_dict = dict(
        subject=dict(),
        before=dict(),
        after=dict()
    )
    for bid, cap in pred.items():
        part_list = cap[0].split('/ /')
        pred_dict['subject'][bid] = [part_list[0].split(':')[-1].strip()]
        if len(part_list) < 1:
            pred_dict['subject'][bid] = [' ']
        else:
            pred_dict['subject'][bid] = [part_list[0].split(':')[-1].strip()]
        if len(part_list) < 2:
            pred_dict['before'][bid] = [' ']
        else:
            pred_dict['before'][bid] = [part_list[1].split(':')[-1].strip()]
        if len(part_list) < 3:
            pred_dict['after'][bid] = [' ']
        else:
            pred_dict['after'][bid] = [part_list[2].split(':')[-1].strip()]
    return pred_dict


def split_gt_parts(pred):
    pred_dict = dict(
        subject=dict(),
        before=dict(),
        after=dict()
    )
    for bid, cap in pred.items():
        part_list = cap[0].split('//')
        pred_dict['subject'][bid] = [part_list[0].split(':')[-1].strip()]
        pred_dict['before'][bid] = [part_list[1].split(':')[-1].strip()]
        pred_dict['after'][bid] = [part_list[2].split(':')[-1].strip()]
    return pred_dict


class EvalCap:
    def __init__(self, pred_dict, gt_dict, df):
        self.evalBoundaries = []
        self.eval = dict()
        self.BoundariesToEval = dict()

        self.gts = gt_dict
        self.res = pred_dict
        self.df = df

    def tokenize(self):

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        self.gts = tokenizer.tokenize(self.gts)
        self.res = tokenizer.tokenize(self.res)

    def evaluate(self):
        self.tokenize()

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(self.df), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setBoundaryToEvalBoundaries(scs, self.gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setBoundaryToEvalBoundaries(scores, self.gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalBoundaries()

    def setEval(self, score, method):
        self.eval[method] = score

    def setBoundaryToEvalBoundaries(self, scores, b_ids, method):
        for b_id, score in zip(b_ids, scores):
            if not b_id in self.BoundariesToEval:
                self.BoundariesToEval[b_id] = dict()
                self.BoundariesToEval[b_id]["boundary_id"] = b_id
            self.BoundariesToEval[b_id][method] = score

    def setEvalBoundaries(self):
        self.evalBoundaries = [eval for imgId, eval in self.BoundariesToEval.items()]


def evaluate_on_caption(pred_dict, gt_dict, outfile=None):
    Eval = EvalCap(pred_dict, gt_dict, 'corpus')

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    Eval.evaluate()
    result = Eval.eval

    if outfile:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


class EvalRet:
    def __init__(self, sim_matrix, query_ids, ctx_ids, metrics=None):
        if metrics is None:
            self.metrics = ['mAP', 'r@1', 'r@5', 'r@10', 'r@50']
        self.raw_matrix = sim_matrix
        self.query_ids = query_ids
        self.raw_ctx_ids = ctx_ids
        self.ctx_ids, self.vid_ctx_dict, self.matrix = self.keep_highest()
        self.metric2func = {
            'mAP': self.mean_average_precision,
            'r@1': partial(self.mean_reall_at_k, k=1),
            'r@5': partial(self.mean_reall_at_k, k=5),
            'r@10': partial(self.mean_reall_at_k, k=10),
            'r@50': partial(self.mean_reall_at_k, k=50),
        }

    def keep_highest(self):
        vid_ctx_dict = dict()
        for ids_idx in range(len(self.raw_ctx_ids)):
            b_id = self.raw_ctx_ids[ids_idx]
            vid = b_id[:11]
            if vid not in vid_ctx_dict:
                vid_ctx_dict[vid] = []
            vid_ctx_dict[vid].append(ids_idx)

        ctx_ids = []
        matrix = None
        for vid, ids_list in vid_ctx_dict.items():
            ctx_ids.append(vid)
            max_column = self.raw_matrix[:, ids_list]
            max_column = np.expand_dims(np.max(max_column, axis=1), axis=1)
            if matrix is None:
                matrix = max_column
            else:
                matrix = np.concatenate((matrix, max_column), axis=1)

        assert matrix.shape[1] == len(ctx_ids), 'keep_highest error, column num not equals to ctx num.'

        return ctx_ids, vid_ctx_dict, matrix

    def get_ranking_matrix(self):
        sorted_indices = np.argsort(-self.matrix, axis=1)
        rs = []
        ranked_for_vis = dict()
        for row_idx in range(len(self.query_ids)):
            qid = self.query_ids[row_idx]

            # rank retrived ctx_id for each query
            ranked_ctxid_for_qid = [self.ctx_ids[i] for i in sorted_indices[row_idx].tolist()]
            # GT to be 1; otherwise 0;
            res_qid = np.asarray([int(vid == qid[:11]) for vid in ranked_ctxid_for_qid])
            ap_qid = self.average_precision(res_qid)
            ranked_for_vis.update(
                {qid: {'gt': qid[:11], 'res': ranked_ctxid_for_qid, 'aveP': ap_qid}}
            )
            rs.append(res_qid)

        rs = np.stack(rs, axis=0)

        return rs, ranked_for_vis

    def evaluate(self):
        rs, ranked_for_vis = self.get_ranking_matrix()
        res_dict = dict()
        for met in self.metrics:
            met_func = self.metric2func[met]
            res_dict[met] = 100 * met_func(rs)

        self.res_dict = res_dict
        self.res_rank = ranked_for_vis

    @staticmethod
    def precision_at_k(r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> precision_at_k(r, 3)
        0.33333333333333331
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):
        """Score is average precision (area under PR curve)
        Relevance is binary (nonzero is relevant).
        >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> delta_r = 1. / sum(r)
        >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
        0.7833333333333333
        >>> average_precision(r)
        0.78333333333333333
        Args:
            r: Relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Average precision
        """
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    @staticmethod
    def get_rounded_percentage(float_number, n_floats=2):
        return round(float_number * 100, n_floats)

    def mean_average_precision(self, rs):
        """Score is mean average precision
        Relevance is binary (nonzero is relevant).
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
        >>> mean_average_precision(rs)
        0.78333333333333333
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
        >>> mean_average_precision(rs)
        0.39166666666666666
        Args:
            rs: Iterator of relevance scores (list or numpy) in rank order
                (first element is the first item)
        Returns:
            Mean average precision
        """
        ap = [self.average_precision(r) for r in rs]
        return np.mean(ap)

    @staticmethod
    def mean_reall_at_k(rs, k):
        assert len(rs.shape) == 2, "Ranking score should be of dimension 2."
        n_q, n_ctx = rs.shape

        assert k <= n_ctx, f"Receive k({k}) > n_ctx ({n_ctx}) when calculating recall@{k}."
        return (rs[:, :k].sum(axis=1) / rs.sum(axis=1)).sum() / n_q


def evaluate_on_retrieval(sim_matrix, query_ids, ctx_idx, outfile=None):
    Eval = EvalRet(sim_matrix, query_ids, ctx_idx)
    Eval.evaluate()
    rank = Eval.res_rank
    metric = Eval.res_dict
    if outfile:
        with open(outfile[0], 'w') as fp:
            json.dump(rank, fp, indent=4)
        with open(outfile[1], 'w') as fp:
            json.dump(metric, fp, indent=4)
    else:
        print(metric)
    return rank, metric


class EvalPwl:
    def __init__(self, pred, gt, vid_lengths):
        self.vid_lengths = vid_lengths

        self.pred = dict()
        for bid, scores in pred.items():
            self.pred[bid] = self.get_idx_from_scores_with_gaussian_smoothing(
                gaussian_sigma=1, threshold=0.5, seq_indices=scores['time'], seq_scores=scores['scores'])

        self.gt = dict()
        for bid, gt_list in gt.items():
            self.gt[bid] = []
            for gt_meta in gt_list:
                self.gt[bid].append(gt_meta['timestamp'])

        self.th = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.all_res = dict()
        self.metric = dict()
        for item in self.th:
            self.all_res[item] = dict()

    def evaluate(self):
        for bid, gt_timestamp_list in self.gt.items():
            assert bid in self.pred, 'gt bid not found in prediction'
            pred_timestamp_list = self.pred[bid]
            for th_ratio in self.th:
                threshold = th_ratio * self.vid_lengths[bid[:11]]
                score = self.compute_f1(pred_timestamp_list, gt_timestamp_list, threshold)
                self.all_res[th_ratio][bid] = score
        for th, scores in self.all_res.items():
            score_list = []
            for bid, score in scores.items():
                score_list.append(score)
            avg_score = np.asarray(score_list).mean()
            self.metric[th] = avg_score

        return self.metric

    @staticmethod
    def get_idx_from_scores_with_gaussian_smoothing(gaussian_sigma=1, threshold=0.5, seq_indices=None, seq_scores=None):
        seq_indices = np.array(seq_indices)
        seq_scores = np.array(seq_scores)
        gaussian_smt_scores = filters.gaussian_filter1d(seq_scores, gaussian_sigma)
        #     print(gaussian_smt_scores)
        bdy_indices = []
        internals_indices = []
        for i in range(len(gaussian_smt_scores)):
            if gaussian_smt_scores[i] >= threshold:
                internals_indices.append(i)
            elif gaussian_smt_scores[i] < threshold and len(internals_indices) != 0:
                bdy_indices.append(internals_indices)
                internals_indices = []
            if i == len(gaussian_smt_scores) - 1 and len(internals_indices) != 0:
                bdy_indices.append(internals_indices)

        bdy_indices_in_video = []
        if len(bdy_indices) != 0:
            for internals in bdy_indices:
                center = int(np.mean(internals))
                bdy_indices_in_video.append(seq_indices[center])
        return bdy_indices_in_video

    def compute_f1(self, pred_timestamp_list, gt_timestamp_list, th):
        if not pred_timestamp_list:
            return 0
        num_pos = len(gt_timestamp_list)
        num_det = len(pred_timestamp_list)
        assert num_det > 0
        # calculate distance matrix between a1 and a2, each row represent all detected boundaries
        dist_matrix = np.zeros((len(gt_timestamp_list), len(pred_timestamp_list)))
        for b1_idx in range(len(gt_timestamp_list)):
            dist_matrix[b1_idx] = abs(np.asarray(pred_timestamp_list) - gt_timestamp_list[b1_idx])

        # calculate f1 score based on threshold
        # count tp, each column represents one threshold
        tp = 0
        for b1_idx in range(len(gt_timestamp_list)):
            min_idx = np.argmin(dist_matrix[b1_idx])
            if dist_matrix[b1_idx][min_idx] < th:
                tp += 1
                for i in range(len(gt_timestamp_list)):
                    dist_matrix[i][min_idx] = 10000

        # calculate f1
        fn = num_pos - tp
        fp = num_det - tp
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if (rec + prec) == 0:
            f1 = 0
        else:
            f1 = 2 * rec * prec / (rec + prec)

        return f1


def evaluate_on_locating(pred, gt, vid_lengths, outfile=None):
    Eval = EvalPwl(pred, gt, vid_lengths)
    metric = Eval.evaluate()

    saved_pred = dict()
    for bid, datas in pred.items():
        saved_pred[bid] = dict(
            time=[float(t) for t in datas['time']],
            scores=[float(s) for s in datas['scores']],
            res=[float(t) for t in Eval.pred[bid]]
        )

    if outfile:
        for outpath in outfile:
            if os.path.exists(outpath):
                os.remove(outpath)
        with open(outfile[0], 'w') as fp:
            json.dump(saved_pred, fp, indent=4)
        with open(outfile[1], 'w') as fp:
            json.dump(metric, fp, indent=4)
    else:
        print(metric)

    return metric


class EvalPwl_new:
    def __init__(self, pred, gt, vid_lengths):
        self.vid_lengths = vid_lengths

        self.pred = dict()
        for bid, meta in pred.items():
            self.pred[bid] = [meta['time'][i] for i in range(len(meta['time'])) if meta['scores'][i] >= 0.5]

        self.gt = dict()
        for bid, meta in gt.items():
            self.gt[bid] = meta['timestamp']

        self.th = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.all_res = dict()
        self.metric = dict()
        for item in self.th:
            self.all_res[item] = dict()

    def evaluate(self):
        for bid, gt_timestamp_list in self.gt.items():
            pred_timestamp_list = self.pred[bid]
            for th_ratio in self.th:
                threshold = th_ratio * self.vid_lengths[bid[:11]]
                score = self.compute_f1(pred_timestamp_list, gt_timestamp_list, threshold)
                self.all_res[th_ratio][bid] = score
        for th, scores in self.all_res.items():
            score_list = []
            for bid, score in scores.items():
                score_list.append(score)
            avg_score = np.asarray(score_list).mean()
            self.metric[th] = avg_score

        return self.metric

    def compute_f1(self, pred_timestamp_list, gt_timestamp_list, th):
        if not pred_timestamp_list:
            return 0
        num_pos = len(gt_timestamp_list)
        num_det = len(pred_timestamp_list)
        assert num_det > 0
        # calculate distance matrix between a1 and a2, each row represent all detected boundaries
        dist_matrix = np.zeros((len(gt_timestamp_list), len(pred_timestamp_list)))
        for b1_idx in range(len(gt_timestamp_list)):
            dist_matrix[b1_idx] = abs(np.asarray(pred_timestamp_list) - gt_timestamp_list[b1_idx])

        # calculate f1 score based on threshold
        # count tp, each column represents one threshold
        tp = 0
        for b1_idx in range(len(gt_timestamp_list)):
            min_idx = np.argmin(dist_matrix[b1_idx])
            if dist_matrix[b1_idx][min_idx] < th:
                tp += 1
                for i in range(len(gt_timestamp_list)):
                    dist_matrix[i][min_idx] = 10000

        # calculate f1
        fn = num_pos - tp
        fp = num_det - tp
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if (rec + prec) == 0:
            f1 = 0
        else:
            f1 = 2 * rec * prec / (rec + prec)

        return f1


def evaluate_on_locating_new(pred, gt, vid_lengths, outfile=None):
    Eval = EvalPwl_new(pred, gt, vid_lengths)
    metric = Eval.evaluate()

    saved_pred = dict()
    for bid, datas in pred.items():
        saved_pred[bid] = dict(
            time=[float(t) for t in datas['time']],
            scores=[float(s) for s in datas['scores']],
            res=[float(t) for t in Eval.pred[bid]]
        )

    if outfile:
        for outpath in outfile:
            if os.path.exists(outpath):
                os.remove(outpath)
        with open(outfile[0], 'w') as fp:
            json.dump(saved_pred, fp, indent=4)
        with open(outfile[1], 'w') as fp:
            json.dump(metric, fp, indent=4)
    else:
        print(metric)

    return metric


class EvalPwl_2stream:
    def __init__(self, pred, vid_lengths):
        self.pred = pred
        self.vid_lengths = vid_lengths

        self.th = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.all_res = dict()
        self.metric = dict()
        for item in self.th:
            self.all_res[item] = dict()

    def evaluate(self):
        for bid, meta in self.pred.items():
            time_rank = [x for _, x in sorted(zip(meta['score'], meta['proposals']), reverse=True)]
            random_time_rank = copy.deepcopy(time_rank)
            random.shuffle(random_time_rank)
            time_gt = meta['gt']
            for th_ratio in self.th:
                th = th_ratio * self.vid_lengths[bid[:11]]
                score, random_score = self.compute_mAP(time_rank, random_time_rank, time_gt, th)
                self.all_res[th_ratio][bid] = dict(
                    score=score,
                    random=random_score
                )

        for th, scores in self.all_res.items():
            score_list = []
            random_list = []
            for bid, meta in scores.items():
                score_list.append(meta['score'])
                random_list.append(meta['random'])
            avg_score = np.asarray(score_list).mean()
            random_score = np.asarray(random_list).mean()
            self.metric[th] = dict(
                score=avg_score,
                random=random_score
            )

        return self.metric

    def compute_mAP(self, time_rank, random_time_rank, time_gt, th):
        cnt = 0
        matched = False
        for idx in range(len(time_rank)):
            if matched:
                break
            timestamp = time_rank[idx]
            cnt += 1
            for gt in time_gt:
                if abs(timestamp - gt) <= th:
                    matched = True

        score = 1 / cnt if matched else 0

        cnt = 0
        matched = False
        for idx in range(len(random_time_rank)):
            if matched:
                break
            timestamp = random_time_rank[idx]
            cnt += 1
            for gt in time_gt:
                if abs(timestamp - gt) <= th:
                    matched = True

        random_score = 1 / cnt if matched else 0

        return score, random_score


def evaluate_on_locating2stream(pred, vid_lengths, outfile=None):
    Eval = EvalPwl_2stream(pred, vid_lengths)
    metric = Eval.evaluate()

    saved_pred = dict()
    for bid, datas in pred.items():
        saved_pred[bid] = dict(
            proposals=[float(t) for t in datas['proposals']],
            scores=[float(s) for s in datas['score']],
            gt=[float(g) for g in datas['gt']]
        )

    saved_metric = dict()
    for th, meta in metric.items():
        saved_metric[th] = dict(
            score=float(metric[th]['score']),
            random=float(metric[th]['random'])
        )

    if outfile:
        for outpath in outfile:
            if os.path.exists(outpath):
                os.remove(outpath)
        with open(outfile[0], 'w') as fp:
            json.dump(saved_pred, fp, indent=4)
        with open(outfile[1], 'w') as fp:
            json.dump(saved_metric, fp, indent=4)
    else:
        print(metric)

    return metric