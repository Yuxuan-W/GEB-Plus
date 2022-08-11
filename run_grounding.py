import argparse
import shutil
import time
import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.basic_utils import set_seed, get_current_timestamp, remove_rows_cols
from utils.logging_utils import setup_logger
from utils.evaluation_utils import evaluate_grounding
from datasets.grounding_dataset import GroundingDataset, GroundingCorpusDataset, CustomBatchSampler
from modeling.modeling_bert import BertForVideoRetrieval
from pytorch_transformers import BertTokenizer, BertConfig, AdamW, WarmupLinearSchedule


# structure: main() train() test() 3 functions in each run_inference.py
# all datasets code are in dataset, modeling code are in modeling


def get_predict_file(output_dir, args):
    cc = ['grounding', 'pred']
    return os.path.join(output_dir, '{}.json'.format('.'.join(cc)))


def build_tensorboard_writer(args):
    if args.ablation is None:
        ablation = 'full'
    else:
        ablation = 'wo-' + '-'.join(args.ablation)
    tensorboard_dir = os.path.join(args.output_dir, 'grounding_tensorboard-{}-{}'.format(ablation, get_current_timestamp()))
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    os.mkdir(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    return writer


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.json')
    fpath = os.path.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_same_vid_indice(boundary_ids):
    same_vid_indice = dict()
    same_vid_bids = dict()
    for i in range(len(boundary_ids)):
        bid = boundary_ids[i]
        vid = bid[:11]
        if vid not in same_vid_indice:
            same_vid_indice[vid] = []
            same_vid_bids[vid] = []
        same_vid_indice[vid].append(i)
        same_vid_bids[vid].append(bid)
    return same_vid_indice, same_vid_bids


def get_nondiag_gts_pairs(bid_list, gts_list):
    # get those gts pairs which is different from oneself
    nondiag_pairs = []
    for idx in range(len(bid_list)):
        bid = bid_list[idx]
        gts = gts_list[idx]['gt_boundary_id']
        for gt_bid in gts:
            if gt_bid == bid or gt_bid not in bid_list:
                continue
            nondiag_pairs.append([idx, bid_list.index(gt_bid)])
    return nondiag_pairs


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_np(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = np.linalg.norm(a, axis=1)[:, None], np.linalg.norm(b, axis=1)[:, None]
    a_norm = a / np.maximum(a_n, eps * np.ones_like(a_n))
    b_norm = b / np.maximum(b_n, eps * np.ones_like(b_n))
    sim_mt = np.dot(a_norm, b_norm.transpose(1, 0))
    return sim_mt


def norm_softmax_loss(args, x, extra_pairs=None):
    "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
    i_logsm = F.log_softmax(x / args.temperature, dim=1)
    j_logsm = F.log_softmax(x.t() / args.temperature, dim=1)

    # sum over positives
    idiag = torch.diag(i_logsm)
    i_sum = idiag.sum()
    i_num = len(idiag)

    jdiag = torch.diag(j_logsm)
    j_sum = jdiag.sum()
    j_num = len(jdiag)

    if extra_pairs:
        for pairs in extra_pairs:
            i_sum += i_logsm[pairs[0]][pairs[1]]
            i_num += 1
            j_sum += j_logsm[pairs[0]][pairs[1]]
            j_num += 1

    loss_i = i_sum / i_num
    loss_j = j_sum / j_num

    return - loss_i - loss_j


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    if args.ablation is None:
        ablation = 'full'
    else:
        ablation = 'wo-' + '-'.join(args.ablation)
    checkpoint_dir = os.path.join(args.output_dir, 'grounding_checkpoint-{}-{}-{}-{}'.format(
        ablation, epoch, iteration, get_current_timestamp()))
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, os.path.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def grounding_dataloader(args, tokenizer, split, corpus=False, default_shuffle=True, mode='gt'):
    if corpus:
        dataset = GroundingCorpusDataset(args, split, mode)
    else:
        dataset = GroundingDataset(args, tokenizer, split)

    if split == 'train' and default_shuffle:
        shuffle = True
        samples_per_gpu = args.per_gpu_train_batch_size
        samples_per_batch = samples_per_gpu * args.num_gpus
        iters_per_batch = len(dataset) // samples_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} samples per GPU.".format(samples_per_gpu))
        logger.info("Total batch size {}".format(samples_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        samples_per_gpu = args.per_gpu_eval_batch_size
        samples_per_batch = samples_per_gpu * args.num_gpus

    if not shuffle:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        data_loader = DataLoader(
            dataset, num_workers=args.num_workers, sampler=sampler,
            batch_size=samples_per_batch,
            pin_memory=False,
        )
    else:
        data_loader = DataLoader(dataset, num_workers=args.num_workers,
                                 batch_sampler=CustomBatchSampler(batch_size=samples_per_batch, dataset=dataset))
    return data_loader


def pos_neg_transform(inputs):
    for key, value in inputs.items():
        temp_value = []
        for d0 in range(value.shape[0]):
            for d1 in range(value[d0, ...].shape[0]):
                temp_value.append(value[d0, d1, ...])
        inputs[key] = torch.stack(temp_value, dim=0)
    return inputs


def ablation_filter(args, inputs):
    ablation = args.ablation
    if ablation is None:
        return inputs

    if inputs['input_ids'] is not None:
        curr_pointer = inputs['input_ids'].shape[1]
    else:
        curr_pointer = 0

    if 'obj' in ablation:
        inputs['attention_mask'] = remove_rows_cols(inputs['attention_mask'],
                                                    curr_pointer, curr_pointer + inputs['obj_feats'].shape[1],
                                                    curr_pointer, curr_pointer + inputs['obj_feats'].shape[1])
        inputs['obj_feats'] = None
    else:
        curr_pointer += inputs['obj_feats'].shape[1]

    if 'frame' in ablation:
        inputs['attention_mask'] = remove_rows_cols(inputs['attention_mask'],
                                                    curr_pointer, curr_pointer + inputs['frame_feats'].shape[1],
                                                    curr_pointer, curr_pointer + inputs['frame_feats'].shape[1])
        inputs['frame_feats'] = None
    else:
        curr_pointer += inputs['frame_feats'].shape[1]

    if 'frame_diff' in ablation:
        inputs['attention_mask'] = remove_rows_cols(inputs['attention_mask'],
                                                    curr_pointer, curr_pointer + inputs['frame_feats_diff'].shape[1],
                                                    curr_pointer, curr_pointer + inputs['frame_feats_diff'].shape[1])
        inputs['frame_feats_diff'] = None
    else:
        curr_pointer += inputs['frame_feats_diff'].shape[1]

    if 'act' in ablation:
        inputs['attention_mask'] = remove_rows_cols(inputs['attention_mask'],
                                                    curr_pointer, curr_pointer + inputs['act_feats'].shape[1],
                                                    curr_pointer, curr_pointer + inputs['act_feats'].shape[1])
        inputs['act_feats'] = None
    else:
        curr_pointer += inputs['act_feats'].shape[1]

    if 'act_diff' in ablation:
        inputs['attention_mask'] = remove_rows_cols(inputs['attention_mask'],
                                                    curr_pointer, curr_pointer + inputs['act_feats_diff'].shape[1],
                                                    curr_pointer, curr_pointer + inputs['act_feats_diff'].shape[1])
        inputs['act_feats_diff'] = None
    else:
        curr_pointer += inputs['act_feats_diff'].shape[1]

    assert inputs['attention_mask'].shape[1] == curr_pointer, "Num Error in ablation filters"
    return inputs


def train(args, train_dataloader, train_val_dataloader, train_corpus_dataloader, val_dataloader, val_corpus_dataloader, model, tokenizer):
    writer = build_tensorboard_writer(args)
    t_total = len(train_dataloader) * args.num_train_epochs
    gts = train_dataloader.dataset.get_gts()
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    logger.info("***** Running training for Locating Two Stream *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", args.per_gpu_train_batch_size * args.num_gpus)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss = 0, 0.0
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, (boundary_ids, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            model.train()
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': batch[2], 'frame_feats': batch[3],
                'frame_feats_diff': batch[4], 'act_feats': batch[5], 'act_feats_diff': batch[6]
            }
            inputs = ablation_filter(args, inputs)
            cap_embedding, bdy_embedding = model(**inputs)
            same_vid_indice, same_vid_bids = get_same_vid_indice(boundary_ids)

            loss_list = []
            for vid, indices in same_vid_indice.items():
                outputs = sim_matrix(cap_embedding[indices], bdy_embedding[indices])
                bids = same_vid_bids[vid]
                vid_gts = [gts[bid] for bid in bids]
                nondiag_gts_pairs = get_nondiag_gts_pairs(bids, vid_gts)
                loss_item = norm_softmax_loss(args, outputs, nondiag_gts_pairs)
                if loss_item != 0:
                    loss_list.append(loss_item)
            loss_intra = torch.stack(loss_list).mean()

            outputs = sim_matrix(cap_embedding, bdy_embedding)
            bids = boundary_ids
            vid_gts = [gts[bid] for bid in bids]
            nondiag_gts_pairs = get_nondiag_gts_pairs(bids, vid_gts)
            loss_inter = norm_softmax_loss(args, outputs, nondiag_gts_pairs)

            loss = loss_inter + loss_intra
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.item()
            global_step += 1
            writer.add_scalar('Loss/train', loss.item(), global_step)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if global_step % args.logging_steps == 0:
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), ".format
                            (epoch, global_step, optimizer.param_groups[0]["lr"], loss, global_loss / global_step)
                            )
            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                # evaluation
                if args.evaluate_during_training:
                    logger.info("Perform validation at step: %d" % global_step)
                    validate(args, train_val_dataloader, train_corpus_dataloader, val_dataloader, val_corpus_dataloader,
                             model, writer, global_step)
    writer.close()
    return checkpoint_dir


def validate(args, train_dataloader, train_corpus_dataloader,
             val_dataloader, val_corpus_dataloader, model, writer, global_step):


    model.eval()

    with torch.no_grad():

        time_meter = 0

        gts = val_dataloader.dataset.get_gts()
        cap_embed_list = dict()
        ctx_embed_list = dict()

        for step, (boundary_ids, batch) in tqdm(enumerate(val_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            cap_inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': None, 'frame_feats': None,
                'frame_feats_diff': None, 'act_feats': None, 'act_feats_diff': None
            }
            tic = time.time()
            # collect caption embeddings
            cap_embedding, _ = model(**cap_inputs, do_ctx=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                vid = bid[:11]
                if vid not in cap_embed_list:
                    cap_embed_list[vid] = dict(
                        bids=[],
                        embedding=[]
                    )
                cap_embed_list[vid]['bids'].append(bid)
                cap_embed_list[vid]['embedding'].append(cap_embedding[idx, ...])

        all_query = cap_embed_list

        for step, (boundary_ids, timestamps, batch) in tqdm(enumerate(val_corpus_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            ctx_inputs = {
                'input_ids': None, 'attention_mask': batch[0], 'obj_feats': batch[1], 'frame_feats': batch[2],
                'frame_feats_diff': batch[3], 'act_feats': batch[4], 'act_feats_diff': batch[5]
            }
            ctx_inputs = ablation_filter(args, ctx_inputs)
            tic = time.time()
            # collect context embeddings
            _, ctx_embedding = model(**ctx_inputs, do_cap=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                vid = bid[:11]
                if vid not in ctx_embed_list:
                    ctx_embed_list[vid] = dict(
                        bids=[],
                        embedding=[]
                    )
                ctx_embed_list[vid]['bids'].append(bid)
                ctx_embed_list[vid]['embedding'].append(ctx_embedding[idx, ...])

        all_ctx = ctx_embed_list

        target_acc = []
        random_acc = []
        for vid in all_query.keys():
            if len(all_query[vid]['embedding']) <= 1:
                continue
            assert all_query[vid]['bids'] == all_ctx[vid]['bids']
            query = torch.stack(all_query[vid]['embedding'], dim=0)
            ctx = torch.stack(all_ctx[vid]['embedding'], dim=0)
            sims = sim_matrix(query, ctx).cpu().numpy()

            pos_num = 0
            gt_pos_num = 0
            for row_idx in range(sims.shape[0]):
                bid = all_query[vid]['bids'][row_idx]
                gt_bids = gts[bid]['gt_boundary_id']
                gt_indices = [all_query[vid]['bids'].index(gt_bid) for gt_bid in gt_bids]
                if np.argmax(sims[row_idx]) in gt_indices:
                    pos_num += 1
                gt_pos_num += len(gt_indices) / sims.shape[0]
            target_acc.append(pos_num / sims.shape[0])
            random_acc.append(gt_pos_num / sims.shape[0])

        target_acc = np.mean(np.asarray(target_acc))
        random_acc = np.mean(np.asarray(random_acc))
        writer.add_scalar('Acc/val', target_acc, global_step)
        writer.add_scalar('Acc/random_val', random_acc, global_step)


        # train acc
        gts = train_dataloader.dataset.get_gts()
        cap_embed_list = dict()
        ctx_embed_list = dict()

        for step, (boundary_ids, batch) in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            cap_inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': None, 'frame_feats': None,
                'frame_feats_diff': None, 'act_feats': None, 'act_feats_diff': None
            }
            tic = time.time()
            # collect caption embeddings
            cap_embedding, _ = model(**cap_inputs, do_ctx=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                vid = bid[:11]
                if vid not in cap_embed_list:
                    cap_embed_list[vid] = dict(
                        bids=[],
                        embedding=[]
                    )
                cap_embed_list[vid]['bids'].append(bid)
                cap_embed_list[vid]['embedding'].append(cap_embedding[idx, ...])

        all_query = cap_embed_list

        for step, (boundary_ids, timestamps, batch) in tqdm(enumerate(train_corpus_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            ctx_inputs = {
                'input_ids': None, 'attention_mask': batch[0], 'obj_feats': batch[1], 'frame_feats': batch[2],
                'frame_feats_diff': batch[3], 'act_feats': batch[4], 'act_feats_diff': batch[5]
            }
            ctx_inputs = ablation_filter(args, ctx_inputs)
            tic = time.time()
            # collect context embeddings
            _, ctx_embedding = model(**ctx_inputs, do_cap=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                vid = bid[:11]
                if vid not in ctx_embed_list:
                    ctx_embed_list[vid] = dict(
                        bids=[],
                        embedding=[]
                    )
                ctx_embed_list[vid]['bids'].append(bid)
                ctx_embed_list[vid]['embedding'].append(ctx_embedding[idx, ...])

        all_ctx = ctx_embed_list

        target_acc = []
        random_acc = []
        for vid in all_query.keys():
            if len(all_query[vid]['embedding']) <= 1:
                continue
            assert all_query[vid]['bids'] == all_ctx[vid]['bids']
            query = torch.stack(all_query[vid]['embedding'], dim=0)
            ctx = torch.stack(all_ctx[vid]['embedding'], dim=0)
            sims = sim_matrix(query, ctx).cpu().numpy()

            pos_num = 0
            gt_pos_num = 0
            for row_idx in range(sims.shape[0]):
                bid = all_query[vid]['bids'][row_idx]
                gt_bids = gts[bid]['gt_boundary_id']
                gt_indices = [all_query[vid]['bids'].index(gt_bid) for gt_bid in gt_bids]
                if np.argmax(sims[row_idx]) in gt_indices:
                    pos_num += 1
                gt_pos_num += len(gt_indices) / sims.shape[0]
            target_acc.append(pos_num / sims.shape[0])
            random_acc.append(gt_pos_num / sims.shape[0])

        target_acc = np.mean(np.asarray(target_acc))
        random_acc = np.mean(np.asarray(random_acc))
        writer.add_scalar('Acc/train', target_acc, global_step)
        writer.add_scalar('Acc/random_train', random_acc, global_step)


def test(args, test_dataloader, test_corpus_dataloader, model, output_dir, mode):
    predict_file = get_predict_file(output_dir, args)
    evaluate_file = get_evaluate_file(predict_file)

    model.eval()

    with torch.no_grad():

        time_meter = 0

        gts = test_dataloader.dataset.get_gts()
        cap_embed_list = dict()
        ctx_embed_list = dict()

        for step, (boundary_ids, batch) in tqdm(enumerate(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            cap_inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': None, 'frame_feats': None,
                'frame_feats_diff': None, 'act_feats': None, 'act_feats_diff': None
            }
            tic = time.time()
            # collect caption embeddings
            cap_embedding, _ = model(**cap_inputs, do_ctx=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                vid = bid[:11]
                if vid not in cap_embed_list:
                    cap_embed_list[vid] = dict(
                        bids=[],
                        embedding=[]
                    )
                cap_embed_list[vid]['bids'].append(bid)
                cap_embed_list[vid]['embedding'].append(cap_embedding[idx, ...].detach().cpu().numpy())

        all_query = cap_embed_list

        for step, (boundary_ids, timestamps, batch) in tqdm(enumerate(test_corpus_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            ctx_inputs = {
                'input_ids': None, 'attention_mask': batch[0], 'obj_feats': batch[1], 'frame_feats': batch[2],
                'frame_feats_diff': batch[3], 'act_feats': batch[4], 'act_feats_diff': batch[5]
            }
            ctx_inputs = ablation_filter(args, ctx_inputs)
            tic = time.time()
            # collect context embeddings
            _, ctx_embedding = model(**ctx_inputs, do_cap=False)
            time_meter += time.time() - tic
            for idx in range(len(boundary_ids)):
                bid = boundary_ids[idx]
                timestamp = timestamps[idx]
                vid = bid[:11]
                if vid not in ctx_embed_list:
                    ctx_embed_list[vid] = dict(
                        pids=[],
                        timestamp=[],
                        embedding=[]
                    )
                ctx_embed_list[vid]['pids'].append(bid)
                ctx_embed_list[vid]['timestamp'].append(timestamp.item())
                ctx_embed_list[vid]['embedding'].append(ctx_embedding[idx, ...].detach().cpu().numpy())

        all_ctx = ctx_embed_list

        pred_dict = dict()
        for vid in all_query.keys():
            query = all_query[vid]
            ctx = all_ctx[vid]
            # query_embedding = torch.stack(query['embedding'], dim=0)
            # ctx_embedding = torch.stack(ctx['embedding'], dim=0)
            # sims = sim_matrix_np(query_embedding, ctx_embedding).cpu().numpy()
            query_embedding = np.stack(query['embedding'], axis=0)
            ctx_embedding = np.stack(ctx['embedding'], axis=0)
            sims = sim_matrix_np(query_embedding, ctx_embedding)

            for q_idx in range(len(query['bids'])):
                bid = query['bids'][q_idx]
                scores = sims[q_idx]
                assert bid not in pred_dict
                pred_dict[bid] = dict(
                    proposals=[],
                    scores=[],
                    gt=gts[bid[:-1]]['timestamp']
                )
                for c_idx in range(len(ctx['pids'])):
                    pred_dict[bid]['proposals'].append(ctx['timestamp'][c_idx])
                    pred_dict[bid]['scores'].append(scores[c_idx])

        vid_lengths = test_dataloader.dataset.get_video_lengths()
        metric = evaluate_grounding(pred_dict, vid_lengths, mode, outfile=[predict_file, evaluate_file])
    logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))
    logger.info('Evaluation result: {}'.format(str(metric)))
    logger.info('Evaluation result saved to {}'.format(evaluate_file))
    return evaluate_file


def main():
    parser = argparse.ArgumentParser()
    # basic param
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--yaml_file", default='config/locating_two_stream_config.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--eval_model_dir", type=str, default='output/',
                        help="Model directory for evaluation.")
    parser.add_argument("--evaluate_during_training", default=False, action="store_true",
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--ablation", default=None, help="Ablation set, e.g.'obj-frame'")
    parser.add_argument('--gpu_ids', type=str, default='4 6 7')
    parser.add_argument("--per_gpu_train_batch_size", default=48, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=144, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    # hyper-param for training
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=4000,
                        help="Save checkpoint every X steps. Will also perform evaluation.")  # 5000
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial lr.")  # 3e-5
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for gradient clip.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=16, type=int, help="Workers in dataloader.")  # Q? n_gpu * 2
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")  # 40 enough?
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--freeze_embedding", default=True,
                        help="Whether to freeze word embeddings in Bert")

    # param for dataset
    parser.add_argument("--use_gebd", default=False, type=bool,
                        help="The max length of caption tokens.")
    parser.add_argument("--max_token_length", default=90, type=int,
                        help="The max length of caption tokens.")
    parser.add_argument("--max_frame_num", default=10, type=int,
                        help="The max number of frame before or after boundary.")
    parser.add_argument("--max_object_per_frame", default=20, type=int,
                        help="The max object number in single frame.")
    parser.add_argument("--max_action_length", default=3, type=int,
                        help="The max length of action feature, including difference feature.")

    # param for modeling
    parser.add_argument("--num_labels", default=1, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--obj_feature_dim", default=1031, type=int,
                        help="The Object Feature Dimension.")
    parser.add_argument("--frame_feature_dim", default=1026, type=int,
                        help="The Frame Feature Dimension.")
    parser.add_argument("--act_feature_dim", default=2049, type=int,
                        help="The Action Feature Dimension.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument('--temperature', type=float, default=0.05,
                        help="temperature in softmax for loss computing")

    args = parser.parse_args()

    args.gpu_ids = list(map(int, args.gpu_ids.split(' ')))
    args.device = torch.device("cuda:" + str(args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids if torch.cuda.is_available() else "-1"
    args.num_gpus = len(args.gpu_ids)
    assert args.num_gpus <= torch.cuda.device_count(), "Some of GPUs in args are unavailable, check your parameter."

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.ablation is not None:
        args.ablation = args.ablation.split('-')

    if args.use_gebd:
        frame_sampling = 'gebd'
    else:
        frame_sampling = 'all_1s'

    global logger

    logger = setup_logger("locating2stream", output_dir)
    logger.info("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)

    config_class, model_class, tokenizer_class = BertConfig, BertForVideoRetrieval, BertTokenizer
    if args.do_train:
        config = config_class.from_pretrained(args.model_name_or_path)

        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.num_labels = args.num_labels
        config.obj_feature_dim = args.obj_feature_dim
        config.frame_feature_dim = args.frame_feature_dim
        config.act_feature_dim = args.act_feature_dim
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
        checkpoint = args.eval_model_dir
        assert os.path.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    if args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataloader = grounding_dataloader(args, tokenizer, split='train')
        train_val_dataloader = grounding_dataloader(args, tokenizer, split='train', default_shuffle=False)
        train_corpus_dataloader = grounding_dataloader(args, tokenizer, split='train', corpus=True, default_shuffle=False)
        val_dataloader = grounding_dataloader(args, tokenizer, split='val')
        val_corpus_dataloader = grounding_dataloader(args, tokenizer, split='val', corpus=True)
        last_checkpoint = train(args, train_dataloader, train_val_dataloader, train_corpus_dataloader,
                                val_dataloader, val_corpus_dataloader, model, tokenizer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info("Evaluate for Grounding after Training")
            test_dataloader = grounding_dataloader(args, tokenizer, split='test')
            test_corpus_dataloader = grounding_dataloader(args, tokenizer, split='test', corpus=True, mode=frame_sampling)
            test(args, test_dataloader, test_corpus_dataloader, model, last_checkpoint, mode=frame_sampling)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info("Evaluate for Grounding")
        test_dataloader = grounding_dataloader(args, tokenizer, split='test')
        test_corpus_dataloader = grounding_dataloader(args, tokenizer, split='test', corpus=True, mode=frame_sampling)

        if not args.do_eval:
            raise Exception
        else:
            test(args, test_dataloader, test_corpus_dataloader, model, checkpoint, mode=frame_sampling)


if __name__ == '__main__':
    main()
