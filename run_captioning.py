import argparse
import shutil
import time
import torch
import os
import json
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.basic_utils import set_seed, get_current_timestamp, pred_writer, load_pred_cap, remove_rows_cols
from utils.logging_utils import setup_logger
from utils.evaluation_utils import evaluate_on_caption, split_pred_parts, split_gt_parts
from datasets.captioning_dataset import CaptioningDataset
from modeling.modeling_bert import BertForBoundaryCaptioning
from pytorch_transformers import BertTokenizer, BertConfig, AdamW, WarmupLinearSchedule


# structure: main() train() test() 3 functions in each run_inference.py
# all datasets code are in dataset, modeling code are in modeling

def get_predict_file(output_dir, args):
    cc = ['captioning', 'pred', 'beam{}'.format(args.num_beams)]
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    return os.path.join(output_dir, '{}.json'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.json')
    fpath = os.path.splitext(predict_file)[0]
    return fpath + '.eval.json'


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    if args.ablation is None:
        ablation = 'full'
    else:
        ablation = 'wo-' + '-'.join(args.ablation)
    checkpoint_dir = os.path.join(args.output_dir, 'cap_checkpoint-{}-{}-{}-{}'.format(
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


def caption_dataloader(args, tokenizer, split, use_gebd=False):
    dataset = CaptioningDataset(args, tokenizer, split, use_gebd)
    if split == 'train':
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

    sampler = make_data_sampler(dataset, shuffle)
    data_loader = DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=samples_per_batch,
        pin_memory=False,
    )
    return data_loader


def ablation_filter(args, inputs):
    ablation = args.ablation
    if ablation is None:
        return inputs

    curr_pointer = inputs['input_ids'].shape[1]

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


def train(args, train_dataloader, val_dataloader, model, tokenizer):
    t_total = len(train_dataloader) * args.num_train_epochs
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
    logger.info("***** Running training for Captioning *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", args.per_gpu_train_batch_size * args.num_gpus)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (boundary_ids, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            model.train()
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': batch[2],
                'frame_feats': batch[3], 'frame_feats_diff': batch[4], 'act_feats': batch[5],
                'act_feats_diff': batch[6], 'masked_pos': batch[7], 'masked_ids': batch[8]
            }
            inputs = ablation_filter(args, inputs)
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            masked_ids = inputs['masked_ids']
            masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

            loss.mean().backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += loss.mean().item()
            global_acc += batch_acc

            global_step += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if global_step % args.logging_steps == 0:
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                            "score: {:.4f} ({:.4f})".format(epoch, global_step,
                                                            optimizer.param_groups[0]["lr"], loss.mean(),
                                                            global_loss / global_step,
                                                            batch_acc, global_acc / global_step)
                            )
            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                # evaluation
                if args.evaluate_during_training:
                    logger.info("Perform evaluation at step: %d" % global_step)
                    evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
                    # with open(evaluate_file, 'r') as f:
                    #     res = json.load(f)
                    # best_score = max(best_score, res['CIDEr'])
                    # res['epoch'] = epoch
                    # res['global_step'] = step
                    # res['best_CIDEr'] = best_score
                    # eval_log.append(res)
                    # with open(args.output_dir + '/eval_logs.json', 'w') as f:
                    #     json.dump(eval_log, f)
    return checkpoint_dir


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args)
    test(args, val_dataloader, model, tokenizer, predict_file)
    pred_dict = load_pred_cap(predict_file)
    gt_dict = val_dataloader.dataset.get_caption()

    # evaluate_file = get_evaluate_file(predict_file)
    gt = split_gt_parts(gt_dict)
    pred = split_pred_parts(pred_dict)

    res_pred_sub = evaluate_on_caption(pred['subject'], gt['subject'])#, outfile=evaluate_file)
    res_pred_bef = evaluate_on_caption(pred['before'], gt['before'])#, outfile=evaluate_file)
    res_pred_aft = evaluate_on_caption(pred['after'], gt['after'])#, outfile=evaluate_file)

    logger.info('\n ****************** Evaluation Results ******************')
    logger.info('Subject -- CIDEr: %f, SPICE: %f, ROUGE_L: %f',
                res_pred_sub['CIDEr'], res_pred_sub['SPICE'], res_pred_sub['ROUGE_L'])
    logger.info('Status_Before -- CIDEr: %f, SPICE: %f, ROUGE_L: %f',
                res_pred_bef['CIDEr'], res_pred_bef['SPICE'], res_pred_bef['ROUGE_L'])
    logger.info('Status_After -- CIDEr: %f, SPICE: %f, ROUGE_L: %f',
                res_pred_aft['CIDEr'], res_pred_aft['SPICE'], res_pred_aft['ROUGE_L'])
    logger.info('Average -- CIDEr: %f, SPICE: %f, ROUGE_L: %f',
                (res_pred_sub['CIDEr'] + res_pred_sub['CIDEr'] + res_pred_sub['CIDEr']) / 3,
                (res_pred_bef['CIDEr'] + res_pred_bef['CIDEr'] + res_pred_bef['CIDEr']) / 3,
                (res_pred_aft['CIDEr'] + res_pred_aft['CIDEr'] + res_pred_aft['CIDEr']) / 3)


def test(args, test_dataloader, model, tokenizer, predict_file):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
                                         tokenizer.pad_token, tokenizer.mask_token, '.'])
    cache_file = predict_file
    model.eval()
    inputs_param = {'is_decode': True,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,

                    # hyper-parameters of beam search
                    'max_length': args.max_token_length,
                    'num_beams': args.num_beams,
                    'top_n_per_beam': args.top_n_per_beam,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                    "repetition_penalty": args.repetition_penalty,  #
                    "length_penalty": args.length_penalty,
                    'do_sample': args.do_sample,
                    "temperature": args.temperature,  # temperature sampling for beam search
                    "top_k": args.top_k,  # param th in top-k sampling
                    "top_p": args.top_p,  # param th in top-p sampling
                    }

    def gen_rows():
        time_meter = 0

        with torch.no_grad():
            for step, (boundary_keys, batch) in tqdm(enumerate(test_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0], 'attention_mask': batch[1], 'obj_feats': batch[2],
                    'frame_feats': batch[3], 'frame_feats_diff': batch[4], 'act_feats': batch[5],
                    'act_feats_diff': batch[6], 'masked_pos': batch[7]
                }
                inputs = ablation_filter(args, inputs)
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for boundary_key, caps, confs in zip(boundary_keys, all_caps, all_confs):
                    res = []
                    if isinstance(boundary_key, torch.Tensor):
                        boundary_key = boundary_key.item()
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    yield boundary_key, res

        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))

    # test and write to cache_file
    pred_writer(gen_rows(), cache_file)


def main():
    parser = argparse.ArgumentParser()
    # basic param
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--yaml_file", default='config/captioning_config.yaml', type=str, required=False,
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
    parser.add_argument('--gpu_ids', type=str, default='1 3 4 5 7')
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    # hyper-param for training
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X steps. Will also perform evaluation.")  # 5000
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial lr.")  # 3e-5
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=16, type=int, help="Workers in dataloader.")  # Q? n_gpu * 2
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")  # 40 enough?
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--tie_weights", default=True,
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=True,
                        help="Whether to freeze word embeddings in Bert")

    # hyper-param for evaluation
    parser.add_argument("--top_n_per_beam", default=2, type=int, help="select top n in per beam.")
    parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per sample")
    parser.add_argument("--do_sample", action='store_true',
                        help="Whether to use sample strategy in evaluation. otherwise use greedy search.")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")

    # param for dataset
    parser.add_argument("--max_token_length", default=90, type=int,
                        help="The max length of caption tokens.")
    parser.add_argument("--max_frame_num", default=10, type=int,
                        help="The max number of frame before or after boundary.")
    parser.add_argument("--max_object_per_frame", default=20, type=int,
                        help="The max object number in single frame.")
    parser.add_argument("--max_action_length", default=3, type=int,
                        help="The max length of action feature, including difference feature.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Probability to mask input sentence during training.")

    # param for modeling
    parser.add_argument("--num_labels", default=2, type=int,
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

    global logger

    logger = setup_logger("captioning", output_dir)
    logger.info("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)

    config_class, model_class, tokenizer_class = BertConfig, BertForBoundaryCaptioning, BertTokenizer
    if args.do_train:
        config = config_class.from_pretrained(args.model_name_or_path)

        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.num_labels = args.num_labels
        config.obj_feature_dim = args.obj_feature_dim
        config.frame_feature_dim = args.frame_feature_dim
        config.act_feature_dim = args.act_feature_dim
        config.tie_weights = args.tie_weights
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
        train_dataloader = caption_dataloader(args, tokenizer, split='train')
        val_dataloader = caption_dataloader(args, tokenizer, split='val')
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info("Evaluate for Captioning after Training")
            test_dataloader = caption_dataloader(args, tokenizer, split='test')
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info("Evaluate for Captioning")
        test_dataloader = caption_dataloader(args, tokenizer, split='test')

        if not args.do_eval:
            predict_file = get_predict_file(checkpoint, args)
            test(args, test_dataloader, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            evaluate(args, test_dataloader, model, tokenizer, checkpoint)
            # logger.info("Evaluation results saved to: {}".format(evaluate_file))


if __name__ == '__main__':
    main()
