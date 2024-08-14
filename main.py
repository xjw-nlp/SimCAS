from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer, AutoTokenizer
from utils import Recorder
from data_utils import to_cuda, AgentDataset, collate_mp_agent
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
from functools import partial
from model import SimCAS
import logging
from label_smoothing_loss import label_smoothing_loss
from nltk import sent_tokenize, word_tokenize
from config import (
    multinews_setting,
    wcep_setting,
    arxiv_setting,
    govreport_setting,
    pubmed_setting,
    summscreen_setting,
    nrtv_setting,
)
from tqdm import tqdm
import math
import time
import wandb
import evaluate


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def evaluation(args):
    # load data
    if args.config == 'multinews':
        multinews_setting(args)
    elif args.config == 'wcep':
        wcep_setting(args)
    elif args.config == 'govreport':
        govreport_setting(args)
    elif args.config == 'summscreen':
        summscreen_setting(args)
    elif args.config == 'arxiv':
        arxiv_setting(args)
    elif args.config == 'pubmed':
        pubmed_setting(args)
    elif args.config == 'nrtv':
        nrtv_setting(args)
    else:
        raise NotImplementedError("Error dataset name!")

    tok = BartTokenizer.from_pretrained(args.model_type)
    test_set = AgentDataset(args, args.model_type, dataset_name=args.dataset_name, data_type='test', tokenizer=tok, max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    collate_fn = partial(collate_mp_agent, pad_token_id=tok.pad_token_id)
    batch_size = 1
    cnt = 0
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = SimCAS(model_path, tok.pad_token_id, args=args)
    if args.cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device = f'cuda:{args.gpuid[0]}'
    model.eval()

    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))
    
    print(model_name)
    root_dir = "./result/%s" % model_name
    mkdir(root_dir)
    print(args.model_pt)
    if args.config == 'nrtv':
        squad_scorer = evaluate.load('/apdcephfs_qy3/share_1565115/jonxie/metrics/squad')
        em_score = f1_score = 0
        with open(os.path.join(root_dir, "test.out"), 'w') as fout, open(os.path.join(root_dir, "test.target"), 'w') as fref:
            with torch.no_grad():
                for (i, batch) in enumerate(tqdm(dataloader)):
                    if batch['input_ids'].shape[-1] < 10:
                        continue
                    if args.cuda:
                        to_cuda(batch, device)
                    answers = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["input_ids"] != tok.pad_token_id,
                        max_length=args.gen_max_len,
                        min_length=args.gen_min_len,
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in answers]
                    for (hypothesis, refs) in zip(dec, batch['output_texts']):
                        hypothesis = hypothesis.replace("\n", " ")
                        y = process(hypothesis)
                        predictions = [{'prediction_text': "\n".join(y), 'id': '56e10a3be3433e1400422b22'}]
                        per_max_em = per_max_f1 = 0
                        for ref in refs:
                            ref = ref.replace("\n", " ")
                            x = process(ref)
                            references = [{'answers': {'answer_start': [97], 'text': ["\n".join(x)]}, 'id': '56e10a3be3433e1400422b22'}]
                            score = squad_scorer.compute(predictions=predictions, references=references)
                            if score['exact_match'] > per_max_em: per_max_em = score['exact_match']
                            if score['f1'] > per_max_f1: per_max_f1 = score['f1']
                        em_score += per_max_em
                        f1_score += per_max_f1
                        cnt += 1
        em_score = em_score / cnt
        f1_score = f1_score / cnt
        if len(args.gpuid) > 1:
            em_score = torch.FloatTensor([em_score]).to(device)
            dist.all_reduce(em_score, op=dist.reduce_op.SUM)
            em_score = em_score.item() / len(args.gpuid)
            f1_score = torch.FloatTensor([f1_score]).to(device)
            dist.all_reduce(f1_score, op=dist.reduce_op.SUM)
            f1_score = f1_score.item() / len(args.gpuid)
        print("evaluation EM: %.6f, F1: %.6f" % (em_score, f1_score))
    else:
        sample_rouge1, sample_rouge2, sample_rougeL, sample_rougeLsum = 0, 0, 0, 0
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        with open(os.path.join(root_dir, "test.out"), 'w') as fout, open(os.path.join(root_dir, "test.target"), 'w') as fref:
            with torch.no_grad():
                for (i, batch) in enumerate(tqdm(dataloader)):
                    if batch['input_ids'].shape[-1] < 10:
                        continue
                    if args.cuda:
                        to_cuda(batch, device)
                    summaries = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["input_ids"] != tok.pad_token_id,
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for (hypothesis, ref) in zip(dec, batch['output_texts']):
                        hypothesis = hypothesis.replace("\n", " ")
                        ref = ref.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                        fref.write(ref + '\n')
                        fref.flush()
                        x = process(ref)
                        y = process(hypothesis)
                        score = rouge_scorer.score("\n".join(x), "\n".join(y))
                        sample_rouge1 += score["rouge1"].fmeasure
                        sample_rouge2 += score["rouge2"].fmeasure
                        sample_rougeL += score["rougeL"].fmeasure
                        sample_rougeLsum += score["rougeLsum"].fmeasure
                        cnt += 1
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeL = sample_rougeL / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            sample_rougeL = torch.FloatTensor([sample_rougeL]).to(device)
            dist.all_reduce(sample_rougeL, op=dist.reduce_op.SUM)
            sample_rougeL = sample_rougeL.item() / len(args.gpuid)
            sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
        print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f, rougeLsum: %.6f"%(sample_rouge1, sample_rouge2, sample_rougeL, sample_rougeLsum))


def test_qa(gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    cnt = 0
    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))
    squad_scorer = evaluate.load('/apdcephfs_qy3/share_1565115/jonxie/metrics/squad')
    em_score = f1_score = 0
    cnt = 0
    # generation
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(gen_dataloader)):
            if batch['input_ids'].shape[-1] < 10:
                continue
            if args.cuda:
                to_cuda(batch, device)
            answers = _model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["input_ids"] != tok.pad_token_id,
                max_length=args.gen_max_len,
                min_length=args.gen_min_len,
                no_repeat_ngram_size=3,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )
            dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in answers]
            for (hypothesis, refs) in zip(dec, batch['output_texts']):
                hypothesis = hypothesis.replace("\n", " ")
                y = process(hypothesis)
                predictions = [{'prediction_text': "\n".join(y), 'id': '56e10a3be3433e1400422b22'}]
                per_max_em = per_max_f1 = 0
                for ref in refs:
                    ref = ref.replace("\n", " ")
                    x = process(ref)
                    references = [{'answers': {'answer_start': [97], 'text': ["\n".join(x)]}, 'id': '56e10a3be3433e1400422b22'}]
                    score = squad_scorer.compute(predictions=predictions, references=references)
                    if score['exact_match'] > per_max_em: per_max_em = score['exact_match']
                    if score['f1'] > per_max_f1: per_max_f1 = score['f1']
                em_score += per_max_em
                f1_score += per_max_f1
                cnt += 1
    em_score = em_score / cnt
    f1_score = f1_score / cnt
    if len(args.gpuid) > 1:
        em_score = torch.FloatTensor([em_score]).to(device)
        dist.all_reduce(em_score, op=dist.reduce_op.SUM)
        em_score = em_score.item() / len(args.gpuid)
        f1_score = torch.FloatTensor([f1_score]).to(device)
        dist.all_reduce(f1_score, op=dist.reduce_op.SUM)
        f1_score = f1_score.item() / len(args.gpuid)
    print(em_score, f1_score)
    model.train()
    return {
        "em": em_score,
        "f1": f1_score,
        }


def test(gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeL, sample_rougeLsum = 0, 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(gen_dataloader)):
            if batch['input_ids'].shape[-1] < 10:
                continue
            if args.cuda:
                to_cuda(batch, device)
            summaries = _model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["input_ids"] != tok.pad_token_id,
                max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )
            dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            for (hypothesis, ref) in zip(dec, batch['output_texts']):
                hypothesis = hypothesis.replace("\n", " ")
                ref = ref.replace("\n", " ")
                x = process(ref)
                y = process(hypothesis)
                score = rouge_scorer.score("\n".join(x), "\n".join(y))
                sample_rouge1 += score["rouge1"].fmeasure
                sample_rouge2 += score["rouge2"].fmeasure
                sample_rougeL += score["rougeL"].fmeasure
                sample_rougeLsum += score["rougeLsum"].fmeasure
                cnt += 1
    sample_rouge1 = sample_rouge1 / cnt
    sample_rouge2 = sample_rouge2 / cnt
    sample_rougeL = sample_rougeL / cnt
    sample_rougeLsum = sample_rougeLsum / cnt
    if len(args.gpuid) > 1:
        sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
        dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
        sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
        sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
        dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
        sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
        sample_rougeL = torch.FloatTensor([sample_rougeL]).to(device)
        dist.all_reduce(sample_rougeL, op=dist.reduce_op.SUM)
        sample_rougeL = sample_rougeL.item() / len(args.gpuid)
        sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
        dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
        sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    print(sample_rouge1, sample_rouge2, sample_rougeL, sample_rougeLsum)
    model.train()
    return {
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeL": sample_rougeL,
        "sample_rougeLsum": sample_rougeLsum,
        } 


def run(rank, args):
    if args.config == 'multinews':
        multinews_setting(args)
    elif args.config == 'wcep':
        wcep_setting(args)
    elif args.config == 'govreport':
        govreport_setting(args)
    elif args.config == 'summscreen':
        summscreen_setting(args)
    elif args.config == 'arxiv':
        arxiv_setting(args)
    elif args.config == 'pubmed':
        pubmed_setting(args)
    elif args.config == 'nrtv':
        nrtv_setting(args)
    else:
        raise NotImplementedError("Error dataset Name!")
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    # gpuid = args.gpuid[rank]
    gpuid = rank
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        if args.is_wandb:
            wandb.login()
            wandb.init(project=args.project_name, name=args.desc)
        recorder = Recorder(args, args.log, desc=args.desc)
    # build dataloader
    tok = BartTokenizer.from_pretrained(args.model_type)
    train_set = AgentDataset(args, args.model_type, dataset_name=args.dataset_name, data_type='validation', tokenizer=tok, max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    val_set = AgentDataset(args, args.model_type, dataset_name=args.dataset_name, data_type='test', tokenizer=tok, max_input_len=args.max_input_len, max_output_len=args.max_output_len)
    collate_fn = partial(collate_mp_agent, pad_token_id=tok.pad_token_id)
    collate_fn_val = partial(collate_mp_agent, pad_token_id=tok.pad_token_id)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = SimCAS(model_path, tok.pad_token_id, args=args)
    model.initialize()
    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=True)
        else:
            model = model.cuda()
    model.train()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
        
    env_params = []
    agent_params = []
    for name, param in model.named_parameters():
        if 'agent' in name:
            agent_params.append({'params': param, 'lr': 1e-4})
        else:
            env_params.append({'params': param})
            param.requires_grad = False
    
    env_optimizer = optim.Adam(env_params)
    agent_optimizer = optim.Adam(agent_params)
    if is_master:
        recorder.write_config(args, [model], __file__)

    all_step_cnt = 0
    # define evaluation function
    minimum_mle_loss = 1e5
    def eval_fn(*args):
        return 1 - sum(args) / len(args)
    # start training
    
    if is_mp:
        agent_dict = model.module.model.agent_dict
        try:
            agent = model.module.model.model.encoder.agent
        except:
            agent = model.module.model.led.encoder.agent
    else:
        agent_dict = model.model.agent_dict
        try:
            agent = model.model.model.encoder.agent
        except:
            agent = model.model.led.encoder.agent

    is_warmup = True
    for epoch in range(args.epoch):
        env_optimizer.zero_grad()
        agent_optimizer.zero_grad()
        
        avg_mle_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        avg_agent_loss = 0
        avg_all_len = 0
        avg_real_len = 0

        for (i, batch) in enumerate(tqdm(dataloader)):
            if batch["input_ids"].shape[-1] < 10:
                continue
            
            if args.cuda:
                to_cuda(batch, gpuid)
                
            if is_warmup and all_step_cnt == 0:
                for param in model.parameters():
                    param.requires_grad = True
                    
                is_warmup = False
                all_step_cnt == 0
                
            step_cnt += 1
            # forward pass
            output = model(batch["input_ids"], batch["output_ids"])
                
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["output_ids"][:, 1:]  # shift right

            env_loss = mle_fn(probs.transpose(1, 2), gold)
            env_loss /= args.accumulate_step
            avg_all_len += agent_dict['all_len'] / args.accumulate_step
            avg_real_len += agent_dict['real_len'] / args.accumulate_step
            avg_mle_loss += env_loss.item()
                    
            if not is_warmup:
                env_loss.backward()
            
            if is_warmup or (all_step_cnt % args.report_freq == 0 and all_step_cnt % args.eval_interval != 0):
                agent_dict['advantages'] = []
                agent_dict['rewards'] = []
                agent_dict['returns'] = []
                
                num_steps = len(agent_dict['values'])
                gamma = 0.998
                gae_lambda = 0.95
                update_epochs = 2
                mini_batch = 512
                clip_coef = 0.2
                
                with torch.no_grad():
                    next_value = 0
                    lastgaelam = 0
                    valid_reward_num = agent_dict['select_rewards'].size(0)
                    action_sum_arr = []
                    for t in range(num_steps):
                        action_sum_arr.append(torch.sum(agent_dict['actions'][t]).item())
                        agent_dict['rewards'].append(agent_dict['values'][t].new_ones(
                            *agent_dict['values'][t].shape) * agent_dict['skip_rewards'])
                        
                    cur_num = 0
                    for t in range(num_steps):
                        reward = agent_dict['rewards'][t]
                        if cur_num + action_sum_arr[t] <= valid_reward_num:
                            new_num = cur_num + action_sum_arr[t]
                            reward[agent_dict['actions'][t].bool()] = agent_dict['select_rewards'][cur_num: new_num]
                            cur_num = new_num
                            if cur_num == valid_reward_num:
                                break
                        else:
                            final_idx = torch.where(agent_dict['actions'][t].cumsum(dim=-1) == valid_reward_num - cur_num)
                            agent_dict['actions'][t][:, final_idx[1][0].item() + 1:] = 0
                            reward[agent_dict['actions'][t].bool()] = agent_dict['select_rewards'][cur_num:]
                            break
                        
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextvalues = next_value
                        else:
                            nextvalues = agent_dict['values'][t + 1].mean(1)
                            
                        delta = agent_dict['rewards'][t] + gamma * nextvalues - agent_dict['values'][t]
                        advantages = lastgaelam = delta + gamma * gae_lambda * lastgaelam
                        returns = advantages + agent_dict['values'][t]
                        agent_dict['advantages'].append(advantages)
                        agent_dict['returns'].append(returns)
                        
                    agent_dict['advantages'].reverse()
                    agent_dict['returns'].reverse()
                    
                for _ in range(update_epochs):
                    if num_steps > 1:
                        seq_len = agent_dict['values'][0].size(-1)
                        b_inds = np.arange(seq_len).reshape(1, seq_len).repeat(num_steps, axis=0)
                    else:
                        seq_len = agent_dict['values'][0].size(-1)
                        b_inds = np.arange(seq_len).reshape(1, seq_len)
                        
                    for i in range(num_steps):
                        np.random.shuffle(b_inds[i])
                    
                    count = 0
                    avg_policy_loss = 0
                    avg_value_loss = 0
                    avg_entropy = 0
                    for batch_st in range(0, seq_len, mini_batch):
                        batch_ed = batch_st + mini_batch
                        num_ids = np.arange(num_steps)
                        np.random.shuffle(num_ids)
                        for num_idx in num_ids:
                            mb_inds = b_inds[num_idx, batch_st: batch_ed]
                            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                                agent_dict['obs'][num_idx][0][:, mb_inds], agent_dict['obs'][num_idx][1], agent_dict['actions'][num_idx][:, mb_inds])
                            logratio = newlogprob - agent_dict['logprobs'][num_idx][:, mb_inds]
                            ratio = logratio.exp()
                            
                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                            
                            try:
                                mb_advantages = agent_dict['advantages'][num_idx][:, mb_inds]
                                if mb_advantages.size(-1) > 1:
                                    mb_advantages_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                                else:
                                    mb_advantages_norm = mb_advantages
                                
                                pg_loss1 = -mb_advantages_norm * ratio
                                pg_loss2 = -mb_advantages_norm * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                                
                                v_loss_unclipped = (newvalue - agent_dict['returns'][num_idx][:, mb_inds]) ** 2
                                v_clipped = agent_dict['values'][num_idx][:, mb_inds] + torch.clamp(
                                    newvalue - agent_dict['values'][num_idx][:, mb_inds], -clip_coef, clip_coef)
                                v_loss_clipped = (v_clipped - agent_dict['returns'][num_idx][:, mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_clipped, v_loss_unclipped)
                                v_loss = 0.5 * v_loss_max.mean()
                                
                                entropy_loss = entropy.mean()
                                agent_loss = pg_loss - 0.1 * entropy_loss + v_loss
                                avg_agent_loss += agent_loss.item()
                                avg_policy_loss += pg_loss.item()
                                avg_value_loss += v_loss.item()
                                avg_entropy += entropy_loss.item()
                                agent_loss.backward()
                                count += 1
                                agent_optimizer.step()
                                agent_optimizer.zero_grad()
                            except:
                                print(f'mb_ids: {mb_inds}, ratio: {ratio}, raw advantage: {mb_advantages}, advantage: {mb_advantages_norm}, \
                                      v_clipped: {v_loss_clipped}, v_unclipped: {v_loss_unclipped}, p_loss1: {pg_loss1}, p_loss2: {pg_loss2}, entropy: {entropy_loss}')
                    
                    if is_master:
                        print(f'policy_loss: {avg_policy_loss / count}, value_loss: {avg_value_loss / count}, entropy: {avg_entropy / count}')
                        
                    if approx_kl.item() > 0.1:
                        print('reaching KL threshold')
                        break
            
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in env_optimizer.param_groups:
                    param_group['lr'] = lr
                
                if not is_warmup:
                    env_optimizer.step()
                    env_optimizer.zero_grad()

            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                recorder.print("epoch: %d, batch: %d, agent_loss: %.6f, avg loss: %.6f, avg mle loss: %.6f"
                %(epoch+1, epoch_step, avg_agent_loss / args.report_freq, avg_loss / args.report_freq, avg_mle_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}, location: {env_loss.get_device()}")
                recorder.print()
                if args.is_wandb:
                    wandb.log({'loss': avg_loss / args.report_freq, 'mle_loss': avg_mle_loss / args.report_freq, 'learning_rate': lr,
                                'real_len': avg_real_len / args.report_freq, 'all_len': avg_all_len / args.report_freq})
                avg_mle_loss, avg_loss, avg_agent_loss, avg_real_len, avg_all_len = 0, 0, 0, 0, 0
            
            del env_loss, output, probs

            if all_step_cnt % args.save_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                if is_master:
                    if is_mp:
                        recorder.save(model.module, f"model_{all_step_cnt}.bin")
                    else:
                        recorder.save(model, f"model_{all_step_cnt}.bin")

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                if is_master:
                    if is_mp:
                        recorder.save(model.module, "model_cur.bin")
                    else:
                        recorder.save(model, "model_cur.bin")
                if len(args.gpuid) > 1:
                    dist.barrier()
                if args.config != 'nrtv':
                    result = test(val_dataloader, model, args, tok, gpuid, args.do_sample)
                    mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeL"])
                    if mle_loss < minimum_mle_loss and is_master:
                        minimum_mle_loss = mle_loss
                        if is_mp:
                            recorder.save(model.module, "model_generation.bin")
                        else:
                            recorder.save(model, "model_generation.bin")
                        recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                    if is_master:
                        recorder.print("val generation loss: %.6f"%(mle_loss))
                        recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f, rougeLsum: %.6f"
                        %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeL"], result["sample_rougeLsum"]))
                        if args.is_wandb:
                            wandb.log({'gen_rouge1': result["sample_rouge1"], 'gen_rouge2': result["sample_rouge2"], "gen_rougeL": result["sample_rougeL"], \
                                    "gen_rougeLsum": result["sample_rougeLsum"]})
                else:
                    result = test_qa(val_dataloader, model, args, tok, gpuid, args.do_sample)
                    mle_loss = eval_fn(result["em"], result["f1"])
                    if mle_loss < minimum_mle_loss and is_master:
                        minimum_mle_loss = mle_loss
                        if is_mp:
                            recorder.save(model.module, "model_generation.bin")
                        else:
                            recorder.save(model, "model_generation.bin")
                        recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                    if is_master:
                        recorder.print("val generation loss: %.6f"%(mle_loss))
                        recorder.print("val EM: %.6f, F1: %.6f" % (result["em"], result["f1"]))
                        if args.is_wandb:
                            wandb.log({'EM': result["em"], 'F1': result["f1"]})


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("--is_offline", action="store_true", help="data processing mode")
    parser.add_argument("-p", "--port", type=int, default=12358, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpuid))
    args.gpuid = list(range(len(args.gpuid)))
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
