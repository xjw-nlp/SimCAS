from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer
from datasets import load_dataset, load_from_disk
import random


MAX_LENGTH = 16384


def to_cuda(batch, gpuid):
    for n in batch:
        if 'ids' in n:
            batch[n] = batch[n].to(gpuid)


class AgentDataset(Dataset):
    def __init__(self, args, model_type, dataset_name, data_type='train',  max_input_len=16384, max_output_len=1024, tokenizer=None):
        if tokenizer:
            self.tok = tokenizer
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
                
        self.hf_dataset = load_from_disk(dataset_name)[data_type]
        self.ds_name = dataset_name
        if data_type == 'train':
            self.max_input_len = max_input_len
        else:
            self.max_input_len = MAX_LENGTH
            
        self.max_output_len = max_output_len
        self.data_type = data_type
        self.config = args.config
        self.chunk_len = args.chunk_len
        self.is_offline = args.is_offline
        print(self.tok.get_added_vocab())

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        if self.is_offline:
            input_ids = torch.tensor(entry['input_ids'][:self.max_input_len // self.chunk_len])
            input_texts = entry['input_texts']
            output_texts = entry['output_texts']
        else:
            if self.config == 'multinews':
                input_texts = entry['document'].split('|||||')
                random.shuffle(input_texts)
                input_texts = '</s>'.join(input_texts)
                output_texts = entry['summary']
            elif self.config == 'wcep':
                input_texts = entry['document'].split('</s>')
                random.shuffle(input_texts)
                input_texts = '</s>'.join(input_texts)
                output_texts = entry['summary']
            elif self.config == 'arxiv':
                input_texts = entry['article']
                output_texts = entry['abstract']
            elif self.config == 'pubmed':
                input_texts = entry['article']
                output_texts = entry['abstract']
            elif self.config == 'govreport':
                input_texts = entry['report']
                output_texts = entry['summary']
            elif self.config == 'summscreen':
                input_texts = '\n'.join(entry['Transcript'])
                output_texts = entry['Recap'][0]
            elif self.config == 'nrtv':
                text, text_sum = entry['document']['text'], entry['document']['summary']['text']
                question = entry['question']['text']
                input_texts = f"Answer the question: {question}\n\nSummary: {text_sum}\n\nSource: {text}"
                output_texts = [item['text'] for item in entry['answers']]
            else:
                raise ValueError("Dataset name does not match the examples here")
        
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            input_shape = input_ids.shape
            if input_shape[-1] > self.chunk_len:
                add_num = (input_shape[-1] - 3) // (self.chunk_len - 2)
                _input_ids = input_ids.new_ones(1, (add_num + 1) * (self.chunk_len - 2)) * self.tok.pad_token_id
                _input_ids[:, 0: input_shape[-1] - 2] = input_ids[:, 1: -1]
                _input_ids = _input_ids.reshape(add_num + 1, (self.chunk_len - 2))
                input_ids = torch.nn.functional.pad(_input_ids, (1, 1), value=self.tok.bos_token_id)
                input_ids[:, -1] = self.tok.eos_token_id

        if self.config == 'nrtv':
            output_ids = self.tok.batch_encode_plus([random.choice(output_texts)], truncation=True, max_length=self.max_output_len, return_tensors='pt', padding=True)['input_ids']
        else:
            output_ids = self.tok.batch_encode_plus([output_texts], truncation=True, max_length=self.max_output_len, return_tensors='pt', padding=True)['input_ids']
        return input_ids, output_ids, input_texts, output_texts


def collate_mp_agent(batch, pad_token_id):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    input_ids = [x[0] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in input_ids])
    input_ids = [pad(x, max_len) for x in input_ids]
    input_ids = torch.stack(input_ids)

    output_ids = [x[1][0] for x in batch]
    output_ids = pad([x for x in output_ids])

    input_texts = [x[2] for x in batch]
    output_texts = [x[3] for x in batch]
    result = {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "input_texts": input_texts,
        "output_texts": output_texts,
        }
    return result