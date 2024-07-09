from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer
from datasets import load_dataset, load_from_disk
import random


MAX_LENGTH = 65536


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
        self.context_len = args.chunk_len
        print(self.tok.get_added_vocab())

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]

        if self.config == 'multinews':
            # nll pre-training
            input_texts = entry['document'].split('|||||')
            random.shuffle(input_texts)
            num_document = len(input_texts)
            max_len_per_doc = self.max_input_len // num_document + 1
            input_idx_arr = []
            for i in range(num_document):
                if i:
                    input_idx_arr.append(self.tok.batch_encode_plus([input_texts[i]], truncation=True, max_length=max_len_per_doc, return_tensors='pt', padding=True)['input_ids'][:, 1:])
                else:
                    input_idx_arr.append(self.tok.batch_encode_plus([input_texts[i]], truncation=True, max_length=max_len_per_doc, return_tensors='pt', padding=True)['input_ids'])
            
            input_ids = torch.cat(input_idx_arr, dim=-1)
            output_texts = entry['summary']
        elif self.config == 'wcep':
            # WCEP
            input_texts = entry['document']
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = entry['summary']
        elif self.config == 'arxiv':
            input_texts = entry['article']
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = entry['abstract']
        elif self.config == 'pubmed':
            input_texts = entry['article']
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = entry['abstract']
        elif self.config == 'govreport':
            input_texts = entry['report']
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = entry['summary']
        elif self.config == 'summscreen':
            input_texts = ''.join(entry['Transcript'])
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = entry['Recap'][0]
        elif self.config == 'nrtv':
            text, text_sum = entry['document']['text'], entry['document']['summary']['text']
            question = entry['question']['text']
            input_texts = f"Answer the question: {question}\n\nSummary: {text_sum}\n\nSource: {text}"
            input_ids = self.tok.batch_encode_plus([input_texts], truncation=True, max_length=self.max_input_len, return_tensors='pt', padding=True)['input_ids']
            output_texts = random.choice(entry['answers'])['text']
        else:
            raise ValueError("Dataset name does not match the examples here")
        
        input_shape = input_ids.shape
        if input_shape[-1] > self.context_len:
            add_num = (input_shape[-1] - 3) // (self.context_len - 2)
            _input_ids = input_ids.new_ones(1, (add_num + 1) * (self.context_len - 2)) * self.tok.pad_token_id
            _input_ids[:, 0: input_shape[-1] - 2] = input_ids[:, 1: -1]
            _input_ids = _input_ids.reshape(add_num + 1, (self.context_len - 2))
            input_ids = torch.nn.functional.pad(_input_ids, (1, 1), value=self.tok.bos_token_id)
            input_ids[:, -1] = self.tok.eos_token_id
        
        output_ids = self.tok.batch_encode_plus([output_texts], truncation=True, max_length=self.max_output_len, return_tensors='pt', padding=True)['input_ids']
        return input_ids, output_ids, input_texts, output_texts, entry


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
    entry = [x[4] for x in batch]
    result = {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "input_texts": input_texts,
        "output_texts": output_texts,
        "entry": entry,
        }
    return result