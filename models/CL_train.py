import transformers
import json
import random
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import TrainingArguments
from datasets import Dataset
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CL_Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        n = len(inputs['query'])
        total_loss = 0
        for i in range(n):
            query = inputs['query'][i]
            pos = inputs['pos'][i]
            neg = inputs['neg'][i]
            all_inputs = {
                'input_ids': torch.concat([query['input_ids'], pos['input_ids'], neg['input_ids']], dim=0),
                'attention_mask': torch.concat([query['attention_mask'], pos['attention_mask'], neg['attention_mask']], dim=0),
            }
            
            outputs = model.module.encoder(**all_inputs).last_hidden_state
            outputs_query = outputs[0:1, :, :]
            outputs_pos = outputs[1:5, :, :]
            outputs_neg = outputs[5:21, :, :]

            query_idx = torch.sum(query['attention_mask']==1, dim=1)
            pos_idx = torch.sum(pos['attention_mask']==1, dim=1)
            neg_idx = torch.sum(neg['attention_mask']==1, dim=1)

            query_vec = [
                outputs_query[0, query_idx[0]-2, :]
            ]
            pos_vec = [
                outputs_pos[i, pos_idx[i]-2, :]
                for i in range(4)
            ]
            neg_vec = [
                outputs_neg[i, neg_idx[i]-2, :]
                for i in range(16)
            ]
            query_vec = torch.stack(query_vec)
            query_vec = query_vec / torch.norm(query_vec, dim=1, keepdim=True)
            pos_vec = torch.stack(pos_vec)
            pos_vec = pos_vec / torch.norm(pos_vec, dim=1, keepdim=True)
            neg_vec = torch.stack(neg_vec)
            neg_vec = neg_vec / torch.norm(neg_vec, dim=1, keepdim=True)
            temp = 1
            pos_matrix = torch.exp(torch.mm(query_vec, pos_vec.T) / temp)
            neg_matrix = torch.exp(torch.mm(query_vec, neg_vec.T) / temp)
            pos_sum = torch.sum(pos_matrix)
            neg_sum = torch.sum(neg_matrix)

            loss = -torch.log(pos_sum / (pos_sum + neg_sum))
            total_loss += loss
        
        return total_loss


class CL_DataCollator:
    tokenizer: T5Tokenizer
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, features, return_tensors='None'):
        query_list = []
        pos_list = []
        neg_list = []
        for x in features:
            s0 = 0
            s1 = s0 + len(x['query'])
            s2 = s1 + len(x['pos'])
            s3 = s2 + len(x['neg'])
            all_text = []
            all_text.extend(x['query'])
            all_text.extend(x['pos'])
            all_text.extend(x['neg'])
            tokenized_text = self.tokenizer(all_text, return_tensors='pt', padding=True)
            query = {
                'input_ids': tokenized_text['input_ids'][s0:s1, :],
                'attention_mask': tokenized_text['attention_mask'][s0:s1, :],
            }
            pos = {
                'input_ids': tokenized_text['input_ids'][s1:s2, :],
                'attention_mask': tokenized_text['attention_mask'][s1:s2, :],
            }
            neg = {
                'input_ids': tokenized_text['input_ids'][s2:s3, :],
                'attention_mask': tokenized_text['attention_mask'][s2:s3, :],
            }
            query_list.append(query)
            pos_list.append(pos)
            neg_list.append(neg)

        return {
            'query': query_list,
            'pos': pos_list,
            'neg': neg_list,
        }


def inversion_template(x):
    return '[inversion]' + x + '[/inversion]'

def run_CL_train(args):
    with open(args.train_dataset_path, 'r') as f:
        data = json.load(f)
        data = [inversion_template(y) for x in data for y in x['output']]
    
    n = len(data)

    cmp_k = 4

    comparison_data = []
    for i in range(n):
        t = i//4

        pos_samples = []
        neg_samples = []
        for j in range(4):
            pos_samples.append(data[t*4+j])
        for j in range(4*cmp_k):
            neg_samples.append(data[random.randint(0,n-1)])
        comparison_data.append({
            'query': [data[i]],
            'pos': pos_samples,
            'neg': neg_samples,
        })

    query_list = [x['query'] for x in comparison_data]
    pos_list = [x['pos'] for x in comparison_data]
    neg_list = [x['neg'] for x in comparison_data]
    train_ds = Dataset.from_dict({
        'query': query_list,
        'pos': pos_list,
        'neg': neg_list,
    })

    training_args = TrainingArguments(
        output_dir='.cache',
        num_train_epochs=args.contrastive_learning_train_epoch,
        learning_rate=args.contrastive_learning_learning_rate,
        lr_scheduler_type="constant_with_warmup",
        gradient_accumulation_steps=8,
        per_device_train_batch_size=4,
        deepspeed="ds_config.json",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    
    model_path = args.encoder_model_path
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    trainer = CL_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=CL_DataCollator(tokenizer),
    )

    trainer.train()

    model.save_pretrained(args.encoder_save_path, safe_serialization=False)
    tokenizer.save_pretrained(args.encoder_save_path)
    model.to('cpu')