import json
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

from datasets import Dataset

from models.inversion_model import InvEncoder, InvDecoder, InvEncoderDecoder, Projector, get_hidden_size

class FT_DataCollator:
    def __call__(self, features, return_tensors=None):
        # print(len(features))
        encoder_input_ids = []
        encoder_input_attention_mask = []
        decoder_input_ids = []
        decoder_input_attention_mask = []
        labels = []
        labels_attention_mask = []
        encoder_input_length = 0
        decoder_input_length = 0
        labels_length = 0
        for feature in features:
            encoder_input_ids.append(torch.tensor(feature['encoder_input_ids']))
            encoder_input_attention_mask.append(torch.tensor(feature['encoder_input_attention_mask']))
            encoder_input_length = max(encoder_input_length, feature['encoder_input_attention_mask'].count(1))
            decoder_input_ids.append(torch.tensor(feature['decoder_input_ids']))
            decoder_input_attention_mask.append(torch.tensor(feature['decoder_input_attention_mask']))
            decoder_input_length = max(decoder_input_length, feature['decoder_input_attention_mask'].count(1))
            labels.append(torch.tensor(feature['labels']))
            labels_attention_mask.append(torch.tensor(feature['labels_attention_mask']))
            labels_length = max(labels_length, feature['labels_attention_mask'].count(1)+1)
        
        return {
            'encoder_input_ids': torch.stack(encoder_input_ids)[:, :encoder_input_length],
            'encoder_input_attention_mask': torch.stack(encoder_input_attention_mask)[:, :encoder_input_length],
            'decoder_input_ids': torch.stack(decoder_input_ids)[:, :decoder_input_length],
            'decoder_input_attention_mask': torch.stack(decoder_input_attention_mask)[:, :decoder_input_length],
            'labels': torch.stack(labels)[:, :labels_length],
            'labels_attention_mask': torch.stack(labels_attention_mask)[:, :labels_length],
        }

class FT_Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss
    
def inversion_template(x):
    return '[inversion]' + x + '[/inversion]'

def local_alignment(args, data):
    encoder = InvEncoder(model_path=args.encoder_model_path)
    decoder = InvDecoder(model_path=args.decoder_model_path)
    encoder_hidden_size = get_hidden_size(args.encoder_model_name)
    decoder_hidden_size = get_hidden_size(args.decoder_model_name)
    projector = Projector(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        model_path=args.projector_model_path,
    )
    model = InvEncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        projector=projector,
    )

    t1 = T5Tokenizer.from_pretrained(args.encoder_model_path)
    t2 = AutoTokenizer.from_pretrained(args.decoder_model_path)
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(False)
    
    num_param = 0
    num_trainable_param = 0
    for param in model.parameters():
        num_param += torch.prod(torch.tensor(param.size()))
        if param.requires_grad==True:
            num_trainable_param += torch.prod(torch.tensor(param.size()))
    print(f'param: {num_param} | trainable: {num_trainable_param}')

    input_ids = [inversion_template(x['predict']) for x in data]
    labels = [x['prompt'] for x in data]

    print(f'local alignment train input num: {len(input_ids)}')

    t2.pad_token = t2.eos_token
    res = t1(input_ids, return_tensors='pt', padding=True, truncation=True) # t5-base
    encoder_input_ids = res['input_ids']
    encoder_input_attention_mask = res['attention_mask']

    res = t2(input_ids, return_tensors='pt', padding=True, truncation=True) # llama2
    decoder_input_ids = res['input_ids']
    decoder_input_attention_mask = res['attention_mask']

    res = t2(labels, return_tensors='pt', padding=True, truncation=True) # llama2
    labels = res['input_ids']
    labels = torch.concat([labels, torch.full((labels.shape[0], 1), t2.eos_token_id)], dim=1)
    labels_attention_mask = res['attention_mask']
    labels_attention_mask = torch.concat([labels_attention_mask, torch.full((labels.shape[0], 1), 0)], dim=1)


    assert len(input_ids) == len(labels)

    train_dataset = Dataset.from_dict({
        'encoder_input_ids': encoder_input_ids,
        'encoder_input_attention_mask': encoder_input_attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'decoder_input_attention_mask': decoder_input_attention_mask,
        'labels': labels,
        'labels_attention_mask': labels_attention_mask,
    })
    
    training_args = TrainingArguments(
        output_dir='.cache',
        num_train_epochs=args.fine_tuning_train_epoch,
        learning_rate=args.fine_tuning_learning_rate,
        lr_scheduler_type="constant_with_warmup",
        save_steps=False,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        deepspeed="ds_config.json",
        save_strategy="no",
        report_to="none",
        logging_steps=1,
    )

    trainer = FT_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=FT_DataCollator(),
    )
    trainer.train()
    model.encoder.save(save_path=args.encoder_save_path)
    model.projector.save(save_path=args.projector_save_path)
    t1.save_pretrained(args.encoder_save_path)
    model.to('cpu')

def global_finetuning(args, data):
    encoder = InvEncoder(model_path=args.encoder_model_path)
    decoder = InvDecoder(model_path=args.decoder_model_path)
    encoder_hidden_size = get_hidden_size(args.encoder_model_name)
    decoder_hidden_size = get_hidden_size(args.decoder_model_name)
    projector = Projector(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        model_path=args.projector_model_path,
    )
    model = InvEncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        projector=projector,
    )

    t1 = T5Tokenizer.from_pretrained(args.encoder_model_path)
    t2 = AutoTokenizer.from_pretrained(args.decoder_model_path)
    for param in model.encoder.parameters():
        param.requires_grad_(True)
    for param in model.decoder.parameters():
        param.requires_grad_(False)

    input_ids = [inversion_template(x['predict']) for x in data]
    labels = [x['prompt'] for x in data]

    print(f'global fine-tuning train input num: {len(input_ids)}')

    t2.pad_token = t2.eos_token
    res = t1(input_ids, return_tensors='pt', padding=True, truncation=True) # t5-base
    encoder_input_ids = res['input_ids']
    encoder_input_attention_mask = res['attention_mask']

    res = t2(input_ids, return_tensors='pt', padding=True, truncation=True) # llama2
    decoder_input_ids = res['input_ids']
    decoder_input_attention_mask = res['attention_mask']

    res = t2(labels, return_tensors='pt', padding=True, truncation=True) # llama2
    labels = res['input_ids']
    labels = torch.concat([labels, torch.full((labels.shape[0], 1), t2.eos_token_id)], dim=1)
    labels_attention_mask = res['attention_mask']
    labels_attention_mask = torch.concat([labels_attention_mask, torch.full((labels.shape[0], 1), 0)], dim=1)


    assert len(input_ids) == len(labels)

    train_dataset = Dataset.from_dict({
        'encoder_input_ids': encoder_input_ids,
        'encoder_input_attention_mask': encoder_input_attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'decoder_input_attention_mask': decoder_input_attention_mask,
        'labels': labels,
        'labels_attention_mask': labels_attention_mask,
    })

    training_args = TrainingArguments(
        output_dir='.cache',
        num_train_epochs=args.fine_tuning_train_epoch,
        learning_rate=args.fine_tuning_learning_rate,
        lr_scheduler_type="constant_with_warmup",
        save_steps=False,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        deepspeed="ds_config.json",
        save_strategy="no",
        report_to="none",
        logging_steps=1,
    )

    trainer = FT_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=FT_DataCollator(),
    )
    trainer.train()
    model.encoder.save(save_path=args.encoder_save_path)
    model.projector.save(save_path=args.projector_save_path)
    t1.save_pretrained(args.encoder_save_path)
    model.to('cpu')

def run_FT_train(args):

    with open(args.train_dataset_path, 'r') as f:
        data = json.load(f)
        data = [
            {
                'prompt': x['prompt'],
                'predict': y,
            }
            for x in data
            for y in x['output']
        ]

    n = len(data)
    data1 = data[:n//5]
    data2 = data[n//5:]

    local_alignment(args, data1)
    args.encoder_model_path = args.encoder_save_path
    args.projector_model_path = args.projector_save_path

    global_finetuning(args, data2)