import sys
sys.path.append('../..')

import json
import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer
from models.inversion_model import InvEncoder, InvDecoder, InvEncoderDecoder, Projector, get_hidden_size
from sklearn.feature_selection import mutual_info_regression  

device = 'cuda'
dataset_file_path = '../data/mixed_test.json'
encoder_model_path = '' # Fill with the path of inversion model encoder
decoder_model_path = '' # Fill with the path of inversion model decoder
projector_model_path = '' # Fill with the path of inversion model projector
encoder_model_name = 't5-base'
decoder_model_name = 'llama2'
encoder = InvEncoder(model_path=encoder_model_path)
decoder = InvDecoder(model_path=decoder_model_path)
encoder_hidden_size = get_hidden_size(encoder_model_name)
decoder_hidden_size = get_hidden_size(decoder_model_name)
projector = Projector(
    encoder_hidden_size=encoder_hidden_size,
    decoder_hidden_size=decoder_hidden_size,
    model_path=projector_model_path,
)
inv_model = InvEncoderDecoder(
    encoder=encoder,
    decoder=decoder,
    projector=projector,
)
inv_model.to('cuda')

def get_hidden_states(y):
    t1 = AutoTokenizer.from_pretrained(encoder_model_path)

    res = t1(y, return_tensors='pt', padding=True, truncation=True).to(device)
    encoder_input_ids = res['input_ids']
    encoder_input_attention_mask = res['attention_mask']

    hidden_states, hidden_attention_mask = inv_model.forward_hidden_states(
        encoder_input_ids=encoder_input_ids,
        encoder_input_attention_mask=encoder_input_attention_mask,
    )
    encoder_embeds = hidden_states

    t2 = AutoTokenizer.from_pretrained(decoder_model_path)
    res = t2(y, return_tensors='pt').to(device)
    decoder_input_ids = res['input_ids']
    decoder_attention_mask = res['attention_mask']
    text_embeds = decoder.embed_input_ids(decoder_input_ids)

    merge_embeds = torch.concat([encoder_embeds, text_embeds], dim=1).to('cuda')
    merge_attention = torch.concat([hidden_attention_mask, decoder_attention_mask], dim=1).to('cuda')

    outputs = decoder.model(inputs_embeds=merge_embeds, attention_mask=merge_attention, output_hidden_states=True, output_attentions=True)

    layer_hidden_states = []
    last_token_embeds = encoder_embeds[:, -1, :]
    for i in range(len(outputs.hidden_states)):
        hidden_states = outputs.hidden_states[i][:, -1, :]
        layer_hidden_states.append(hidden_states)
    res = {
        'last_token_embeds': last_token_embeds.cpu().tolist(),
        'layer_hidden_states': [_.cpu().tolist() for _ in layer_hidden_states],
    }
    
    return res

with open(dataset_file_path, 'r') as f:
    data = json.load(f)

predicts = [y for x in data for y in x['output']]

states_res = []
for y in tqdm.tqdm(predicts, total=len(predicts)):
    states_res.append(get_hidden_states(y))
    torch.cuda.empty_cache()


layer_num = 33

layer_indices = []
for layer in range(layer_num):
    layer_hidden_states = [torch.tensor(x['layer_hidden_states'][layer]) for x in states_res]
    avg_layer_hidden_states = sum(layer_hidden_states) / len(layer_hidden_states)
    flattened_tensor = avg_layer_hidden_states.flatten()

    sorted_indices = torch.argsort(flattened_tensor).tolist()
    min_indices = sorted_indices[:6]
    max_indices = sorted_indices[-6:]
    middle_index = len(flattened_tensor) // 2 
    mid_indices = sorted_indices[middle_index-3: middle_index+3]

    layer_indices.append({
        'min_indices': min_indices,
        'max_indices': max_indices,
        'mid_indices': mid_indices,
    })

res = {}
for layer in range(layer_num):
    pick_indices = layer_indices[layer]['min_indices'] + layer_indices[layer]['max_indices'] + layer_indices[layer]['mid_indices']
    for k in pick_indices:
        X = [torch.tensor(t['last_token_embeds'][0]) for t in states_res]
        y = [t['layer_hidden_states'][layer][0][k] for t in states_res]
        mi = mutual_info_regression(X, y)
        print(mi.shape)
        print(f'layer {layer} dim {k}: {mi} avg {np.average(mi)}')
        res[f'layer-{layer:02}-dim-{k:04}'] = mi.tolist()
        with open('interpretability_layer_perspective_scores.json', 'w') as f:
            json.dump(res, f)
