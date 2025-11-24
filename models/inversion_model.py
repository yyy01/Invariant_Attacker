import torch
from torch import nn
from transformers import LlamaModel, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union, Dict, Any

def get_hidden_size(model_name):
    if model_name == 't5-base':
        return 768
    if model_name == 'llama2':
        return 4096
    return NotImplementedError()

class InvDecoder(nn.Module):
    def __init__(
            self,
            model_path,
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def forward(
            self,
            input_embeds,
            attention_mask,
            labels,
    ):
        outputs = self.model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
    
    def embed_input_ids(self, input_ids):
        return self.model.model.embed_tokens(input_ids)
    
class InvEncoder(nn.Module):
    def __init__(
            self,
            model_path,
    ):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)


    def forward(self, input_ids, attention_mask, output_attentions=True):
        ss = input_ids.shape

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = encoder_outputs
        attentions = None
        if output_attentions:
            attentions = encoder_outputs['attentions']
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=attentions,
        )

    def save(self, save_path):
        self.model.save_pretrained(save_path, safe_serialization=False)

class Projector(nn.Module):

    def __init__(
            self,
            encoder_hidden_size=None,
            decoder_hidden_size=None,
            model_path=None,
    ):
        super().__init__()

        projector_type = 'linear'
        
        if projector_type == 'linear':
            self.model = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')
            
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cuda:0')
            self.model.load_state_dict(state_dict)
            return


    def forward(self, x):
        return self.model(x)

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

class InvEncoderDecoder(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            projector,
            tokenizer=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder

    def forward(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
            decoder_input_ids,
            decoder_input_attention_mask,
            labels,
            labels_attention_mask,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
                
        encoder_outputs = self.encoder.forward(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state[0]
        encoder_attentions = encoder_outputs.attentions

        hidden_states = self.projector(hidden_states)

        # soft + y + x
        y_embeds = self.decoder.embed_input_ids(decoder_input_ids)
        hidden_states = torch.concat([hidden_states, y_embeds], dim=1)
        hidden_attention_mask = torch.concat([encoder_input_attention_mask, decoder_input_attention_mask], dim=1)

        labels_input_embeds = self.decoder.embed_input_ids(labels)
        all_input_embeds = torch.concat([hidden_states, labels_input_embeds], dim=1)

        padding_labels = torch.full((encoder_input_ids.shape[0], encoder_input_ids.shape[1]+decoder_input_ids.shape[1]), -100).to('cuda')
        padding_labels = torch.concat([padding_labels, labels], dim=1)
        all_attention_mask = torch.concat([hidden_attention_mask, labels_attention_mask], dim=1)

        decoder_outputs = self.decoder.forward(
            input_embeds=all_input_embeds,
            attention_mask=all_attention_mask,
            labels=padding_labels,
        )

        return decoder_outputs
    

    def generate(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
            decoder_input_ids,
            decoder_input_attention_mask,
            eos_token_id,
            max_new_tokens,
    ):
        
        encoder_outputs = self.encoder.forward(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state[0]
        encoder_attentions = encoder_outputs.attentions

        hidden_states = self.projector(hidden_states)
        
        # soft + y + x
        y_embeds = self.decoder.embed_input_ids(decoder_input_ids)
        hidden_states = torch.concat([hidden_states, y_embeds], dim=1)
        hidden_attention_mask = torch.concat([encoder_input_attention_mask, decoder_input_attention_mask], dim=1)

        all_input_embeds = hidden_states
        all_attention_mask = hidden_attention_mask

        output_token_ids = self.decoder.model.generate(
            inputs_embeds=all_input_embeds,
            attention_mask=all_attention_mask,
            do_sample=False,
            max_new_tokens=128,
        )
        return output_token_ids[0]
    
    def forward_hidden_states(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
    ):
        
        encoder_outputs = self.encoder.forward(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state[0]
        encoder_attentions = encoder_outputs.attentions

        hidden_states = self.projector(hidden_states)

        return hidden_states, encoder_input_attention_mask



class InvSystem(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            projector,
            tokenizer=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder

    def forward(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
            decoder_input_ids,
            decoder_input_attention_mask,
            labels,
            labels_attention_mask,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
                
        encoder_outputs = self.encoder.forward(
            input_ids=encoder_input_ids[0],
            attention_mask=encoder_input_attention_mask[0],
        )

        hidden_states = encoder_outputs.last_hidden_state[0]
        encoder_attentions = encoder_outputs.attentions

        hidden_states = self.projector(hidden_states)


        # soft + y + x
        y_embeds = self.decoder.embed_input_ids(decoder_input_ids[0])

        # group concat
        hidden_states = hidden_states.reshape(1, -1, hidden_states.shape[2])
        encoder_input_attention_mask = encoder_input_attention_mask.reshape(1, -1)
        y_embeds = y_embeds.reshape(1, -1, y_embeds.shape[2])
        decoder_input_attention_mask = decoder_input_attention_mask.reshape(1, -1)

        hidden_states = torch.concat([hidden_states, y_embeds], dim=1)
        hidden_attention_mask = torch.concat([encoder_input_attention_mask, decoder_input_attention_mask], dim=1)

        labels_input_embeds = self.decoder.embed_input_ids(labels)
        all_input_embeds = torch.concat([hidden_states, labels_input_embeds], dim=1)

        padding_labels = torch.full((all_input_embeds.shape[0], hidden_states.shape[1]), -100).to('cuda')
        padding_labels = torch.concat([padding_labels, labels], dim=1)
        all_attention_mask = torch.concat([hidden_attention_mask, labels_attention_mask], dim=1)

        decoder_outputs = self.decoder.forward(
            input_embeds=all_input_embeds,
            attention_mask=all_attention_mask,
            labels=padding_labels,
        )

        return decoder_outputs
    

    def generate(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
            decoder_input_ids,
            decoder_input_attention_mask,
            eos_token_id,
            max_new_tokens,
    ):
        
        
        encoder_outputs = self.encoder.forward(
            input_ids=encoder_input_ids[0],
            attention_mask=encoder_input_attention_mask[0],
        )

        hidden_states = encoder_outputs.last_hidden_state[0]
        encoder_attentions = encoder_outputs.attentions

        hidden_states = self.projector(hidden_states)


        # soft + y + x
        y_embeds = self.decoder.embed_input_ids(decoder_input_ids[0])

        # group concat
        hidden_states = hidden_states.reshape(1, -1, hidden_states.shape[2])
        encoder_input_attention_mask = encoder_input_attention_mask.reshape(1, -1)
        y_embeds = y_embeds.reshape(1, -1, y_embeds.shape[2])
        decoder_input_attention_mask = decoder_input_attention_mask.reshape(1, -1)

        hidden_states = torch.concat([hidden_states, y_embeds], dim=1)
        hidden_attention_mask = torch.concat([encoder_input_attention_mask, decoder_input_attention_mask], dim=1)

        all_input_embeds = hidden_states
        all_attention_mask = hidden_attention_mask
        print('all_input_embeds.shape', all_input_embeds.shape)
        print('all_attention_mask.shape', all_attention_mask.shape)
        print('all_attention_mask', all_attention_mask)

        self.decoder.model.eval()

        del hidden_states
        del hidden_attention_mask
        del encoder_outputs

        next_tokens = torch.tensor([[]], dtype=torch.int32).to('cuda')
        for i in range(max_new_tokens):
            if i>0:
                next_tokens_embeds = self.decoder.embed_input_ids(next_tokens)
                input_embeds = torch.concat([all_input_embeds, next_tokens_embeds], dim=1)
                attention_mask = torch.concat([all_attention_mask, torch.full((1, i), 1).to('cuda')], dim=1)
            else:
                input_embeds = all_input_embeds
                attention_mask = all_attention_mask
            res = self.decoder.model.forward(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )
            logits = res.logits
            next_token_logits = logits[:, -1, :]
            
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) 
            next_tokens = torch.concat([next_tokens, next_token], dim=1)
            if next_token[0].item() == eos_token_id:
                break
            del res
            del logits
            del next_token_logits

        print('next_tokens', next_tokens)
        return next_tokens[0]
        