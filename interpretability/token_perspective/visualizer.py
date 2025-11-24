import torch
import numpy as np
class Visualizer:
    def __init__(self, model, tokenizer) -> None:
        """
        Initialize the Visualizer class.

        Parameters:
        - llm (LLMModel): The llm to visualize.

        Attributes:
        - model: The inference pipeline of the provided model.
        - tokenizer (Tokenizer): Tokenizer associated with the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def _map_subwords_to_words(self, sentence: str):
        """
        Convert a sentence into tokens and map subword tokens to their corresponding words.

        Parameters:
        - sentence (str): The input sentence.

        Returns:
        - mapping (list): List mapping subword tokens to word indices.
        - tokens (list): Tokenized version of the input sentence.
        """
        tokens = self.tokenizer.tokenize(sentence)
        mapping = []
        word_idx = 0
        for token in tokens:
            if token.startswith("▁"):
                word_idx += 1
            mapping.append(word_idx)
        return mapping, tokens
    
    def _normalize_importance(self, word_importance):
        """
        Normalize importance values of words in a sentence using min-max scaling.

        Parameters:
        - word_importance (list): List of importance values for each word.

        Returns:
        - list: Normalized importance values for each word.
        """
        min_importance = np.min(word_importance)
        max_importance = np.max(word_importance)
        return (word_importance - min_importance) / (max_importance - min_importance)
    
    def vis_by_grad_embeds(self, input_embeds, label_sentence: str) -> dict:
        self.model.eval()
        
        labels = self.tokenizer(label_sentence, return_tensors='pt').to('cuda')
        
        label_embeds = self.model.get_input_embeddings()(labels['input_ids']).to('cuda')

        model_inputs = {}
        model_inputs['input_embeds'] = torch.concat([input_embeds, label_embeds], dim=1)
        model_inputs['labels'] = torch.concat([torch.full((input_embeds.shape[0], input_embeds.shape[1]), -100).to('cuda'), labels['input_ids']], dim=1)

        embeddings = model_inputs['input_embeds']
        embeddings.requires_grad_()
        embeddings.retain_grad()

        outputs = self.model(
            inputs_embeds=embeddings,
            # attention_mask=model_inputs['attention_mask'],
            labels=model_inputs['labels']
        )
        outputs.loss.backward()

        grads = embeddings.grad
        token_grads = grads[0]
        return token_grads
    