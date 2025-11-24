import json
import nltk
import numpy as np
import tqdm

import evaluate
from .cs_score import cs_score
from .gpt_judge import gpt_judge

class INV2A_Metrics:
    def __init__(self):
        self.metric_bleu = evaluate.load('sacrebleu')
        self.exact_match = evaluate.load('exact_match')

    def bleu(self, prediction_str, reference_str):
        bleu_res = [
            self.metric_bleu.compute(
                predictions=[p],
                references=[r]
            )['score']
            for p, r in zip(prediction_str, reference_str)
        ]
        return bleu_res
    
    def f1(self, prediction_str, reference_str):
        # Token F1
        assert len(prediction_str) == len(reference_str)
        
        num_samples = len(prediction_str)
        f1_res = []
        for i in range(num_samples):
            true_words = nltk.tokenize.word_tokenize(reference_str[i])
            pred_words = nltk.tokenize.word_tokenize(prediction_str[i])

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)

            tp = len(true_words_set & pred_words_set)
            fp = len(true_words_set) - tp
            fn = len(pred_words_set) - tp

            precision = tp / (tp + fp + 1e-20)
            recall = tp / (tp + fn + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            
            f1_res.append(f1)
        return f1_res
    
    def em(self, prediction_str, reference_str):
        em_res = [
            self.exact_match.compute(
                predictions=[p],
                references=[r]
            )['exact_match']
            for p, r in zip(prediction_str, reference_str)
        ]
        return em_res
    
    def cs(self, args, prediction_str, reference_str):
        if args.api_key is None:
            return None
        return cs_score(args.api_key, prediction_str, reference_str)
    
    def gpt(self, args, prediction_str, reference_str):
        if args.api_key is None:
            return None
        return gpt_judge(args.api_key, prediction_str, reference_str)
    
    def calc_all_scores(self, args, prediction_str, reference_str):
        bleu_scores = self.bleu(prediction_str, reference_str)
        f1_scores = self.f1(prediction_str, reference_str)
        em_scores = self.em(prediction_str, reference_str)
        cs_scores = self.cs(args, prediction_str, reference_str)
        gpt_scores = self.gpt(args, prediction_str, reference_str)
        res = {
            'bleu': round(np.average(bleu_scores).item(), 2) if bleu_scores is not None else None,
            'f1': round(np.average(f1_scores).item()*100, 2) if f1_scores is not None else None,
            'em': round(np.average(em_scores).item()*100, 2) if em_scores is not None else None,
            'cs': round(np.average(cs_scores).item()*100, 2) if cs_scores is not None else None,
            'gpt': round(np.average(gpt_scores).item()*100, 2) if gpt_scores is not None else None,
        }

        return res