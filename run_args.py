import argparse

class INV2A_Train_Parser:  
    def __init__(self):  
        self.parser = argparse.ArgumentParser(description="INV2A_Train_Parser")  

        self.parser.add_argument('--train_dataset_path', type=str)

        self.parser.add_argument('--encoder_model_name', type=str, choices=['t5-base'], default='t5-base')
        self.parser.add_argument('--encoder_model_path', type=str)
        self.parser.add_argument('--decoder_model_name', type=str, choices=['llama2', 'llama3'], default='llama2')
        self.parser.add_argument('--decoder_model_path',type=str)
        self.parser.add_argument('--projector_model_path', type=str)

        self.parser.add_argument('--do_contrastive_learning', type=str, choices=['True', 'False'])
        self.parser.add_argument('--contrastive_learning_train_epoch', type=int)
        self.parser.add_argument('--contrastive_learning_learning_rate', type=float)

        self.parser.add_argument('--fine_tuning_train_epoch', type=int)
        self.parser.add_argument('--fine_tuning_learning_rate', type=float)
        
        self.parser.add_argument('--task', type=str, choices=['user', 'system'], default='user')

        self.parser.add_argument('--model_save_path', type=str, default='savings')
        self.parser.add_argument('--seed', type=int, default=42)

    def parse(self):
        args, unknown = self.parser.parse_known_args()
        args.do_contrastive_learning = args.do_contrastive_learning == 'True'
        return args

class INV2A_Inference_Parser:  
    def __init__(self):  
        self.parser = argparse.ArgumentParser(description="INV2A_Inference_Parser")  
        
        self.parser.add_argument('--inference_dataset_path', type=str)

        self.parser.add_argument('--encoder_model_name', type=str, choices=['t5-base'], default='t5-base')
        self.parser.add_argument('--encoder_model_path', type=str)
        self.parser.add_argument('--decoder_model_name', type=str, choices=['llama2', 'llama3'], default='llama2')
        self.parser.add_argument('--decoder_model_path',type=str)
        self.parser.add_argument('--projector_model_path', type=str)

        self.parser.add_argument('--max_inversion_tokens', type=int)
        self.parser.add_argument('--api_key', type=str)

        self.parser.add_argument('--result_save_path', type=str)

    def parse(self):
        args, unknown = self.parser.parse_known_args()
        return args

import experiments

class INV2A_Experiments_Parser:  
    def __init__(self):  
        self.parser = argparse.ArgumentParser(description="INV2A_Refine_Parser")  

        self.subparsers = self.parser.add_subparsers(dest='command')

        self.parser1 = self.subparsers.add_parser('main_result', help='Main result scores.')
        self.parser1.set_defaults(func=experiments.main_result)
        self.parser1.add_argument('--api_key', type=str)

        self.parser2 = self.subparsers.add_parser('ablation_without_CL', help='Train model without contrastive learning.')
        self.parser2.set_defaults(func=experiments.ablation_without_CL)
        self.parser2.add_argument('--train_dataset_path', type=str)
        self.parser2.add_argument('--inference_dataset_path', type=str)
        self.parser2.add_argument('--encoder_model_name', type=str)
        self.parser2.add_argument('--encoder_model_path', type=str)
        self.parser2.add_argument('--decoder_model_name', type=str)
        self.parser2.add_argument('--decoder_model_path', type=str)
        self.parser2.add_argument('--model_save_path', type=str)
        self.parser2.add_argument('--seed', type=int, default=42)
        self.parser2.add_argument('--api_key', type=str)
        self.parser2.add_argument('--result_save_path', type=str)

        self.parser3 = self.subparsers.add_parser('ablation_with_refine', help='Inference with refine.')
        self.parser3.set_defaults(func=experiments.ablation_with_refine)
        self.parser3.add_argument('--inference_dataset_path', type=str)
        self.parser3.add_argument('--encoder_model_name', type=str)
        self.parser3.add_argument('--encoder_model_path', type=str)
        self.parser3.add_argument('--decoder_model_name', type=str)
        self.parser3.add_argument('--decoder_model_path', type=str)
        self.parser3.add_argument('--projector_model_path', type=str)
        self.parser3.add_argument('--api_key', type=str)
        self.parser3.add_argument('--result_save_path', type=str)

    def parse(self):
        args, unknown = self.parser.parse_known_args()
        return args
