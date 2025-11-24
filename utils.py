import os  
import json  

path_mapping = {
    't5-base': "google-t5/t5-base",
    'llama2': "meta-llama/Llama-2-7b-chat-hf",
    'llama3': "meta-llama/Meta-Llama-3-8B-Instruct",
}

def create_json_file(path):  
    if path.endswith('.json'):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(os.path.dirname(path))
        except:
            pass
        return path
    else:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(os.path.dirname(path))
        except:
            pass
        new_path = os.path.join(path, 'result.json')
        return new_path
    