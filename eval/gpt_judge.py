import time
import openai
import tiktoken
from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import os

class Chatbot :
    def __init__(
        self,
        engine: str,
    ) -> None:
        self.engine = engine
        self.last_request_time = 0
        self.request_interval = 1
        self.max_backoff_time = 60

    def get_embedding(self, prompt):
        prompt = prompt.replace('\x00','')
        is_retry = True

        while is_retry:
            elapsed_time = time.monotonic() - self.last_request_time
            if elapsed_time < self.request_interval:
                time.sleep(self.request_interval - elapsed_time)
            self.last_request_time = time.monotonic()
            try:
                response = openai.chat.completions.create(
                    model=self.engine,
                    seed = 0,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                is_retry = False
            except:
                print("Exceed max tries.")
            
            if is_retry == True:
                self.request_interval *= 2
                if self.request_interval > self.max_backoff_time:
                    self.request_interval = self.max_backoff_time
                print(f"Rate limit hit. Sleeping for {self.request_interval} seconds.")
                time.sleep(self.request_interval)

        embedding = response.choices[0].message.content
    
        return embedding
    
    def reset(self) :
        self.last_request_time = 0
        self.request_interval = 1  # seconds
        self.max_backoff_time = 60  # seconds

class ChatbotWrapper :
    def __init__(self, api_key, engine) -> None:
        openai.api_key = api_key
        self.engine = engine
    
    def ask_batch(self, batch_data: list[str], thread_num=1) -> list:
        executor = ThreadPoolExecutor(max_workers=thread_num)
        chatbot_q = Queue(maxsize=thread_num)
        for j in range(thread_num):
            chatbot_q.put(Chatbot(engine=self.engine))
        results = list(tqdm(executor.map(ChatbotWrapper.ask, [chatbot_q for _ in range(len(batch_data))], batch_data), 
                       total=len(batch_data)))
        batch_reponses = []
        for _, res in enumerate(results):
            batch_reponses.append(res)
        return batch_reponses

    @staticmethod
    def ask(chatbot_q: Queue, question:str) -> list:
        if chatbot_q.empty():
            raise Exception("no available chatbot")
        chatbot = chatbot_q.get()
        response = chatbot.get_embedding(question)
        chatbot_q.put(chatbot)
        return response

def gpt_judge(api_key, prediction_str, reference_str):
    model_engine = 'gpt-4o'
    num_threads = 100
    prompt_template = 'Are prompt A and prompt B likely to produce similar outputs? \n Prompt A: ${prompt_a} \n Prompt B: ${prompt_b} \n Please answer YES or NO. Answer:'

    chatbot = ChatbotWrapper(api_key, model_engine)
    prompt_list = [prompt_template.format(prompt_a=pred, prompt_b=ref) for pred, ref in zip(prediction_str, reference_str)]
    all_probs = chatbot.ask_batch(prompt_list, num_threads)
    all_probs = [1 if len(_) >= 3 and _[:3] in ['YES', 'Yes', 'yes'] else 0 for _ in all_probs]
    
    return all_probs
        