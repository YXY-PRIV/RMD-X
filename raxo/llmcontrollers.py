import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import re

openai.api_key = ""


def prepare_chatgpt_message(main_prompt):
    messages = [{"role": "user", "content": main_prompt}]
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, temperature=0.7, max_tokens=40, model="gpt-3.5-turbo"):
    if max_tokens > 0:
        response = openai.chat.completions.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature,
            max_tokens=max_tokens)
    else:
        response = openai.chat.completions.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature)

    reply = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    return reply, total_tokens


def trim_question(question):
    question = question.split('Question: ')[-1].replace('\n', ' ').strip()
    if 'Answer:' in question:  # Some models make up an answer after asking. remove it
        q, a = question.split('Answer:')[:2]
        if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
            question = a.strip()
        else:
            question = q.strip()
    return question


class LLMBot:
    def __init__(self, model_tag, max_chat_token=-1):
        self.model_tag = model_tag
        self.model_name = "ChatGPT"
        self.max_chat_token = max_chat_token
        self.total_tokens = 0

    def reset(self):
        self.total_tokens = 0

    def get_used_tokens(self):
        return self.total_tokens

    def get_name(self):
        return self.model_name

    def __call_llm(self, main_prompt, temperature, max_token):
        total_prompt = prepare_chatgpt_message(main_prompt)
        reply, n_tokens = call_chatgpt(total_prompt, temperature=temperature,
                                       model=self.model_tag, max_tokens=max_token)
        return reply, total_prompt, n_tokens

    def infer(self, main_prompt, temperature=0.7):
        reply, _, n_tokens = self.__call_llm(main_prompt, temperature, max_token=self.max_chat_token)
        reply = reply.strip()
        reply = re.sub(r'\b\d+\.\s+', '', reply)
        self.total_tokens += n_tokens
        return reply


class HrchyPrompter:
    def __init__(self, num_sub=5):
        self.num_sub = num_sub

    def query_hyponims(self, category):
        #promt = f"You are an airport X-ray security assistant. List me the {self.num_sub} most prominent types of {category} that plane-travelers might pack in luggage (even if not allowed) for detection in x-ray scans. Return only the names separated by '&'."
        promt = f"You are an airport X-ray security assistant. Based on this category name: {category}. Tell me the most near type of object you would search for in a small suitcase that a person might carry (even if not allowed). Give only the name"
        return promt
    


