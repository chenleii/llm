from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

prompt = "帮我写个helloword程序"
messages = [
    {"role": "system", "content": "你是一名开发领域的专家"},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": prompt}
]
# ['system\nYou are a helpful assistant.\nuser\nGive me a short introduction to large language model.\nassistant\nA large language model is a type of artificial intelligence system that uses machine learning algorithms to generate human-like text based on input from users. These models are designed to']


class OutputIter:
    def __init__(self, iter):
        self.iter = iter
        self.output = ''

    def __iter__(self):
        return self

    def __next__(self):
        next_text = next(self.iter)
        self.output += next_text
        return next_text


#
class Model:
    def __init__(self, device='mps'):
        self.device = device
        # qwen/Qwen1.5-0.5B-Chat
        self.model = AutoModelForCausalLM.from_pretrained(
            "/Users/chen/.cache/modelscope/hub/qwen/Qwen1___5-0___5B-Chat",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/Users/chen/.cache/modelscope/hub/qwen/Qwen1___5-0___5B-Chat"
        )

    def generate(self, messages, timeout=60):
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(self.device)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            timeout=timeout,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=text,
            streamer=streamer,
            max_length=512,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # for new_text in streamer:
        #     yield new_text
        return OutputIter(streamer)

INS = Model()



