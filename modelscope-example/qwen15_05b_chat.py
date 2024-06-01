from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from modelscope.hub.check_model import check_model_is_id
from modelscope.hub.snapshot_download import snapshot_download

device = "mps"  # the device to load the model onto

model_id = 'qwen/Qwen1.5-0.5B-Chat'
is_id = check_model_is_id(model_id)
model_dir = snapshot_download(model_id)

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "帮我写个helloword程序"
messages = [
    {"role": "system", "content": "你是一名开发领域的专家"},
    {"role": "user", "content": prompt}
]

# 一次生成全部回答
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=16)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

print('----------------分割线------------------------')

# 一个字一个字回答
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors='pt',
).to(device)
streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=30.0, skip_special_tokens=True)
generation_kwargs = dict(
    input_ids=text,
    streamer=streamer,
    max_length=128,
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
for new_text in streamer:
    print(new_text, end='')
