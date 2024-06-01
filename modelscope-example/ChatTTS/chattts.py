import os

from modelscope.hub.check_model import check_model_is_id
from modelscope.hub.snapshot_download import snapshot_download

import ChatTTS
from IPython.display import Audio

device = "mps"  # the device to load the model onto

model_id = 'pzc163/chatTTS'
is_id = check_model_is_id(model_id)
model_dir = snapshot_download(model_id)


chat = ChatTTS.Chat()
chat.load_models(source='local', local_path=model_dir)

texts = ["我来啦我来啦我来啦", ]

wavs = chat.infer(texts, use_decoder=True)
Audio(wavs[0], rate=24_000, autoplay=True)