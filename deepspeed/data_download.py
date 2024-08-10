from modelscope.hub.check_model import check_model_is_id
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset

model_id = 'qwen/Qwen1.5-0.5B-Chat'
is_id = check_model_is_id(model_id)
model_dir = snapshot_download(model_id)

dataset_dir =  MsDataset.load('YorickHe/CoT_zh', subset_name='default', split='train')


print(f'模型文件路径：${model_dir}')
print(f'数据集路径：${dataset_dir}')