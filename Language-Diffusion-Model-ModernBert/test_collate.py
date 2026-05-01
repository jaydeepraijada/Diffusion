from data_utils import SFTCollator
from datasets import load_from_disk
from torch.utils.data import DataLoader

data = load_from_disk('/workspace/data/sft')['train']
loader = DataLoader(data, batch_size=2, collate_fn=SFTCollator())
batch = next(iter(loader))
print('input_ids shape:', batch['input_ids'].shape)
print('query_mask shape:', batch['query_mask'].shape)
print('query_mask sample:', batch['query_mask'][0][:50])
print('num answer tokens:', batch['query_mask'][0].sum().item())
