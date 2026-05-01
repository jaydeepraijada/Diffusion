import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import get_tokenizer

def SFTCollator(model_name="answerdotai/ModernBERT-base"):

    tokenizer = get_tokenizer(model_name)
    eos_token = tokenizer.eos_token_id
    end_id_token = tokenizer.convert_tokens_to_ids("<END_ID>")

    def _compute_query_mask(token_ids):
        query_mask = []
        occurance = 0
        is_answer = False
        for t in token_ids:
            query_mask.append(1 if is_answer else 0)
            if t == end_id_token:
                if occurance == 0:
                    occurance += 1
                else:
                    is_answer = True
        return query_mask

    def _collate_fn(batch, max_length=1024):

        raw_ids = [b["input_ids"]["input_ids"] if isinstance(b["input_ids"], dict) else b["input_ids"] for b in batch]
        raw_ids = [ids[:max_length] for ids in raw_ids]
        inputs = [torch.tensor(ids) for ids in raw_ids]
        query_masks = [torch.tensor(_compute_query_mask(ids)) for ids in raw_ids]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=eos_token, batch_first=True)
        query_masks = torch.nn.utils.rnn.pad_sequence(query_masks, padding_value=1, batch_first=True)

        return {"input_ids": inputs, "query_mask": query_masks}

    return _collate_fn

if __name__ == "__main__":

    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    data = load_from_disk("/mnt/datadrive/data/prepped_data/alpaca")["train"]
    loader = DataLoader(data, batch_size=4, collate_fn=SFTCollator())
    next(iter(loader))

