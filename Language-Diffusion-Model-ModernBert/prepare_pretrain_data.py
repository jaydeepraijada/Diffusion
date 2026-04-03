from datasets import load_dataset, load_from_disk, concatenate_datasets
import time
import argparse

from tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Finemath Data Prep")

parser.add_argument(
    "--test_split_pct", 
    default=0.005, 
    help="What percent of data do you want to use for Train/Test split",
    type=float
)

parser.add_argument(
    "--context_length", 
    default=1024, 
    help="Pass in argument to override the default in Config, but then make sure config \
        reflects this when training",
    type=int
)

parser.add_argument(
    "--path_to_data_store", 
    required=True, 
    help="Path to where you want to save the final tokenized dataset",
    type=str
)

parser.add_argument(
    "--huggingface_cache_dir",
    default=None,
    help="path to huggingface cache directory if different from default",
    type=str
)

parser.add_argument(
    "--dataset_split_seed",
    default=42, 
    help="Seed to ensure reproducible split of dataset",
    type=int
)

parser.add_argument(
    "--num_workers",
    default=16, 
    help="Number of workers you want to use to process dataset",
    type=int
)

parser.add_argument(
    "--hf_model_name",
    default="answerdotai/ModernBERT-base",
    help="Name of model so we can use the huggingface tokenizer", 
    type=str
)

parser.add_argument(
    "--large_dataset",
    action="store_true"
)

parser.add_argument(
    "--batch_size",
    type=int, 
    default=1000
)

def prepare_data(args):

    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    ### Load tokenizer ###
    tokenizer = get_tokenizer(args.hf_model_name)

    ### Load Datasets ###
    if args.large_dataset:
        fw = load_dataset("HuggingFaceFW/fineweb", 
                        name="sample-10BT", 
                        split="train", 
                        cache_dir=cache_dir,
                        num_proc=args.num_workers)
        
        fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", 
                            name="sample-10BT", 
                            split="train", 
                            cache_dir=cache_dir,
                            num_proc=args.num_workers)
        
        wiki = load_dataset("wikimedia/wikipedia", 
                            "20231101.en",
                            split="train",
                            cache_dir=cache_dir,
                            num_proc=args.num_workers)
        
        ### Remove All Columns that are Not Text ###
        fw = fw.remove_columns([col for col in fw.column_names if col != "text"])
        fw_edu = fw_edu.remove_columns([col for col in fw_edu.column_names if col != "text"])
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

        ### Concatenate Datasets Together ###
        dataset = concatenate_datasets([fw, fw_edu, wiki])

    else:

        dataset = load_dataset("manu/project_gutenberg", 
                               split="en",
                               cache_dir=cache_dir, 
                               num_proc=args.num_workers)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    ### Train/Test Split Dataset ###
    dataset = dataset.train_test_split(test_size=args.test_split_pct, seed=args.dataset_split_seed)

    ### Tokenize Dataset ###
    def compute_tokens(examples):
        
        tokenized = tokenizer(examples["text"], 
                              return_attention_mask=False, 
                              add_special_tokens=True,
                              max_length=None,
                              truncation=False)

        ### Chunk Text ###
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            for i in range(0, len(ids), context_length):
                chunk = ids[i:i+context_length]
                if len(chunk) < context_length:
                    chunk = chunk + [tokenizer.pad_token_id] * (context_length - len(chunk))
                input_ids_list.append(chunk)
        
        return {"input_ids": input_ids_list}
    
    tokenized_data = dataset.map(
        compute_tokens, 
        batched=True, 
        batch_size=args.batch_size,
        num_proc=args.num_workers, 
        remove_columns="text"
    )

    ### Save Data ###
    print("Saving to:", path_to_save)
    tokenized_data.save_to_disk(path_to_save)


if __name__ == "__main__":

    args = parser.parse_args()
    prepare_data(args)

    ### Test that it worked ###
    start = time.time()
    data = load_from_disk(args.path_to_data_store)
    end = time.time()

    print("Time to Load Dataset", end-start)
    print(data)