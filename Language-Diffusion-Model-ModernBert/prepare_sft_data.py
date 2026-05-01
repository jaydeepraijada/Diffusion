from datasets import load_dataset, load_from_disk
import time
import argparse

from tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="SFT Data Prep (Alpaca)")

parser.add_argument(
    "--test_split_pct", 
    default=0.01, 
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
    "--max_samples",
    type=int,
    default=None,
    help="Truncate dataset to this many samples (useful for local testing)"
)

def prepare_data(args):

    """
    Simple data prep function that will load our datasets, concatenate them together, 
    and then tokenize/group them and save to disk. 
    """

    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    ### Load tokenizer ###
    tokenizer = get_tokenizer(args.hf_model_name)

    ### Load Datasets ###
    split = f"train[:{args.max_samples}]" if args.max_samples is not None else "train"
    dataset = load_dataset("Open-Orca/OpenOrca",
                           split=split,
                           num_proc=args.num_workers,
                           cache_dir=cache_dir)
    
    def apply_chat_template(query, response):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ],
            tokenize=True, #This means we want to return the tokenized version of the prompt, not the string version
            add_special_tokens=True,  
        )
    
    def preprocess(example):
        instruction = example["system_prompt"] + " " + example["question"] if example["system_prompt"] else example["question"]
        output = example["response"]

        tokenized = apply_chat_template(instruction, output)
        return {"input_ids": tokenized, "length": len(tokenized)}

    ### Train/Test Split Dataset ###
    dataset = dataset.train_test_split(test_size=args.test_split_pct, seed=args.dataset_split_seed)

    num_proc = args.num_workers if args.num_workers > 1 else None
    tokenized_data = dataset.map(
        preprocess,
        num_proc=num_proc,
        remove_columns=["id", "system_prompt", "question", "response"]
    )
    
    def keep_within_context(example):
        ## This function will filter out any examples that are longer than the context length of our model, since those will not be useful for training.
        return example["length"] <= context_length
    
    print("Number of Samples In Dataset:", len(tokenized_data["train"]))
    tokenized_data = tokenized_data.filter(keep_within_context, num_proc=num_proc)
    tokenized_data = tokenized_data.remove_columns("length")
    print("Number of Samples After Length Filter:", len(tokenized_data["train"]))

    def get_answer_mask(example):

        """
        This function will return a mask for the answer portion of the prompt,
        which will be used during training to only calculate loss on the answer portion.
        """

        tokenized = example["input_ids"]
     
        query_mask = []
        occurance = 0
        is_answer = False
   
        for t in tokenized:
            check = (t==tokenizer.convert_tokens_to_ids("<END_ID>"))
            if not is_answer:
                query_mask.append(0)
            else:
                query_mask.append(1)

            if check:
                if occurance == 0:
                    occurance += 1
                else:
                    is_answer = True

        example["query_mask"] = query_mask

        return example

    tokenized_data = tokenized_data.map(
        get_answer_mask,
        num_proc=num_proc
    )

    ### Save Data ###
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