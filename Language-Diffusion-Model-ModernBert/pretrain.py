import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm import tqdm
from tokenizer import get_tokenizer
from huggingface_hub import HfApi


def parse_args():
    ### PARSE COMMAND LINE ARGS ###
    parser = argparse.ArgumentParser(description="RoBERTa Pretraining Arguments on Wikipedia + BookCorpus")
    parser.add_argument(
        "--experiment_name", 
        required=True, 
        type=str
    )

    parser.add_argument(
        "--working_directory", 
        required=True, 
        type=str
    )
    
    ##########################
    ### HUGGINGFACE CONFIG ###
    ##########################

    parser.add_argument(
        "--hf_model_name",
        help="Huggingface model name we want to use for the tokenizer",
        default="answerdotai/ModernBERT-base",
        type=str
    )

    #########################
    ### DATASET ARGUMENTS ###
    #########################

    parser.add_argument(
        "--path_to_prepped_data",
        required=True,
        help="Path to data prepared in `prepare_data.py`",
        type=str
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloading",
        default=24, 
        type=int
    )

    ##############################
    ### TRAINING CONFIGURATION ###
    ##############################

    parser.add_argument(
        "--per_gpu_batch_size",
        help="Overall batch size per gpu during training",
        default=16, 
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Splits per_gpu_batch_size by gradient_accumulation_steps",
        default=1, 
        type=int
    )

    parser.add_argument(
        "--num_training_steps", 
        help="Number of training steps to take",
        default=100000,
        type=int
    )

    parser.add_argument(
        "--max_grad_norm",
        help="Max gradient norm used for stabilizing training with gradient clipping",
        default=1.0, 
        type=float
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_steps", 
        help="Number of iterations for every log of metrics to wandb",
        default=1,
        type=int
    )

    parser.add_argument(
        "--evaluation_interval", 
        help="Number of iterations for every evaluation and plotting",
        default=2500, 
        type=int
    )

    parser.add_argument(
        "--checkpoint_interval",
        help="Number of iterations for checkpointing",
        default=2500,
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate for all Learning Rate Schedulers", 
        default=5e-5, 
        type=float
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay constant for AdamW optimizer", 
        default=0.01, 
        type=float
    )

    #############################
    ### LOGGING CONFIGURATION ###
    #############################

    parser.add_argument(
        "--log_wandb",
        help="Flag to enable logging to wandb",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    ############################
    ### HUGGINGFACE HUB PUSH ###
    ############################

    parser.add_argument(
        "--hf_push_repo",
        default=None,
        help="HF Hub repo id (e.g. username/my-model) to push checkpoints to. "
             "Set HF_TOKEN env var or pass --hf_token.",
        type=str
    )

    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN env var if not provided.",
        type=str
    )

    args = parser.parse_args()

    return args



args = parse_args()

path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)
if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)

hf_api = None
if args.hf_push_repo:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    hf_api = HfApi(token=hf_token)
    if accelerator.is_main_process:
        hf_api.create_repo(args.hf_push_repo, repo_type="model", exist_ok=True)
        accelerator.print(f"HF push enabled → {args.hf_push_repo}")


### Get Tokenizer ###
tokenizer = get_tokenizer(args.hf_model_name)

### Load model ### 
model = AutoModelForMaskedLM.from_pretrained(args.hf_model_name)
model.resize_token_embeddings(len(tokenizer)) #add new special tokens to model

model_parameters = filter(lambda p: p.requires_grad, model.parameters()) #only count parameters that require gradients
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

mini_batch_size = args.per_gpu_batch_size // args.gradient_accumulation_steps #this is for gradient accumulation

def collate_fn(batch):
    """
    Collate function for batching tokens. We do this because our data is already tokenized and
    we just need to stack the input ids together. We also convert them to tensors here.
    """
    tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
    return {"input_ids": tokens}  


tokenized_data = load_from_disk(args.path_to_prepped_data)
train_dataloader = DataLoader(tokenized_data["train"], 
                              batch_size=mini_batch_size,
                              collate_fn=collate_fn,
                              shuffle= True)

eval_dataloader = DataLoader(tokenized_data["test"],
                             batch_size = mini_batch_size,
                             collate_fn = collate_fn,
                             shuffle = False)

### Optimizer ###
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=args.learning_rate,   
                              weight_decay=args.weight_decay)

scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
    num_training_steps=args.num_training_steps * accelerator.num_processes,
)

loss_func = nn.CrossEntropyLoss(reduction="none") #reduction none?

model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
) #prepare everything for accelerator (handles multi gpu, mixed precision, etc)

train = True
completed_steps = 0
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_local_main_process) #disable progress bar for non main processes

while train:

    accumulate_steps = 0
    accumulate_loss = 0

    for batch in train_dataloader:

        #### Grab Input IDs ####
        input_ids = batch["input_ids"].to(accelerator.device)

        ### Attend to All Tokens (EVEN PADDING) ###
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device) 
        #we attend to all tokens including padding because we want the model to learn to predict padding tokens as well

        ### Random sample t to mask tokens (we want to mask a random fraction of tokens) ###
        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5) #sample t for each sample in the batch and expand to seq_len. Clamp min to avoid division by zero later on
        # ... (rest of the masking logic) ...
        mask = torch.bernoulli(t).bool() #mask is a boolean tensor where True means we will mask that token #Bernoulli samples binary values based on the probabilities in t
         

        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id) #mask input ids where mask is True
        labels = input_ids.masked_fill(~mask, -100) #labels are the original input ids where mask is False and -100 where mask is True (we only compute loss on masked tokens)

        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"] #get logits from model  
        #(B, seq_len, vocab_size)

        num_classes = logits.shape[-1] 
        loss = loss_func(logits.reshape(batch_size*seq_len, num_classes), #reshape logits and labels for loss computation
                         labels.flatten())
        
        loss = loss.reshape(batch_size, seq_len) / t #scale loss by t (as t gets larger we have more masked tokens and its a harder problem so we scale the loss to make it fair)
        loss = loss.mean() #average loss over batch and sequence length


        loss = loss / args.gradient_accumulation_steps #scale loss by gradient accumulation steps
        accumulate_loss += loss 

        accelerator.backward(loss) #backpropagate loss 

        accumulate_steps += 1

        if accumulate_steps % args.gradient_accumulation_steps == 0:

            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm) #clip gradients for stability
            optimizer.step() #update model parameters
            optimizer.zero_grad(set_to_none=True) #zero out gradients for next step
            scheduler.step() #update learning rate

            if completed_steps % args.logging_steps == 0:

                accumulate_loss = accumulate_loss.detach() #detach accumulated loss from computation graph for logging

                if accelerator.state.num_processes > 1:
                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss)) #gather loss from all processes and average for logging

                log = {"train_loss": accumulate_loss,
                       "learning_rate": scheduler.get_last_lr()[0]} #log dictionary for wandb   
                
                logging_string = f"[{completed_steps}/{args.num_training_steps}] Training Loss: {accumulate_loss}" #string to print to console

                if accelerator.is_main_process:
                    progress_bar.write(logging_string) #print to console from main process

                if args.log_wandb:
                    accelerator.log(log, step=completed_steps)

             ## Evaluation Loop ###
            if completed_steps % args.evaluation_interval == 0:
                if accelerator.is_main_process:
                    progress_bar.write("Evaluating Model!!")
                
                model.eval()

                ### Dictionary to Store Results ###
                log = {"val_loss": 0}

                ### Iterate Data ###
                num_losses = 0
                for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    
                    ### Grab Input IDs ###
                    input_ids = batch["input_ids"].to(accelerator.device)

                    ### Attend to All Tokens (EVEN EOS) ###
                    batch_size, seq_len = input_ids.shape
                    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

                    ### Random sample t to mask each token with that probability ###'
                    batch_size, seq_len = input_ids.shape
                    t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len)
                    mask = torch.bernoulli(t).bool()

                    ### Mask Data and Dont Compute Loss for Unmasked Data ###
                    masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
                    labels = input_ids.masked_fill(~mask, -100)

                    ### Compute Logits ###
                    with torch.inference_mode():
                        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
                    
                    ### Compute Loss (per token) ###
                    num_classes = logits.shape[-1]
                    loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                                     labels.flatten())
                    
                    ### Scale loss by t. As t gets larger, we have more mask tokens and its becomes a tougher problem ###
                    ### So naturally samples with large t, will have a worse loss. Just to make it fair we scale our ###
                    ### loss per sample by the t ###
                    loss = loss.reshape(batch_size, seq_len) / t
                    loss = loss.mean()

                    ### Grab Loss ###
                    loss = loss.detach()
                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))

                    ### Add to our Logs ###
                    log["val_loss"] += loss
                    num_losses += 1
                
                ### Divide Log by Num Losses ###
                log["val_loss"] = log["val_loss"] / num_losses

                ## Print to Console ###
                logging_string = f"[{completed_steps}/{args.num_training_steps}] Validation Loss: {log['val_loss']}"
        
                ### Print out Log ###
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                if args.log_wandb:
                    accelerator.log(log, step=completed_steps)

                model.train()
            
            ### Checkpoint Model (Only need main process for this) ###
            if (completed_steps % args.checkpoint_interval == 0):
                
                ### Save Checkpoint ###
                path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

                if accelerator.is_main_process:
                    progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

                ### Make sure that all processes have caught up before saving checkpoint! ###
                accelerator.wait_for_everyone()

                ### Save checkpoint using only the main process ###
                if accelerator.is_main_process:
                    accelerator.save_state(output_dir=path_to_checkpoint)

                ### Push checkpoint to HF Hub ###
                if accelerator.is_main_process and hf_api is not None:
                    progress_bar.write(f"Pushing checkpoint_{completed_steps} to {args.hf_push_repo} ...")
                    hf_api.upload_folder(
                        folder_path=path_to_checkpoint,
                        repo_id=args.hf_push_repo,
                        path_in_repo=f"checkpoint_{completed_steps}",
                        commit_message=f"Checkpoint at step {completed_steps}",
                    )
                    progress_bar.write("Push complete.")
            
            if completed_steps >= args.num_training_steps:
                train = False
                if accelerator.is_main_process:
                    progress_bar.write("Completed Training!!")
                break

            ### Iterate Progress Bar and Completed Steps ###
            completed_steps += 1
            progress_bar.update(1)

            ### Reset Loss Accumulate For Next Accumulation ###
            accumulate_loss = 0

path_to_final = os.path.join(path_to_experiment, "final_model")
accelerator.save_state(output_dir=path_to_final)

if accelerator.is_main_process and hf_api is not None:
    accelerator.print(f"Pushing final model to {args.hf_push_repo} ...")
    hf_api.upload_folder(
        folder_path=path_to_final,
        repo_id=args.hf_push_repo,
        path_in_repo="final_model",
        commit_message="Final model",
    )
    accelerator.print("Final model pushed.")

accelerator.end_training()

            
























