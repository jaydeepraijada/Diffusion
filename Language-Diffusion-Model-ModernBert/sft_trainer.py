import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ModernBertForMaskedLM, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm import tqdm
from tokenizer import get_tokenizer
from safetensors.torch import load_file

from data_utils import SFTCollator

def parse_args():
    ### PARSE COMMAND LINE ARGS ###
    parser = argparse.ArgumentParser(description="")
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
    
    parser.add_argument(
        "--path_to_pretrained_checkpoint",
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
        default=30000,
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
        default=10000,
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate for all Learning Rate Schedulers", 
        default=1e-5, 
        type=float
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay constant for AdamW optimizer", 
        default=0.05, 
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

    args = parser.parse_args()

    return args

args = parse_args()

### Instantiate Accelerate ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)
if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)

### Define Tokenizer ###
tokenizer = get_tokenizer(args.hf_model_name)

### Load Model ###
model = ModernBertForMaskedLM.from_pretrained(args.hf_model_name)
model.resize_token_embeddings(len(tokenizer))
state_dict = load_file(args.path_to_pretrained_checkpoint)
model.load_state_dict(state_dict, strict=False)
model.tie_weights()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

## Define DataLoader ###
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps


tokenized_data = load_from_disk(args.path_to_prepped_data)
train_dataloader = DataLoader(tokenized_data["train"], 
                              batch_size=mini_batchsize,
                              collate_fn=SFTCollator(args.hf_model_name), 
                              shuffle=True)

eval_dataloader = DataLoader(tokenized_data["test"], 
                             batch_size=mini_batchsize,
                             collate_fn=SFTCollator(args.hf_model_name), 
                             shuffle=False)

### Define Optimizer ###
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=args.learning_rate, 
                              weight_decay=args.weight_decay)

### Define Scheduler ###
scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)

### Loss Function ###
loss_func = nn.CrossEntropyLoss(reduction="none")

### Prepare Everything ###
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

### Start Training ###
train = True
completed_steps = 0
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_local_main_process)

while train:

    ### Keep Track of Accumulated Mini-Steps ###
    accumulate_steps = 0
    
    ### Accumulated Loss ###
    accumulate_loss = 0
    
    for batch in train_dataloader:        

        ### Grab Input IDs ###
        input_ids = batch["input_ids"].to(accelerator.device)
        query_mask = batch["query_mask"].to(accelerator.device)

        ### Attend to All Tokens (EVEN EOS) ###
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

        ### Random sample t to mask each token with that probability ###'
        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
        mask = torch.bernoulli(t)

        ### Mask only valid where it is not query ###
        mask = mask * query_mask
        mask = mask.bool()

        ### Mask Data and Dont Compute Loss for Unmasked Data ###
        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        ### Compute Logits ###
        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
  
        ### Compute Loss (per token) ###
        num_classes = logits.shape[-1]
        loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                         labels.flatten())
        
        
        ### Scale loss by t. As t gets larger, we have more mask tokens and its becomes a tougher problem ###
        ### So naturally samples with large t, will have a worse loss. Just to make it fair we scale our ###
        ### loss per sample by the t ###
        loss = loss.reshape(batch_size, seq_len) / t

        ### Different answers are of different lengths, so lets scale by that too ###
        answer_lengths = query_mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        answer_lengths = answer_lengths.clamp_min(1)  
        loss = loss / answer_lengths

        ### Add up all the per-token losses and average across batch ###
        loss = loss.sum(dim=1).mean()

        ### Scale Loss by Gradient Accumulation Steps ###
        loss = loss / args.gradient_accumulation_steps
        accumulate_loss += loss

        ### Compute Gradients ###
        accelerator.backward(loss)

        accumulate_steps += 1

        if accumulate_steps % args.gradient_accumulation_steps == 0:

            ### Update Model ###
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ### Update Scheduler ###
            scheduler.step()

            ### Log Results!! ###
            if completed_steps % args.logging_steps == 0:
                
                accumulate_loss = accumulate_loss.detach()
            
                if accelerator.state.num_processes > 1:

                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))

                log = {"train_loss": accumulate_loss,
                       "learning_rate": scheduler.get_last_lr()[0]}

                logging_string = f"[{completed_steps}/{args.num_training_steps}] Training Loss: {accumulate_loss}"
                
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                if args.log_wandb:
                    accelerator.log(log, step=completed_steps) 

            ### Evaluation Loop ###
            if completed_steps % args.evaluation_interval == 0:
                if accelerator.is_main_process:
                    progress_bar.write("Evaluating Model!!")
                
                model.eval()

                ### Dictionary to Store Results ###
                log = {"val_loss": 0}

                ### Iterate Data ###
                num_losses = 0
                for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    
                    input_ids = batch["input_ids"].to(accelerator.device)
                    query_mask = batch["query_mask"].to(accelerator.device)

                    batch_size, seq_len = input_ids.shape
                    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

                    batch_size, seq_len = input_ids.shape
                    t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len)
                    mask = torch.bernoulli(t).bool()

                    mask = mask * query_mask
                    mask = mask.bool()

                    masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
                    labels = input_ids.masked_fill(~mask, -100)

                    with torch.inference_mode():
                        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
                    
                    num_classes = logits.shape[-1]
                    loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                                     labels.flatten())
                    
                    loss = loss.reshape(batch_size, seq_len) / t
                    answer_lengths = query_mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
                    answer_lengths = answer_lengths.clamp_min(1)  
                    loss = loss / answer_lengths
                    loss = loss.sum(dim=1).mean()

                    loss = loss.detach()
                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))

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

path_to_checkpoint = os.path.join(path_to_experiment, f"final_model")
accelerator.save_state(output_dir=path_to_checkpoint)
accelerator.end_training()