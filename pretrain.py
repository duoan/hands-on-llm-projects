"""
This module demostrates the pretrain a launage model with transformers.
- dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- model: roneneldan/TinyStories-1M
- tokenizer: roneneldan/TinyStories-1M
- reference: https://arxiv.org/pdf/2305.07759

You are free to use any dataset and pre-trained model

# https://huggingface.co/datasets/roneneldan/TinyStories
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import torch
import torch.distributed as dist
from datetime import datetime, timezone
from datasets import load_dataset, DatasetDict
import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from transformers import TrainerCallback
import torch


class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="Once upon a time", max_length=100, log_to_wandb=True):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.log_to_wandb = log_to_wandb

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Generate text when evaluation happens"""
        if model is None:
            return
        
        model.eval()  # Switch to eval mode
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(**inputs, max_length=self.max_length, pad_token_id=self.tokenizer.eos_token_id)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Step {state.global_step}] Generated Text: {generated_text}\n")

        # Log to Weights & Biases if enabled
        if self.log_to_wandb:
            import wandb
            wandb.log({"Generated Text": generated_text}, step=state.global_step)

        model.train()  # Switch back to training mode


class GradientMonitorCallback(TrainerCallback):
    """
    A callback for monitoring parameter gradient norms during training in Hugging Face Transformers.
    Logs gradient norms to Weights & Biases (wandb).
    """
    def __init__(self, log_interval=10):
        """
        Args:
            log_interval (int): How often to log gradient norms (in steps).
        """
        self.log_interval = log_interval
    
    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        """
        Logs gradient norms at the end of each step.
        """
        if state.global_step % (self.log_interval * args.gradient_accumulation_steps) == 0 and model:
            grad_norms = {}
            param_dists = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
                    if name.endswith('weight'):
                        # traing parameter weights
                        param_dists[f"param_dist/{name}"] = wandb.Histogram(param.detach().cpu().float().numpy())
            # Log only if there are gradients
            if grad_norms:
                wandb.log(grad_norms, step=state.global_step)
            if param_dists:
                 wandb.log(param_dists, step=state.global_step)

def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_path = "roneneldan/TinyStories"
    
    # https://huggingface.co/docs/transformers/en/model_doc/qwen2
    model_path = "Qwen/Qwen2.5-0.5B"
    tokenizer_path = "roneneldan/TinyStories-1M"
    output_path = f"results/{model_path}/pt/{run_id}"
    tokenized_datapath = os.path.join("data", dataset_path, "tokenized_dataset")

    os.makedirs(output_path, exist_ok=True)
    
    # Use the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.vocab_size = tokenizer.vocab_size
    config.hidden_size = 768
    config.intermediate_size = 768 * 4
    config.num_hidden_layers = 2
    config.num_attention_heads = 8
    config.num_key_value_heads = 2  # use Group Query Attention (GQA) 
    config.use_cache = False
    config.attention_dropout = 0.1
    
    print(config)
    
    # Initialize a new model from config
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one
        attn_implementation="flash_attention_2", # eager, flash_attention_2
        trust_remote_code=True,
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params:,}")  # Format with commas
    # Count total parameters (trainable + frozen)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (Trainable + Frozen): {total_params:,}")
    

    training_args = TrainingArguments(
        run_name=run_id,
        output_dir=output_path,
        overwrite_output_dir=True,
        # fused kernel optimization
        optim='adamw_torch_fused',
        learning_rate=5e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts",
        num_train_epochs=5, # More epochs for training from scratch
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        save_steps=0.2,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=True,
        weight_decay=0.01,  # Increased for better regularization
        max_grad_norm=1.0,  # Increased for training from scratch
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=True,
        # additional metrics
        include_num_input_tokens_seen=True,
    )

    # Ensure W&B is only initialized on rank 0
    if dist.is_initialized():
        local_rank = dist.get_rank()
    else:
        local_rank = 0  # Assume single-GPU or non-distributed mode

    # Initialize W&B only on the primary process
    if local_rank == 0:
        wandb.login()
        wandb.init(
            project=f"hands-on-llm-pretrain-{model_path.replace("/", "-")}",
            id=run_id,
            config=training_args.to_dict(),
        )
    else:
        os.environ["WANDB_MODE"] = "offline"

    # process dataset
    def map_callback(examples):
        sequence_length = 512
        eos_token = tokenizer.eos_token
        text_examples = [text + eos_token for text in examples["text"]]
        tokenized_examples = tokenizer(
            text_examples,
            truncation=True,
            max_length=sequence_length,
            add_special_tokens=True,
            return_attention_mask=True,
            # return_tensors="pt",
        )
        return tokenized_examples

    if not os.path.exists(tokenized_datapath):
        dataset = load_dataset(
            dataset_path,
            trust_remote_code=True,
        ).map(
            map_callback,
            batched=True,
            batch_size=5000,
            remove_columns=["text"],
            num_proc=32,
        )
        dataset.save_to_disk(tokenized_datapath)
    else:
        dataset = DatasetDict.load_from_disk(tokenized_datapath)

    print("Example input:\n", dataset['train'][0])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
        # pad_to_multiple_of=8, # For efficient Flash Attention
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.1),
            TextGenerationCallback(tokenizer, prompt="A long time ago"),
            GradientMonitorCallback(),
        ],
    )
    
    trainer.train()
    model.config.use_cache = True
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    wandb.save(output_path)


if __name__ == "__main__":
    main()
