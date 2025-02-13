"""
This module demostrates the pretrain a launage model with transformers.
- dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- model: roneneldan/TinyStories-1M
- reference: https://arxiv.org/pdf/2305.07759

You are free to use any dataset and pre-trained model

# https://huggingface.co/datasets/roneneldan/TinyStories
# https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import torch
import torch.distributed as dist
from itertools import chain
from datasets import load_dataset, DatasetDict
import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

dataset_path = "roneneldan/TinyStories"
model_path = "roneneldan/TinyStories-1M"
output_path = f"results/${model_path}/pt"
tokenized_datapath = os.path.join("data", dataset_path, "tokenized_dataset")

os.makedirs(output_path, exist_ok=True)


def main():

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        # https://huggingface.co/docs/transformers/perf_infer_gpu_one
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=16,
        save_steps=10_000,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1,  # clip gradient at max 1
        eval_strategy="steps",
        eval_steps=10_000,
        do_eval=True,
        logging_steps=10,
        report_to="wandb",
        remove_unused_columns=False,
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
            project="hands-on-llm-pt",
            name=f"{model_path.replace("/","-")}-pt",
            config=training_args.to_dict(),
        )
    else:
        os.environ["WANDB_MODE"] = "offline"

    # process dataset
    def map_callback(examples):
        sequence_length = 512
        eos_token = "<|im_end|>"
        text_examples = [text + eos_token for text in examples["text"]]
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {
            k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
        }
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # align the sequence
        total_length = (total_length // sequence_length) * sequence_length
        result = {
            k: [
                t[i : i + sequence_length]
                for i in range(0, total_length, sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

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

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
