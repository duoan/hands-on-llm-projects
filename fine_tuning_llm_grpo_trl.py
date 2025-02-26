import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import wandb


@dataclass
class JobConfig:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    dataset_id = "AI-MO/NuminaMath-TIR"
    base_model_id = "Qwen/Qwen2-0.5B-Instruct"
    finetune_model_id = "Qwen2-0.5B-GRPO-NuminaMath"
    run_name = finetune_model_id + "_run_" + run_id
    output_dir = os.path.join("results", finetune_model_id, "grpo", run_id)


class ReasoningGenerationCallback(TrainerCallback):
    """Regularly generate the reasoning for the giving example prompt.
    It helps us to debug and understand the model performance step by step.
    """

    def __init__(
        self,
        tokenizer,
        prompt,
        max_length=500,
        log_interval=10,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate reasoning when happens"""
        if model is None or state.global_step % self.log_interval > 0:
            return

        model.eval()  # Switch to eval mode
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_length=self.max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Step {state.global_step}] Generated Reasoning: {generated_text}\n")

        model.train()  # Switch back to training mode


job_config = JobConfig()

# Init W&D to track training performance
wandb.login()
wandb.init(
    project=f"hands-on-llm-grpo-{job_config.finetune_model_id}",
    id=job_config.run_id,
    config=job_config,
)

# Preparing dataset
train_dataset, test_dataset = load_dataset(
    job_config.dataset_id, split=["train[:5%]", "test[:5%]"]
)

print(train_dataset)

print("Math Example:\n", train_dataset[0])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant."
    "The user asks a question, and the Assistant solves it."
    "The assistant first thinks about the reasoning process in the mind step by step and then provides the user with the answer."
    "The reasoning process and answer are must strictly enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


train_dataset = train_dataset.map(make_conversation).remove_columns("messages")
test_dataset = test_dataset.map(make_conversation).remove_columns("messages")
print("Conversation Example\n", train_dataset[0]["prompt"])


# Loading the Baseline Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(job_config.base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    job_config.base_model_id,
    torch_dtype="auto",
    device_map="auto",
)


## Configuring LoRA
# LoRA will allow us to efficiently fine-tune the model with a reduced number of parameters,
# enabling faster and more resource-efficient training.
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Loading Reward Functions

# Format Enforcement:
# Ensures that the generation follows a specific format using <think> </think> <answer> </answer> tags for reasoning.


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


# Solution Accuracy: Verifies whether the solution to the problem is correct.
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


## Configuring Group Relative Policy Optimization (GRPO) Training Parameters
# To be simple, just train one epoch and reducing the max_completion_length, num_generations,
# and max_prompt_length from their default values

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    run_name=job_config.run_id,
    output_dir=job_config.output_dir,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=["wandb"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

## Training the Model

test_prompt = test_dataset["prompt"][0]
test_prompt = " ".join(entry["content"] for entry in test_prompt)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[
        ReasoningGenerationCallback(
            tokenizer, prompt=test_prompt, log_interval=training_args.logging_steps
        )
    ],
)

trainer.train()

trainer.save_model(training_args.output_dir)
trainer.push_to_hub(
    dataset_name=job_config.dataset_id,
    model_name=job_config.finetune_model_id,
)
