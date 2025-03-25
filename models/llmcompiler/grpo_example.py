# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig


dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", 
                        logging_steps=10,
                           max_completion_length=1024,
                        num_generations=8,
                        save_steps=10,
                        log_completions=True)
training_args.set_training(batch_size=16, 
                        num_epochs=1, 
                        # max_steps=1, 
                        gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    peft_config = lora_config,
)
trainer.train()
