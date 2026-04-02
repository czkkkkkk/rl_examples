# train_grpo.py
import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.rewards import accuracy_reward
from math_verify import parse

parser = TrlParser(GRPOConfig)
(config,) = parser.parse_args_and_config()

if os.environ.get("SLURM_TB_LOG_DIR"):
    config.logging_dir = os.environ["SLURM_TB_LOG_DIR"]

SYSTEM_PROMPT = "Solve the math problem. Put your final answer in \\boxed{}."

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
dataset = dataset.filter(lambda x: len(parse(x["solution"])) > 0)
dataset = dataset.map(
    lambda x: {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}] + x["prompt"]}
)

trainer = GRPOTrainer(
    args=config,
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()