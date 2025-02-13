# Hands-on Large Language Model Projects

| project  | description                                  | code link                    |
| -------- | -------------------------------------------- | ---------------------------- |
| pretrain | Pretrain a LLM with using opensource dataset | [pretrain.py](./pretrain.py) |



## Preparation
Please install [uv](https://github.com/astral-sh/uv) first.
```
git clone https://github.com/duoan/hands-on-llm-projects.git
cd hands-on-llm-projects
uvx sync
source .venv/bin/activate

# install flash attention
uv pip install flash-attn --no-build-isolation

```

## Run

### pretrain
```
# Single GPU
python pretrain.py

# Multi GPUs
# Update accelerate_config.yaml accordingly.
accelerate launch --config-file accelerate_config.yaml pretrain.py
```