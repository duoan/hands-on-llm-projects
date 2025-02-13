# Hands-on Large Language Model Projects

| project  | description                                  | code link                    |
| -------- | -------------------------------------------- | ---------------------------- |
| pretrain | Pretrain a LLM with using opensource dataset | [pretrain.py](./pretrain.py) |

## Accelerate

### flash attention

```
uv pip install flash-attn --no-build-isolation
```

### DeepSpeed

```
uv add deepspeed
# update accelerate_config.yaml accordingly.
accelerate launch --config-file accelerate_config.yaml pretrain.py
```
