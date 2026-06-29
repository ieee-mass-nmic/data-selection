# Table 1 — PEFT configurations and trainable parameters (llama2-7b, 32 layers; #trainable = estimate)

| PEFT | Family | Inserted into | Layers | Operator | Capacity | # Trainable | % backbone | Touched sites |Ω_p\| |
|---|---|---|---|---|---|--:|--:|---|
| `L-r8-qv` | lora | q_proj,v_proj | all (32) | add. low-rank | r=8 | 4,194,304 | 0.062% | 8/24 |
| `L-r8-mlp` | lora | up_proj,down_proj | all (32) | add. low-rank | r=8 | 4,194,304 | 0.062% | 8/24 |
| `L-r4-qv` | lora | q_proj,v_proj | all (32) | add. low-rank | r=4 | 2,097,152 | 0.031% | 8/24 |
| `L-r32-qkvo` | lora | q_proj,k_proj,v_proj,o_proj | all (32) | add. low-rank | r=32 | 33,554,432 | 0.498% | 8/24 |
| `L-r8-lowlayers` | lora | q_proj,v_proj | low ⅓ (10) | add. low-rank | r=8 | 1,310,720 | 0.019% | 2/24 |
| `L-r8-highlayers` | lora | q_proj,v_proj | high ⅓ (10) | add. low-rank | r=8 | 1,310,720 | 0.019% | 3/24 |
| `IA3-attnmlp` | ia3 | k_proj,v_proj,down_proj | all (32) | multiplicative | — | 393,216 | 0.006% | 16/24 |
| `AD-b64` | adapter | o_proj,down_proj | all (32) | add. bottleneck | b=64 | 16,777,216 | 0.249% | 24/24 |
