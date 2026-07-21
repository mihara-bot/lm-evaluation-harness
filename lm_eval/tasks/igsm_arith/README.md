# OLMo iGSM-Easy Arithmetic

This task evaluates zero-shot symbolic arithmetic over the fixed
[`Mihara-bot/olmo-igsm-arith`](https://huggingface.co/datasets/Mihara-bot/olmo-igsm-arith)
test set. The data is an OLMo i-GSM-style variant with conventional arithmetic
operators, strict left-to-right evaluation, and all results taken modulo 7.

## Tasks

- `igsm_arith`: size-weighted aggregate over all three depths.
- `igsm_arith_d2`: 250 target-depth-2 examples.
- `igsm_arith_d3`: 250 target-depth-3 examples.
- `igsm_arith_d4`: 250 target-depth-4 examples.

The task definitions pin dataset revision
`ff400bde5db5637e76fc3ec931b5d9840ed8e6f9`. The source JSONL has SHA-256
`c8d77a9d3cc9cc6c5631b1b973231b674ea3d8845ec08fe5de406873e7efca9b`.

## Evaluation compatibility

The prompt is exactly:

```text
{preamble}

{equations}
Question: {query}?
Answer:
```

The model is scored by comparing the raw continuation log-likelihoods of
` 0`, ` 1`, ..., ` 6`. Only raw multiple-choice `acc` is reported; this mirrors
the OLMo evaluator and gives a random baseline of `1/7` (about 14.29%). The
generator-provided `cot` field is deliberately excluded from the prompt.
