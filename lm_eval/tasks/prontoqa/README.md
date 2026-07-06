# PrOntoQA

PrOntoQA is a synthetic question-answering benchmark for analyzing deductive
reasoning chains in language model outputs.

This implementation uses the `tasksource/prontoqa` Hugging Face files for the
`ProofsOnly` random/no-adjacency setting. It flattens the raw JSON files into
one evaluation document per `test_example`, using the bundled in-context
examples to format the prompt.

## Tasks

- `prontoqa`: group that runs the overall task and hop-specific tasks.
- `prontoqa_all`: 1-hop through 4-hop examples.
- `prontoqa_hop1` through `prontoqa_hop4`: examples filtered by hop count.
- `prontoqa_mc`: multiple-choice/loglikelihood group for the overall task.
- `prontoqa_all_mc`: 1-hop through 4-hop examples scored with
  multiple-choice loglikelihood.
- `prontoqa_mc_by_hop`: multiple-choice/loglikelihood group for hop-specific
  reporting without also running the overall task.
- `prontoqa_hop1_mc` through `prontoqa_hop4_mc`: hop-specific
  multiple-choice/loglikelihood variants.

## Metrics

- `final_statement_match`: whether the model output contains the gold final
  proof statement.
- `proof_exact_match`: whether the generated proof exactly matches the gold
  chain after whitespace normalization.
- `acc` / `acc_norm`: for the multiple-choice variants, whether the model
  assigns higher loglikelihood to the gold proof than to a hard negative proof
  created by negating the final proof statement.

## Citation

```bibtex
@inproceedings{
  PrOntoQA,
  title={Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought},
  author={Abulhair Saparov and He He},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=qFVVBzXxR2V}
}
```
