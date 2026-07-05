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

## Metrics

- `final_statement_match`: whether the model output contains the gold final
  proof statement.
- `proof_exact_match`: whether the generated proof exactly matches the gold
  chain after whitespace normalization.

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
