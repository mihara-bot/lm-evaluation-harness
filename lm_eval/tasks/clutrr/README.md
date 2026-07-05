# CLUTRR

CLUTRR (Compositional Language Understanding and Text-based Relational
Reasoning) evaluates whether models can infer kinship relations from short
stories.

This implementation reads the `gen_train234_test2to10` CSV files mirrored by
the original `CLUTRR/v1` Hugging Face dataset script. It avoids relying on the
dataset script itself, because recent versions of `datasets` no longer execute
dataset scripts. The task evaluates answer selection over the 18 kinship labels
in the dataset.

## Tasks

- `clutrr`: group that runs the overall task and task-length-specific tasks.
- `clutrr_all`: all examples from the selected split.
- `clutrr_task_1_2` through `clutrr_task_1_10`: examples filtered by the
  CLUTRR `task_name` field.

## Citation

```bibtex
@article{sinha2019clutrr,
  Author = {Koustuv Sinha and Shagun Sodhani and Jin Dong and Joelle Pineau and William L. Hamilton},
  Title = {CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text},
  Year = {2019},
  journal = {Empirical Methods of Natural Language Processing (EMNLP)},
  arxiv = {1908.06177}
}
```
