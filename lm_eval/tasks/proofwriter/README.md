# ProofWriter

ProofWriter is a natural-language logical reasoning benchmark. Each example
contains a set of facts and rules, a query statement, and a gold answer from
`True`, `False`, or `Unknown`.

This implementation evaluates answer selection only. It uses the
`tasksource/proofwriter` Hugging Face dataset and scores the model with
multiple-choice accuracy over the three labels.

## Tasks

- `proofwriter`: group that runs the overall task and depth-specific tasks.
- `proofwriter_all`: all examples from the dataset split.
- `proofwriter_depth0` through `proofwriter_depth5`: examples filtered by
  `QDep`, the per-question proof depth. If `QDep` is unavailable, the task
  falls back to `maxD` and then the depth encoded in `config`.

## Citation

```bibtex
@inproceedings{tafjord-etal-2021-proofwriter,
  title = "{P}roof{W}riter: Generating Implications, Proofs, and Abductive Statements over Natural Language",
  author = "Tafjord, Oyvind and Dalvi, Bhavana and Clark, Peter",
  booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
  year = "2021",
}
```
