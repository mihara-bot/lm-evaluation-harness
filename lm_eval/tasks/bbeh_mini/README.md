# BBEH Mini

BIG-Bench Extra Hard (BBEH) is a harder successor to BBH. The official
repository provides a full set with 4,520 examples and a mini set with 460
examples.

This task evaluates the official mini set from
`google-deepmind/bbeh/bbeh/mini/data.json` with generated answers. Scoring uses
the answer extraction and fuzzy exact-match rules from the official
`bbeh/evaluate.py` implementation.

## Tasks

- `bbeh-mini`: group that runs the overall mini task and all subtask splits.
- `bbeh_mini_all`: the full 460-example BBEH mini set.
- `bbeh_mini_<subtask>`: one 20-example split for each BBEH subtask.

The official mini JSON file contains only `input` and `target` fields. The
subtask labels here were recovered by exact-matching the 460 mini examples
against the official full BBEH task files; every subtask matches exactly 20 mini
examples, with no missing or duplicated examples.

## Citation

```bibtex
@article{bbeh,
  title={BIG-Bench Extra Hard},
  author={Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, Nishanth Dikkala, Gladys Tyen, Xin Liu, Uri Shalit, Silvia Chiappa, Kate Olszewska, Yi Tay, Vinh Q. Tran, Quoc V. Le, Orhan Firat},
  journal={arXiv preprint arXiv:2502.19187},
  year={2025},
}
```
