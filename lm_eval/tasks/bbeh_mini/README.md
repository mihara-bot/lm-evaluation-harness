# BBEH Mini

BIG-Bench Extra Hard (BBEH) is a harder successor to BBH. The official
repository provides a full set with 4,520 examples and a mini set with 460
examples.

This task evaluates the official mini set from
`google-deepmind/bbeh/bbeh/mini/data.json` with generated answers. Scoring uses
the answer extraction and fuzzy exact-match rules from the official
`bbeh/evaluate.py` implementation.

## Tasks

- `bbeh-mini`: the 460-example BBEH mini set.

The mini JSON file contains only `input` and `target` fields, so this
implementation reports the overall mini score rather than per-subtask scores.

## Citation

```bibtex
@article{bbeh,
  title={BIG-Bench Extra Hard},
  author={Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, Nishanth Dikkala, Gladys Tyen, Xin Liu, Uri Shalit, Silvia Chiappa, Kate Olszewska, Yi Tay, Vinh Q. Tran, Quoc V. Le, Orhan Firat},
  journal={arXiv preprint arXiv:2502.19187},
  year={2025},
}
```
