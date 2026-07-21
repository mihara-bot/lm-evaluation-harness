# RuleTaker

RuleTaker evaluates closed-world logical reasoning over natural-language facts
and rules. This implementation mirrors the OLMo evaluation setup: it uses the
`original` RuleTaker V2020.2.5 `meta-test` files, reconstructs each theory from
the ordered triples and rules, and scores zero-shot log-likelihood over the
lowercase continuations ` true` and ` false`.

## Tasks

- `ruletaker`: size-weighted aggregate over the five supported depths.
- `ruletaker_d0`
- `ruletaker_d1`
- `ruletaker_d2`
- `ruletaker_d3`
- `ruletaker_d5`

The official archive is downloaded and extracted through the Hugging Face
datasets cache. For offline evaluation, call the task through a custom YAML
that includes the depth YAML and overrides `dataset_kwargs.data_dir` with an
already extracted RuleTaker directory.

## Metric compatibility

The primary and only metric is raw multiple-choice `acc`. This matches OLMo's
sum of continuation log-likelihoods; length-normalized `acc_norm` is
intentionally not reported.

## Citation

```bibtex
@inproceedings{clark2020transformers,
  title={Transformers as Soft Reasoners over Language},
  author={Peter Clark and Oyvind Tafjord and Kyle Richardson},
  booktitle={IJCAI},
  year={2020}
}
```
