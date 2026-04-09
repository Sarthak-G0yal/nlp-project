# Chunk-to-Event Bridge Results

Notebook: `ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb`

## Scope

This page summarizes the first executed results from the new chunk-to-event bridge implementation.
It reflects the bridge notebook profile snapshot, which may lag behind the latest standalone scaling notebook run.

## Model Cost Profile (Bridge Snapshot)

Snapshot measured rows:

- distilbert: chunk_f1 0.9566, train_seconds 120.0704, params_millions 66.3806
- bert-base-uncased: chunk_f1 0.9602, train_seconds 222.7203, params_millions 108.9093
- roberta-base: chunk_f1 0.9665, train_seconds 224.6860, params_millions 124.0727

Snapshot pending rows:

- google/bert_uncased_L-2_H-128_A-2
- prajjwal1/bert-tiny
- huawei-noah/TinyBERT_General_4L_312D

In the latest standalone scaling run (`ipynb/Larger_Transformer_Comparison.ipynb`), NanoBERT-like and TinyBERT now have measured results, while RoBERTa and bert-tiny failed in that environment-specific execution.

Canonical latest scaling status:

- measured: distilbert, bert-base-uncased, google/bert_uncased_L-2_H-128_A-2, huawei-noah/TinyBERT_General_4L_312D
- failed: roberta-base (CUDA OOM), prajjwal1/bert-tiny (tokenizer backend dependency)

The bridge notebook cost profile should be refreshed to replace stale snapshot rows before using bridge ranking outputs as the final model recommendation.

## Recommendation Policy Output

Policy settings used in notebook:

- `max_time_multiplier = 2.0`
- `min_f1_gain = 0.005`

Observed recommendation result in bridge snapshot:

- quality-first candidate: roberta-base
- latency-first candidate: roberta-base
- tiny model entries retained as pending and excluded from measured recommendation ranking

Current interpretation after syncing with latest standalone scaling state:

- roberta-base should not be treated as currently deployable in this environment because the latest scaling run failed.
- distilbert remains the default deployable baseline until bridge profile cells are rerun with refreshed model rows.

## Email Event Pipeline Output (Current Run)

From the executed email demo cell:

- accepted events: 5/5
- total cost estimate: 38.00
- cost per accepted event: 7.60

Interpretation:

- pipeline executed end-to-end with confidence and abstention fields available
- current cost estimate is still a proxy metric tied to model parameter scale

## Result Meaning

The bridge notebook confirms that:

1. shallow chunk outputs can be converted into typed event records with confidence and abstention metadata,
2. compute-aware recommendation can be layered on top of chunk-model quality data,
3. pending-model-aware reporting is working, enabling incremental scaling studies without breaking analysis.

## Sync Note with Latest Scaling Run

- latest scaling measured: distilbert, bert-base-uncased, google/bert_uncased_L-2_H-128_A-2, huawei-noah/TinyBERT_General_4L_312D
- latest scaling failures: roberta-base (CUDA OOM), prajjwal1/bert-tiny (tokenizer backend dependency)
- bridge recommendation table still reflects the earlier snapshot until its profile cells are rerun

## Next Metrics to Add

1. measured inference latency/token and memory usage
2. event-level precision/recall/F1 on labeled event sets
3. cost-per-correct-event and utility-per-budget curves