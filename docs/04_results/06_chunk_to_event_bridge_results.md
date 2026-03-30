# Chunk-to-Event Bridge Results

Notebook: `ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb`

## Scope

This page summarizes the first executed results from the new chunk-to-event bridge implementation.

## 1) Model Cost Profile (Current)

Measured rows:

- distilbert: chunk_f1 0.9566, train_seconds 120.0704, params_millions 66.3806
- bert-base-uncased: chunk_f1 0.9602, train_seconds 222.7203, params_millions 108.9093
- roberta-base: chunk_f1 0.9665, train_seconds 224.6860, params_millions 124.0727

Pending rows:

- google/bert_uncased_L-2_H-128_A-2
- prajjwal1/bert-tiny
- huawei-noah/TinyBERT_General_4L_312D

## 2) Recommendation Policy Output

Policy settings used in notebook:

- `max_time_multiplier = 2.0`
- `min_f1_gain = 0.005`

Observed recommendation result:

- quality-first candidate: roberta-base
- latency-first candidate: roberta-base
- tiny model entries retained as pending and excluded from measured recommendation ranking

## 3) Email Event Pipeline Output (Current Run)

From the executed email demo cell:

- accepted events: 5/5
- total cost estimate: 38.00
- cost per accepted event: 7.60

Interpretation:

- pipeline executed end-to-end with confidence and abstention fields available
- current cost estimate is still a proxy metric tied to model parameter scale

## 4) Result Meaning

The bridge notebook confirms that:

1. shallow chunk outputs can be converted into typed event records with confidence and abstention metadata,
2. compute-aware recommendation can be layered on top of chunk-model quality data,
3. pending-model-aware reporting is working, enabling incremental scaling studies without breaking analysis.

## 5) Next Metrics to Add

1. measured inference latency/token and memory usage
2. event-level precision/recall/F1 on labeled event sets
3. cost-per-correct-event and utility-per-budget curves