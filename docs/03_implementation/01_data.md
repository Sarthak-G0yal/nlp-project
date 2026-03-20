# 8. Data and Schema

## Dataset

CoNLL-2000 formatted as parquet files:

- `dataset/train.parquet`
- `dataset/test.parquet`

## Data Shape

- train: 8,937 sentences
- test: 2,013 sentences

## Columns

- `id`: sentence id
- `tokens`: list of tokens
- `pos_tags`: list of POS label ids
- `chunk_tags`: list of chunk label ids

## Validation Performed

- required column check
- token/chunk sequence length consistency checks
- decoding integer ids to readable labels for analysis
