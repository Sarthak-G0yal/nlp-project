# 14. Error Analysis

## Baseline Error Pattern

The baseline made frequent mistakes in:

- minority chunk classes with small support
- chunk boundary transitions for less frequent phrase forms

This is expected from sparse local features with limited context.

## Transformer Error Pattern

The transformer substantially reduces systematic boundary errors and improves consistency on dominant chunk categories.

Remaining errors are likely concentrated in:

- rare labels
- ambiguous boundary contexts
- annotation edge cases

## Why This Matters

For information extraction pipelines, improved chunk boundaries directly improve downstream extraction quality and reduce noisy spans.
