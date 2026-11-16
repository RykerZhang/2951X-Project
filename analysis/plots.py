"""Plots (owned by the experiments & analysis lane). Reads only the JSONL files in logs/;
never re-runs experiments.

TODO:
1. Composition-matrix heatmap: rows = skill A, columns = skill B, cell = fraction of runs where
   B succeeds right after A (one plot for random starts, one for in-chain starts)
2. Failure-distribution bar chart: one group per condition, share of L1-L4 / other,
   with error bars over the 10 seeds
3. Confusion matrix: rows = label_true, columns = label_runtime (once T2 is defined)
"""
