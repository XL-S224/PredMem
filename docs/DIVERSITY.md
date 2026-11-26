# Diversity protocol

1. Evaluate every generated output; do not discard failed/nonfinite samples from the denominator.
2. Label successful joint outputs with backbone scRMSD<2Å. Motif additionally requires generated motif RMSD<1Å. Keep the alternative repository criterion (scTM>.8 and refolded-motif RMSD<1Å) separate.
3. Cluster generator-decoded structures with Foldseek10.941cd33, TM threshold.5, alignment-type1, cov-mode0, min-seq-id0, four threads. The historical Linux-x86 binary SHA256 is `ce5f08d8f1e15598f618ac6155a756608fe0939b9534e773b5b5472a74acb842`.
4. Cluster within length or motif; sum group counts. Report successful-subset clusters separately from all-output clusters.
5. For cluster probabilities p_j=n_j/N in each group, effective clusters are exp(−Σp_j log p_j). Sum these group-level values; do not pool incomparable length strata first.
6. Compare equal sample budgets. Changes in successful output count influence successful-subset cluster counts. Matched-success subsampling is an additional analysis, not interchangeable with the raw counts.
7. Aggregate seed-level values with equal weight and sample SD (ddof1). For one seed, SD across seeds is undefined. Do not label a sample bootstrap as seed variation.

Different Foldseek builds can change partitions for identical PDBs. The launcher enforces version or binary hash constraints when configured. Clustering records the actual version and binary hash.
