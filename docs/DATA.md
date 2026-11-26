# Benchmark data

Use `cameo2022.tar.gz`, `pdb_date.tar.gz`, and `scaffolding_pdbs.tar.gz` from the reference implementation (`EchoChou990919/mplm_inference`). With the project environment active, extract the relevant archives from the repository root.

```bash
python scripts/prepare_data.py --archive /path/to/cameo2022.tar.gz --destination data
python scripts/prepare_data.py --archive /path/to/pdb_date.tar.gz --destination data
python scripts/prepare_data.py --archive /path/to/scaffolding_pdbs.tar.gz --destination data
```

```text
data/
├── cameo2022/
│   ├── aatype.fasta
│   ├── struct.fasta
│   ├── pdb/
│   └── preprocessed/
├── pdb_date/
│   ├── aatype.fasta
│   ├── struct.fasta
│   ├── pdb/
│   └── preprocessed/
└── scaffolding_pdbs/
    ├── aa_seq.fasta
    ├── struct_seq.fasta
    └── ori_pdbs/
```

Keep the original target IDs, order, reference masks, and residue intervals. Structure prediction reads `aatype.fasta`. DPLM inverse folding reads structure tokens from `struct.fasta`. ESM3 inverse folding uses reference coordinates from the original `preprocessed/` pickles when available and the reference PDB otherwise. Only load pickles and checkpoints from trusted sources.

Motif reference spans are in `configs/motif-reference-spans.yaml`. Keep the original variant grouping when evaluating solved variants and clustering.

| Archive | SHA-256 |
|---|---|
| `cameo2022.tar.gz` | `543f6ddc48c7a9a7849a33c9a36bbb4dc97a5c65e9e22865f46e884a83511fb6` |
| `pdb_date.tar.gz` | `ea114d21f3dcc58e6b1b8651ed7439a3288fc3e7fb11ade4b168f0ec4969974f` |
| `scaffolding_pdbs.tar.gz` | `7e05bf199ae8e8c5afb8ea286479606d24f29a0b5842a18f85aaf3001fc473ee` |
