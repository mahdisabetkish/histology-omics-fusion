# Running on a Slurm cluster

The published results were produced on a single workstation GPU, not here. These
scripts exist so the pipeline can be reproduced on a cluster without rewriting
anything, and because the experiment matrix is embarrassingly parallel: the
twelve supervised runs have no dependency on each other once pretraining is
done.

Adjust the `#SBATCH` headers to your site's partition, account and module names
before submitting. Nothing below assumes a particular scheduler configuration
beyond GPU allocation through `--gres`.

## Order

```bash
# 1. Data. CPU only, no GPU needed, and the slow part is the network.
jid_dl=$(sbatch --parsable slurm/01_data.sbatch)

# 2. Self-supervised pretraining. One GPU, three variants as an array.
jid_ssl=$(sbatch --parsable --dependency=afterok:$jid_dl slurm/02_pretrain.sbatch)

# 3. Everything that depends on a pretrained checkpoint, plus the
#    supervised baselines. Array over the run matrix.
jid_tr=$(sbatch --parsable --dependency=afterok:$jid_ssl slurm/03_train.sbatch)

# 4. Scoring and dashboard export.
sbatch --dependency=afterok:$jid_tr slurm/04_export.sbatch
```

Or in one call:

```bash
bash slurm/submit_all.sh
```

## Notes

`--array` indices map to entries in the job lists inside each script. Re-running
an array is safe: every stage skips work that already produced a
`summary.json`, so a partial failure can be resubmitted without `--force`.

The download stage writes roughly 6.6 GB and the preprocessing stage another
2.3 GB. Both go under `data/` in the repository directory, so submit from a
filesystem with room and reasonable read performance, not from a home quota.

GPU memory never exceeds about 4 GB at the default batch sizes, so a small
allocation is enough. Wall time is dominated by the image encoder; the
expression-only runs finish in a couple of minutes.
