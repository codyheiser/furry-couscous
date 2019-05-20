Command used to generate [__scvis__](https://github.com/shahcompbio/scvis) outputs from terminal:
```
scvis train --data_matrix_file inputs/<filename> --out_dir dev/scvis_out/<samplename> --verbose --verbose_interval 20 --config_file dev/scvis_model_config.yaml
```
__Note__: `--data_matrix_file` must be tab-delimited (`.tsv`) with first row as gene names and no cell labels (as in `inputs/GSM1626793_P14Retina_1.processed.norowlabels.tsv`). Counts should be in __float__ format, not __int__.
