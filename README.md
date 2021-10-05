# GDP

GDP is a generalized demographic parity for continuous sensitive attribues.

## Installation

```bash
conda env create -f environment.yml
```
## Dataset
1. Tabular data: Adults and Crimes dataset can be downloaded from (https://archive.ics.uci.edu/ml/datasets/adult) and (https://archive.ics.uci.edu/ml/datasets/communities+and+crime)

2. Graph data: Pokec_z and Pokec_n can be downloaded from (https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec) as `region_job.xxx` and `region_job_2.xxx`, respectively.
They are sampled from [soc_Pokec](http://snap.stanford.edu/data/soc-Pokec.html). 

## Reproduce the results

To reproduce the performance reported in the paper, you can run the bash files in folders `tabular\`,`graph\` and `comp\`.
```
cd tabular
bash run_dnn.sh
bash run_dnn_adv.sh
```

## License
[MIT](https://choosealicense.com/licenses/mit/)