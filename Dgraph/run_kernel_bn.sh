#!/bin/bash
# t-gat learning on harris data
## tasks = 'clf' 'reg'
## attn_modes = prod map
## alpha_value = 0.0 0.3 1.0 5.0 10.0 20.0 50.0 100.0
## sens_bn = 'yes' 'no'
for task in clf reg
do
    for attn_mode in prod map
    do
        for alpha_value in 0.05 0.1 0.15 0.2
        do
            python -u learn_node.py --data 'harris' --n_epoch 200 --bs 2000 --uniform  \
                        --sens_bn yes --n_degree 20 --clf $task --attn_mode $attn_mode \
                        --hyper_pent $alpha_value --running_times 5 --gpu 1 --n_head 2 &
        done
        wait
    done
done

# python -u learn_node.py --data 'harris' --n_epoch 200 --bs 200 --uniform  \
#                         --n_degree 20 --clf True --attn_mode map \
#                         --hyper_pent 0.1 --running_times 5 --gpu 1 --n_head 2 --prefix kernel
