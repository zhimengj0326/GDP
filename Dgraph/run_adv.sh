#!/bin/bash
# t-gat learning on harris data
## tasks = clf reg
## attn_modes = prod map
## alpha_value = 0.0 100.0 500.0 1000.0 2000.0 5000.0
## sens_bn = yes no
for task in clf reg
do
    for attn_mode in prod map
    do
        for alpha_value in 50.0 100.0 300.0 500.0 800.0 1000.0 2000.0 3000.0 5000.0
        do
            python -u learn_node_adv.py --data 'harris' --n_epoch 200 --bs 2000 --uniform  \
                        --sens_bn no --n_degree 20 --clf $task --attn_mode $attn_mode \
                        --hyper_pent $alpha_value --running_times 5 --gpu 3 --n_head 2 &
        done
        wait
    done
done

# python -u learn_node.py --data 'harris' --n_epoch 200 --bs 200 --uniform  \
#                         --n_degree 20 --clf True --attn_mode map \
#                         --hyper_pent 0.1 --running_times 5 --gpu 1 --n_head 2 --prefix kernel
