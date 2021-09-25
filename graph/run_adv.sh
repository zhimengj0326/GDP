#!/bin/bash
### model = GCN GAT SGC
### data = pokec_n pokec_z
### alpha_values = 0.0 1.0 10.0 50.0 100.0 500.0 1000.0 1500.0 2000.0

for dataset in pokec_z
do
    for model in SGC
    do
        for alpha_value in 1.0 3.0
        do
            python node_adv.py --gpu 0 --seed=42 --epochs=50 --model=$model \
                    --dataset=$dataset --num-hidden=64 \
                    --attn-drop=0.0 --hyper $alpha_value --prefix adv &

            python node_adv.py --gpu 1 --seed=42 --epochs=50 --model=$model \
                    --dataset=$dataset --num-hidden=64 \
                    --attn-drop=0.0 --hyper $alpha_value --sens_bn True --prefix adv_bn &
        done
    done
done

# python node_adv.py --gpu 2 --seed=42 --epochs=50 --model=GAT --dataset=pokec_n --num-hidden=64 \
#                 --attn-drop=0.0 --hyper 0.1 --prefix adv
