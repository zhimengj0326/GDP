#!/bin/bash
### model = GCN GAT SGC
### data = pokec_n pokec_z
### alpha_values = 0.0 0.3 1.0 5.0 10.0 20.0 50.0 100.0 200.0 500.0 1000.0 2000.0

# for dataset in pokec_z
# do
#     for model in SGC
#     do
#         for alpha_value in 0.0 0.3 1.0 5.0 10.0 20.0 50.0 100.0
#         do
#             python node_base.py --gpu 2 --seed=42 --epochs=300 --model=$model \
#                     --dataset=$dataset --num-hidden=64 \
#                     --attn-drop=0.0 --hyper $alpha_value --prefix kernel &
#             python node_base.py --gpu 3 --seed=42 --epochs=300 --model=$model \
#                     --dataset=$dataset --num-hidden=64 \
#                     --attn-drop=0.0 --hyper $alpha_value --sens_bn True --prefix kernel_bn &
#         done
#     done
# done

python node_base.py --gpu 3 --seed=42 --epochs=10 --model=GAT --dataset=pokec_n --num-hidden=64 \
                --attn-drop=0.0 --hyper 0.001 --prefix kernel
