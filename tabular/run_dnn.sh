#!/bin/bash
# alpha_values = 0. 0.1 0.3 0.7 1.0 3.0 5.0 10.0 20.0 50.0 100.0 500.0
# datasets = 'crimes' 'adult'
# for dataset in 'adult'
# do
#     for alpha_value in 30.0 50.0 100.0 150.0 200.0
#     do
#         # python -u main_dnn.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 2  \
#         #                         --hyper_pent $alpha_value &
#         python -u main_dnn.py --data $dataset --times 5 --n_epoch 200 --batch_size 200 --gpu 0  \
#                                 --hyper_pent $alpha_value --sens_bn True &
#     done
# done
python -u main_dnn.py --data 'crimes' --times 1 --n_epoch 10 --batch_size 200 --gpu 0  \
                                --hyper_pent 0.0001