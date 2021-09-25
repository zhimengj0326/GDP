#!/bin/bash
# alpha_values = 0. 10.0 50.0 100.0 500.0 800.0 1000.0 1500 2000.0 3000.0 4000.0 5000.0
# datasets = 'crimes' 'adult'
for dataset in 'crimes' 'adult'
do
    for alpha_value in 0. 10.0 50.0 100.0 500.0 800.0 1000.0 1500 2000.0 3000.0 4000.0 5000.0 8000.0
    do
        # python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 180 --batch_size 200 --gpu 3  \
        #                         --hyper_pent $alpha_value
        python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 180 --batch_size 200 --gpu 3  \
                                --hyper_pent $alpha_value --sens_bn True
    done
done
# for dataset in 'crimes'
# do
#     for alpha_value in 0. 1.0 5.0 10.0 50.0 100.0 500.0 1000.0 2000.0
#     do
#         python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 180 --batch_size 200 --gpu 2  \
#                                 --hyper_pent $alpha_value 
#         python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 180 --batch_size 200 --gpu 3  \
#                                 --hyper_pent $alpha_value --sens_bn True 
#     done
# done

# python -u main_dnn_adv.py --data 'crimes' --times 5 --n_epoch 180 --batch_size 200 --gpu 3  \
#                                 --hyper_pent 5000.0 --sens_bn True