#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  CUDA_VISIBLE_DEVICES=$1 python grince.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug $3 --lam $4 --q $5 --noise_rate 0.1 --seed $seed

done

