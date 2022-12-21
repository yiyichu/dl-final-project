# Deep Learning Final Project
### Yi-Yi Chu and Omar Hammami

## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

For different experiments, please check the python script to see what directory needs to be created.

## Training & Evaluation

For GCL-AllAug, run the following script 
```
./go.sh $GPU_ID $DATASET_NAME random4 0.1
```


For GCL-NoAug, run the following script 
```
./go_noaug.sh $GPU_ID $DATASET_NAME random4 0.1
```

For Uniformity Loss, run the following script 
```
./go_uniform.sh $GPU_ID $DATASET_NAME random4 0.1
```

For random initialized GIN, run the following script 
```
./go_random.sh $GPU_ID $DATASET_NAME random4 0.1
```

For hard negative GCL-NoAug, run the following script 
```
./go_beta_1.sh $GPU_ID $DATASET_NAME random4 0.1
```

For easy negative GCL-NoAug, run the following script 
```
./go_beta_2.sh $GPU_ID $DATASET_NAME random4 0.1
```

## t-SNE Visualization

To run t-SNE visulization, run the following script 
```
python gsimclr_tsne.py $GPU_ID $DATASET_NAME random4 0.1
```

## New Augmentation
To run expriments with new augmentation, run the following script
```
python gsimclr.py --DS NCI1 --lr 0.01 --local --num-gc-layers 3 --aug all1 --seed 1
```



## Acknowledgements

The backbone implementation is reference to https://github.com/Shen-Lab/GraphCL.
