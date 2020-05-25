# Simple Segmentation

## Training

```sh
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -u -m train \
    --lr 0.02 \
    --dataset coco \
    --batch-size 32 \
    --model fcn_resnet50 \
    --aux-loss \
    --output-dir [CHECKPOINT_PATH] \
    2>&1 | tee [TRAINING_LOG]
```
