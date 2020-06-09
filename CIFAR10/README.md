# CIFAR10 Experiments

The code is adapted from [here](https://github.com/google-research/simclr)


## Usage

### Evaluating Self-supervised Representations

Contrastive Learning Objective only
```
python run.py --train_mode=pretrain --train_batch_size=512 --train_epochs=1000 \
              --learning_rate=1.0 --weight_decay=1e-6 --dataset=cifar10 --image_size=32 \
              --eval_split=test --resnet_depth=18 --use_blur=False --color_jitter_strength=0.5 \
              --model_dir=/root/data/githubs/simclr_models/c10_cpc_1 --inv_pred_coeff=0.0 \ 
              --use_tpu=False --temperature=0.5 --hidden_norm=True

python run.py --mode=train_then_eval --train_mode=finetune --fine_tune_after_block=4 \
              --zero_init_logits_layer=True --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
              --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
              --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 --dataset=cifar10 --image_size=32 \
              --eval_split=test --resnet_depth=18 --checkpoint=/root/data/githubs/simclr_models/c10_cpc_1 \
              --model_dir=/root/data/githubs/simclr_models/c10_cpc_1/ft --inv_pred_coeff=0.0 --use_tpu=False \
              --temperature=0.5 --hidden_norm=True 
```

Contrastive Learning Objective + Inverse Predictive Learning Objective
```
python run.py --train_mode=pretrain --train_batch_size=512 --train_epochs=1000 \
              --learning_rate=1.0 --weight_decay=1e-6 --dataset=cifar10 --image_size=32 \
              --eval_split=test --resnet_depth=18 --use_blur=False --color_jitter_strength=0.5 \
              --model_dir=/root/data/githubs/simclr_models/c10_cpc_inv_1 --inv_pred_coeff=0.03 \ 
              --use_tpu=False --temperature=0.5 --hidden_norm=True

python run.py --mode=train_then_eval --train_mode=finetune --fine_tune_after_block=4 \
              --zero_init_logits_layer=True --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
              --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
              --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 --dataset=cifar10 --image_size=32 \
              --eval_split=test --resnet_depth=18 --checkpoint=/root/data/githubs/simclr_models/c10_cpc_1 \
              --model_dir=/root/data/githubs/simclr_models/c10_cpc_inv_1/ft --inv_pred_coeff=0.03 --use_tpu=False \
              --temperature=0.5 --hidden_norm=True 
```
