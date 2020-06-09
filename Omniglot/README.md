# Omniglot Experiments

The code is adapted from [here](https://github.com/leftthomas/SimCLR)


## Usage

### Evaluating Self-supervised Representations

Contrastive Learning Objective only
```
python main.py --loss_type 1 --recon_param 0.0 --inver_param 0.0 --epochs 1000
```

Others please refer to `main.py`

### Measuring Information

+ Step 1: Train an Auto-Encoder (we assume the encoded features do not lose any information)
```
python compute_MI_CondEntro.py --stage AE
```

+ Step 2: Estimate the raw information
```
python compute_MI_CondEntro.py --stage Raw_Information
```

+ Step 3: Estimate the information for the SSL learned representations

    - for Contrastive Learning Objective only
        ```
        python main.py --loss_type 1 --recon_param 0.0 --inver_param 0.0 --epochs 1000 --with_info
        ```
    - for Contrastive Learning Objective + Inverse Predictive Learning Objective
        ```
        python main.py --loss_type 2 --recon_param 0.0 --inver_param 1.0 --epochs 1000 --with_info
        ```
