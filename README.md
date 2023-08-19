

# <div align="center"> **Spatio-Temporal Masked Pre-training for Traffic Forecasting** </div>

* We implement our code based on [STEP](https://github.com/zezhishao/STEP/tree/github ) and  [BasicTS](https://github.com/zezhishao/BasicTS/tree/master).

## 💿 Dependencies

### OS

Linux systems (*e.g.* Ubuntu and CentOS). 

### Python

The code is built based on Python 3.9, PyTorch 1.13.0, and [EasyTorch](https://github.com/cnstark/easytorch).
You can install PyTorch following the instruction in [PyTorch](https://pytorch.org/get-started/locally/). 

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended to create a virtual python environment.

### Other Dependencies

```bash
pip install -r requirements.txt
```



## Getting started

### Download Data

You can download data from [BasicTS](https://github.com/zezhishao/BasicTS/tree/master) and unzip it.

### Preparing Data


- **Pre-process Data**

You can pre-process all datasets by


    cd /path/to/your/project
    bash scripts/data_preparation/all.sh

Then the `dataset` directory will look like this:

```text
datasets
   ├─PEMS03
   ├─PEMS04
   ├─PEMS07
   ├─PEMS08
   ├─raw_data
   |    ├─PEMS03
   |    ├─PEMS04
   |    ├─PEMS07
   |    ├─PEMS08
   ├─README.md
```

### Pre-training on S-Mask and T-Mask

```
cd /path/to/your/project
```

Then run the folloing command to run in Linux screen.

```
screen -d -m python stmask/run.py --cfg='stmask/TMask_PEMS03.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/TMask_PEMS04.py' --gpus='0'

screen -d -m python stmask/run.py --cfg='stmask/TMask_PEMS07.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/TMask_PEMS08.py' --gpus='0'

screen -d -m python stmask/run.py --cfg='stmask/SMask_PEMS03.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/SMask_PEMS04.py' --gpus='0'

screen -d -m python stmask/run.py --cfg='stmask/SMask_PEMS07.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/SMask_PEMS08.py' --gpus='0'
```



### Downstream Predictor

After pre-training , copy your pre-trained best checkpoint to `mask_save/`.
For example:

```bash
cp checkpoints/TMask_200/5afe80b3e7a3dc055158bcfe99afbd7f/TMask_200_best_val_MAE.pt tsformer_ckpt/TSFormer_$DATASET_NAME.pt
```

```
screen -d -m python stmask/run.py --cfg='stmask/STMask_PEMS04.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/STMask_PEMS03.py' --gpus='0' 

screen -d -m python stmask/run.py --cfg='stmask/STMask_PEMS08.py' --gpus='0'

screen -d -m python stmask/run.py --cfg='stmask/STMask_PEMS07.py' --gpus='0' 
```



## 📉  Results table

![Main results.](results/results.png)

