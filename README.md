# Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection #
This repo is the PyTorch codes for "Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection"
> [**Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**](https://arxiv.org/pdf/2406.11260)
> >


## Overall model architecture ##
<center><img src="./figure/model_arch.PNG"> </center>

## Usage ##
```
usage: main_adstyle.py [-h] [--train_dir TRAIN_DIR] [--test_dir TEST_DIR] [--pretrained_dir PRETRAINED_DIR] [--api_key API_KEY] [--num_shards NUM_SHARDS] [--batch_size BATCH_SIZE] [--round ROUND] [--lr LR] [--max_len MAX_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR
                        train set direcotry
  --test_dir TEST_DIR   test set direcotry
  --pretrained_dir PRETRAINED_DIR
                        sheepdog pretrained model direcotry
  --api_key API_KEY     OpenAI API KEY
  --num_shards NUM_SHARDS
  --batch_size BATCH_SIZE
                        batch size
  --round ROUND         training rounds
  --lr LR               learning rate
  --max_len MAX_LEN     max token length
```


## Model ZOO ##
Currently, we provide the pretrained model (Sheepdog) used as a starting point for our training.

### Pretrained Model ###
| Dataset           | model | 
|-------------------|---------------|
|Politifact         | [Download]()  |
|Gossipcop          | [Download]()  |
|Constraint         | [Download]()  |



## Citation

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{park2025adversarial,
  title={Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection},
  author={Park, Sungwon and Han, Sungwon and Xie, Xing and Lee, Jae-Gil and Cha, Meeyoung},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={4024--4033},
  year={2025}
}
```
