# Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection #
This repo is the PyTorch codes for "Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection"
> [**Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**](https://arxiv.org/pdf/2406.11260)
> >


## Overall model architecture ##
<center><img src="./figure/model_arch.PNG"> </center>

## Usage ##
augment_train/test_dataset.py : Code for applying style conversion prompts to the initial training and test datasets



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
