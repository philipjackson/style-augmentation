# README #

Augment training images for deep neural networks by randomizing their visual style, as described in our paper: https://arxiv.org/abs/1809.05375

To the best of our knowledge, this is also the only PyTorch implementation (with trained weights) of the arbitrary style transfer network in "Exploring the structure of a real-time, arbitrary neural artistic stylization network": https://arxiv.org/abs/1705.06830

## Installation

```bash
python setup.py install
```

## Usage example

```python
from styleaug import StyleAugmentor

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt


# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()

# load image:
im = Image.open('mug.png')
im_torch = toTensor(im).unsqueeze(0) # 1 x 3 x 256 x 256
im_torch = im_torch.to('cuda:0' if torch.cuda.is_available() else 'cpu')

# create style augmentor:
augmentor = StyleAugmentor()

# randomize style:
im_restyled = augmentor(im_torch)

# display:
plt.imshow(toPIL(im_restyled.squeeze().cpu()))
plt.show()

```

## Cite

Please cite our paper if you use this code in your own work:
```
@inproceedings{jackson2019style,
  title={Style Augmentation: Data Augmentation via Style Randomization},
  author={Jackson, Philip T and Atapour-Abarghouei, Amir and Bonner, Stephen and Breckon, Toby P and Obara, Boguslaw},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={83--92},
  year={2019}
}

```
