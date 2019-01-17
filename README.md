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
plt.imshow(toPIL(im_restyled.squeeze()))
plt.show()

```

## Cite

Please cite our paper if you use this code in your own work:
```
@article{jackson2018style,
  author    = {Philip T. Jackson and
               Amir Atapour Abarghouei and
               Stephen Bonner and
               Toby P. Breckon and
               Boguslaw Obara},
  title     = {Style Augmentation: Data Augmentation via Style Randomization},
  year      = {2018},
  url       = {http://arxiv.org/abs/1809.05375}
}
```
