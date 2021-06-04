
# SemiOrthogonal
###### _Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation_

This is an **unofficial** re-implementation of the paper *Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation* available on [arxiv](http://arxiv.org/abs/2105.14737). 

### Features

The key features of this implementation are: 

- Constant memory footprint - training on more images does not result in more memory required
- Resumable learning - the training step can be stopped and then resumed with inference in-between
- Limited dependencies - apart from PyTorch, Torchvision and Numpy 

### Installation

```
git clone https://github.com/Pangoraw/SemiOrthogonal.git
```

### Getting started

#### Training

You can choose a backbone model between `resnet18` and `wide_resnet50` but note that just the covariance matrix should take at least 100GB of memory with the `wide_resnet50` and only around 6 for `resnet18`.

```python
from torch.utils.data import DataLoader
from semi_orthogonal import SemiOrthogonal

# i) Initialize
semi_ortho = SemiOrthogonal(num_embeddings=100, device="cpu", backbone="resnet18") 

# ii) Create a dataloader producing image tensors
dataloader = DataLoader(...)

# iii) Consume the data to learn the normal distribution
# Use semi_ortho.train(...)
semi_ortho.train(dataloader)

# Or PaDiM.train_one_batch(...)
for imgs in dataloader:
	semi_ortho.train_one_batch(imgs)
semi_ortho.finalize_training() # compute the approx of C^-1
```
#### Testing

With the same `SemiOrthogonal` instance as in the [Training](#training) section:

```python
for new_imgs in test_dataloader:
	distances = semi_ortho.predict(new_imgs) 
	# distances is a (n * c) matrix of the mahalanobis distances
	# Compute metrics...
```

### Acknowledgements

This implementation was built on the work of:

- [The original *Semi Orthogonal* paper](http://arxiv.org/abs/2105.14737)
- [My re-implementation of PaDiM](https://github.com/Pangoraw/PaDiM)