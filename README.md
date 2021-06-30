## SemiOrthogonal
##### _Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation_
---

![Example with the toothbrush dataset](https://user-images.githubusercontent.com/9824244/121331688-691ef780-c917-11eb-927d-677914acbaf5.png)


This is an **unofficial** re-implementation of the paper *Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation* [1] available on [arxiv](http://arxiv.org/abs/2105.14737). This paper proposes a modification on the PaDiM [2] method, mainly to replace the random dimension selection with an optimized inverse covariance computation using a semi-orthogonal embedding.

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

#### MVTec AD

Here are the metrics compared to the one from the paper (with only one run), with WideResNet50 as a backbone and `k=300`:

|Category|Paper (PRO Score)|This implementation (PRO Score)|
|-|-|-|
|Carpet|.974|.971|
|Grid|.941|.972|
|Leather|.987|.997|
|Tile|.859|.932|
|Wood|.906|.969|
|Bottle|.962|.988|
|Cable|.915|.963|
|Capsule|.952|.967|
|Hazelnut|.970|.985|
|Metal nut|.930|.976|
|Pill|.936|.982|
|Screw|.953|.984|
|Toothbrush|.957|.985|
|Transistor|.929|.969|
|Zipper|.960|.985|
|**Mean**|.942|.975|

To reproduce the results on the MVTec AD dataset, download the files.

```bash
 $ mkdir data

 $ cd data

 $ wget ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz

 $ tar -xvf mvtec_anomaly_detection.tar.xz
```

And run `examples/mvtec.py` for each MVTec category:

```bash
for CATEGORY in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
  echo "Running category $CATEGORY"
  python examples/mvtec.py \
    --data_root data/$CATEGORY/ \
    --backbone wide_resnet50 \
    -k 300
done
```

#### Training

You can choose a backbone model between `resnet18` and `wide_resnet50`, and select the `k` value for the semi-orthogonal matrix size.
For custom image size, you can also pass the image size to the constructor (not square images may not work).

```python
from torch.utils.data import DataLoader
from semi_orthogonal import SemiOrthogonal

# i) Initialize
semi_ortho = SemiOrthogonal(k=100, device="cpu", backbone="resnet18", size=(256,256)) 

# ii) Create a dataloader producing image tensors
dataloader = DataLoader(...)

# iii) Consume the data to learn the normal distribution
# Use semi_ortho.train(...)
semi_ortho.train(dataloader)

# Or SemiOrthogonal.train_one_batch(...)
for imgs in dataloader:
	semi_ortho.train_one_batch(imgs)
semi_ortho.finalize_training() # compute the approx of C^-1
```
#### Testing

With the same `SemiOrthogonal` instance as in the [Training](#training) section:

```python
for new_imgs in test_dataloader:
	distances = semi_ortho.predict(new_imgs)
	# Note: predict only supports one image batches for now ;)
	# distances is a (1, w, h) matrix of the mahalanobis distances
	# Compute metrics or plot the anomaly map...
```

### References

> [1] Kim, J.-H., Kim, D.-H., Yi, S., Lee, T., 2021. Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation. arXiv:2105.14737 [cs].

> [2] Defard, T., Setkov, A., Loesch, A., Audigier, R., 2020. PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization. arXiv:2011.08785 [cs].

### Acknowledgements

This implementation was built on the work of:

- [The original *Semi Orthogonal* paper](http://arxiv.org/abs/2105.14737)
- [taikiinoue45/mvtec-utils](https://github.com/taikiinoue45/mvtec-utils) for the metric evaluation code
- [My re-implementation of PaDiM](https://github.com/Pangoraw/PaDiM)
