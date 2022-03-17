# PadInv - High-fidelity GAN Inversion with Padding Space

> **High-fidelity GAN Inversion with Padding Space** <br>
> Qingyan Bai*, Yinghao Xu*, Jiapeng Zhu, Weihao Xia, Yujiu Yang, Yujun Shen <br>
> *arXiv preprint arXiv:*

![image](TODO)

[[Paper]()]
[[Project Page]()]

This paper aims at achieving high-fidelity GAN Inversion and manipulation. We propose to involve the padding space of the generator to complement the latent space with spatial information. Concretely, we replace the constant padding (e.g., usually zeros) used in convolution layers with some instance- aware coefficients. In this way, the inductive bias assumed in the pre- trained model can be appropriately adapted to fit each individual image. Through learning a carefully designed encoder, we manage to improve the inversion quality both qualitatively and quantitatively, outperforming existing alternatives. We then demonstrate that such a space extension barely affects the native GAN manifold, hence we can still reuse the prior knowledge learned by GANs for various downstream applications. Beyond the editing tasks explored in prior arts, our approach allows a more flexible image manipulation, such as the separate control of face contour and facial details, and enables a novel editing manner where users can customize their own manipulations highly efficiently.

## Qualitative Results

Teaser.

![image]()

Inversion Results.

![image]()

Face blending results.

![image]()

Manipulation with one pair of customized images.

![image]()

## Code Coming Soon

## BibTeX

```bibtex
@article{bai2022padinv,
  title   = {High-fidelity GAN Inversion with Padding Space},
  author  = {Bai, Qingyan and Xu, Yinghao and Zhu, Jiapeng and Xia, Weihao and Yang, Yujiu and Shen, Yujun},
  article = {arXiv preprint arXiv:TODO},
  year    = {2022}
}
```
