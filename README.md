# PadInv - High-fidelity GAN Inversion with Padding Space

> **High-fidelity GAN Inversion with Padding Space** <br>
> Qingyan Bai*, Yinghao Xu*, Jiapeng Zhu, Weihao Xia, Yujiu Yang, Yujun Shen <br>
> *arXiv preprint arXiv:2203.11105*

![image](./docs/assets/framework.png)
**Figure:** Our encoder produces instance-aware coefficients to replace the fixed padding used in the generator. Such a design improves GAN inversion with better spatial details.

[[Paper](https://arxiv.org/abs/2203.11105)]
[[Project Page](https://ezioby.github.io/padinv/)]

In this work, we propose to involve the **padding space** of the generator to complement the native latent space, facilitating high-fidelity GAN inversion. Concretely, we replace the constant padding (*e.g.*, usually zeros) used in convolution layers with some instance-aware coefficients. In this way, the inductive bias assumed in the pre-trained model can be appropriately adapted to fit each individual image. We demonstrate that such a space extension allows a more flexible image manipulation, such as the **separate control** of face contour and facial details, and enables a **novel editing manner** where users can *customize* their own manipulations highly efficiently.

## Qualitative Results

From top to bottom: (a) high-fidelity GAN inversion with spatial details, (b) face blending with contour from one image and details from another, and (c) customized manipulations *with one image pair*.

![image](./docs/assets/teaser.png)

More inversion results.

![image](./docs/assets/inversion.png)

More face blending results.

![image](./docs/assets/face_blending.png)

More customized editing results.

![image](./docs/assets/customized_editing.png)

## Code Coming Soon

## BibTeX

```bibtex
@article{bai2022padinv,
  title   = {High-fidelity GAN Inversion with Padding Space},
  author  = {Bai, Qingyan and Xu, Yinghao and Zhu, Jiapeng and Xia, Weihao and Yang, Yujiu and Shen, Yujun},
  article = {arXiv preprint arXiv:2203.11105},
  year    = {2022}
}
```
