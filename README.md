<div align='center'>
 
# SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://arxiv.org/abs/2310.12508)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue:ICLR 2024](https://img.shields.io/badge/Venue-ICLR%202024%20(Spotlight)-007CFF)](https://iclr.cc/Conferences/2024)

[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Saliency)](https://github.com/OPTML-Group/Unlearn-Saliency)

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/transition_new.gif" alt="Examples" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Example comparison of pre/after unlearning by SalUn. <br/> (Left) Concept "Nudity"; (Middle) Object "Dog"; (Right) Style "Sketch".</em>
    </td>
  </tr>
</table>
</div>

This is the official code repository for the ICLR 2024 Spotlight paper [SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation](https://arxiv.org/abs/2310.12508).

**WARNING**: This repository contains model outputs that may be offensive in nature.

## Abstract

With evolving data regulations, machine unlearning (MU) has become an important tool for fostering trust and safety in today's AI models. However, existing MU methods focusing on data and/or weight perspectives often suffer limitations in unlearning accuracy, stability, and cross-domain applicability. To address these challenges, we introduce the concept of 'weight saliency' for MU, drawing parallels with input saliency in model explanation. This innovation directs MU's attention toward specific model weights rather than the entire model, improving effectiveness and efficiency. The resultant method that we call *saliency unlearning* (SalUn) narrows the performance gap with 'exact' unlearning (model retraining from scratch after removing the forgetting data points). To the best of our knowledge, SalUn is the first principled MU approach that can effectively erase the influence of forgetting data, classes, or concepts in both image classification and generation tasks. As highlighted below, For example, SalUn yields a stability advantage in high-variance random data forgetting, *e.g.*, with a 0.2% gap compared to exact unlearning on the CIFAR-10 dataset. Moreover, in preventing conditional diffusion models from generating harmful images, SalUn achieves nearly 100% unlearning accuracy, outperforming current state-of-the-art baselines like Erased Stable Diffusion and Forget-Me-Not.


<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser-v2.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 2:</strong> Schematic overview of our proposal on Saliency Unlearning (SalUn).</em>
    </td>
  </tr>
</table>

## Getting Started
SalUn can be applied to different tasks such as image classification and image generation. You can click the link below to access a more detailed installation guide.
* [SalUn for Image Classification](Classification/README.md)
* SalUn for Image Generation
  * [Classifier-free Guidance DDPM](DDPM/README.md)
  * [Stable Diffusion](SD/README.md)

## Examples

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/church.jpg" alt="Church unlearn" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 3:</strong> Class-wise unlearning by SalUn on Stable Diffusion for the "Church" class.</em>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/nudity.jpg" alt="Nudity unlearn" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 4:</strong> Concept unlearning by SalUn on Stable Diffusion for the "Nudity" concept.</em>
    </td>
  </tr>
</table>

## Contributors

* [Chongyu Fan](https://a-f1.github.io/)
* [Jiancheng Liu](https://ljcc0930.github.io/)

## Cite This Work
```
@article{fan2023salun,
  title={SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation},
  author={Fan, Chongyu and Liu, Jiancheng and Zhang, Yihua and Wong, Eric and Wei, Dennis and Liu, Sijia},
  journal={arXiv preprint arXiv:2310.12508},
  year={2023}
}
```
