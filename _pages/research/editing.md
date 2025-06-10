## Toward Creativity: Image Editing


My exploration into generative AI and image editing commenced with a compelling challenge focused on Dunhuang Image Inpainting, part of the e-heritage workshop[^iccv19] at ICCV 2019. The objective was to restore ancient paintings by filling in missing regions using an edge-guided contextual attention mechanism. Our team was honored with the **WINNER**🏆 prize in this challenge.

In the same year, the best paper award at ICCV 2019 for SinGAN[^SinGAN] captured my attention, particularly its novel approach to training Generative Adversarial Networks (GANs) on a single image without requiring paired data. This insight directly motivated my initial research project focused on leveraging single-image **GANs** to empower novel image editing capabilities. We conceptualized this work as ["Semantic Image Analogy"]({{ site.baseurl }}/publications/#semantic%20image%20analogy), a tribute to the foundational "Image Analogy"[^IA] paper.

Subsequently, my research delved deeper into understanding and leveraging spatial correlations within and between images. This led me to revisit inpainting with the [RefMatch]({{ site.baseurl }}/publications/#reference-guided) project, where we utilized reference images to extract fine-grained structural details for high-fidelity completion of missing regions. A distinctive aspect of was its reliance on pre-trained Deep Neural Networks (DNNs) solely as feature extractors, eschewing an explicit learning phase. Instead, pattern recognition was achieved through a **multi-scale nearest neighbor search** approach, which is kind of rebellious at that time when deep learning was dominating the field.

Continuing this exploration of correlations, I initiated the [Contextual Outpainting]({{ site.baseurl }}/publications/#contextual outpainting) project. Here, I investigated the semantic relationships between different parts within an image, employing techniques such as **VAEs** (Variational Autoencoders) and **Contrastive Learning**. This line of inquiry has since been extended to incorporate later advancements like **LoRA Adaptors** and **Stable Diffusion** models. Additionally, I contributed to research on [image retouching](({{ site.baseurl }}/publications/#region-aware%20portrait)) guided by sparse, interactive user instructions, a system that utilized **cross-attention mechanisms** and **MoEs** (Mixture of Experts).

My experience also extends to utilizing [single-step/few-step diffusion models]({{ site.baseurl }}/blog/2023/One-Step-Diffusion/) for image enhancement, with a particular focus on facial images. Moving forward, my interest in this domain is expanding from the manipulation of 2D images to tackling the exciting challenges of generating and editing 3D assets🎨.

**References**

[^iccv19]: [ICCV Workshop on eHeritage](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/), 2019
[^SinGAN]: [Learning a Generative Model from a Single Natural Image](https://tamarott.github.io/SinGAN.htm), in ICCV 2019
[^IA]: [Image Analogies](https://dl.acm.org/doi/10.1145/383259.383295), in SIGGPRAPH 2001