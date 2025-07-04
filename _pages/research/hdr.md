##  Toward Faithfulness: HDR Display

High Dynamic Range (HDR) imaging and precise color reproduction are fundamental to creating visually faithful and realistic media. My own deep dive into this field was sparked by the impressive "Live HDR+"[^live-hdr] feature on [Google Pixel Phones]({{ site.baseurl }}/blog/2022/Google-Camera/), a powerful application of the seminal HDRNet framework[^HDRNet]. This experience ignited a deeper interest in the entire HDR tech ecosystem, from capture techniques like bracketing exposure and staggered pixels to advanced display standards such as PQ (Perceptual Quantizer), HLG (Hybrid Log-Gamma), and Dolby Vision. ~~This interest also led to an upgrade💰 of my home cinema setup~~.

The comprehensive adoption of HDR across both content capture and display is an inevitable technological progression. Since late 2023, the industry has converged on a powerful solution for distributing HDR images with backward compatibility: the use of HDR gain maps. This supplementary metadata allows a single file to be rendered correctly on both Standard Dynamic Range (SDR) and HDR displays, an approach now championed by industry leaders like Apple[^Apple], Google[^Google], and Adobe[^Adobe]. [2025 Update: This is now part of the emerging ISO 21496 standard[^ISO21496].]

My research directly tackles this transition by developing learning-based tools to enable seamless HDR adoption. I have mentored two key projects in this domain:
- [MLP Embedded Inverse Tone Mapping (ITM)]({{ site.baseurl }}/publications/#mlp embedded): We developed a framework that embeds a lightweight, per-image MLP network as "neural metadata" within a standard SDR file. This allows for high-fidelity Inverse Tone Mapping on HDR screens, effectively restoring the content's original dynamic range.
- [Learning Gain Maps for ITM]({{ site.baseurl }}/publications/#learning gain map): To bring the benefits of HDR to legacy content, this project developed a neural network capable of predicting gain maps for existing SDR images. This approach expands their dynamic range for compelling HDR presentation. As a key contribution, we also curated a new real-world dataset to drive further research and development.

Looking forward, my goal is to push the boundaries of visual faithfulness further. I plan to extend HDR principles into new dimensions, exploring temporal consistency for video and view-dependent effects for truly immersive and realistic visual experiences.


**References**

[^live-hdr]: [Live HDR+ and Dual Exposure Controls on Pixel 4 and 4a](https://research.google/blog/live-hdr-and-dual-exposure-controls-on-pixel-4-and-4a/), Google Research Blog
[^HDRNet]: [Deep bilateral learning for real-time image enhancement](https://groups.csail.mit.edu/graphics/hdrnet/), in SIGGRPAH 2017
[^Google]: [Ultra HDR Image Format](https://developer.android.com/media/platform/hdr-image-format)
[^Apple]: [Applying Apple HDR effect to your photos](https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos)
[^Adobe]: [Gain Map in Adobe Camera Raw](https://helpx.adobe.com/camera-raw/using/gain-map.html)
[^ISO21496]: [ISO 21496: Gain map metadata for image conversion](https://www.iso.org/standard/86775.html)