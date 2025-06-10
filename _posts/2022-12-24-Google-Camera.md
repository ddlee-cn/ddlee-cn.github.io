---
layout: post
title: 'The Evolution of Google Camera'
date: 2022-12-24
tags: all computational-photography
description: A deep dive into the evolution of Google Camera's algorithms and features, exploring key milestones like HDR+, Night Sight, and Super Res Zoom.
toc:
  beginning: true
---

## **I\. Introduction**

Google's approach to mobile photography has been consistently characterized by a software-first philosophy, leveraging computational photography to transcend the physical limitations of small smartphone sensors and optics. The Google Camera application, particularly on Pixel devices, stands as a testament to this philosophy, repeatedly demonstrating that algorithmic innovation can drive significant advancements in image quality and user-facing features. 

## **II\. HDR+ and Live HDR+**

### HDR+: Burst Photography

The foundation of Google Camera's image quality prowess was established with High Dynamic Range Plus (HDR+)[^HDR]. Introduced initially on Nexus devices and significantly enhanced on Pixel phones, HDR+ tackled the fundamental challenge of capturing scenes with a wide disparity between the darkest shadows and brightest highlights – a common scenario where small mobile sensors typically struggle.

![](https://i.imgur.com/XEiRljo.jpeg)


The core algorithmic principle of HDR+ is **burst photography**[^HDRPaper]. Instead of a single exposure, the camera captures a rapid sequence of deliberately underexposed frames. Underexposing protects highlight detail that would otherwise be clipped. These short-exposure frames also minimize motion blur. The captured burst, typically consisting of 2 to 15 raw images depending on conditions, then undergoes a sophisticated alignment and merging process. Alignment algorithms compensate for minor hand shake and subtle scene movements between frames. The aligned frames are then merged into an intermediate, high bit-depth computational raw image. This merging process effectively averages out noise, particularly read noise and shot noise, which are significant in underexposed shots from small sensors. The result is an image with significantly reduced noise and increased dynamic range compared to any single frame. Finally, advanced tone mapping algorithms are applied to render the high dynamic range data into a visually pleasing image that preserves detail in both shadows and highlights for standard displays.

![](https://i.imgur.com/ikoNItM.jpeg)

![](https://i.imgur.com/uLCF7m5.jpeg)


As detailed in the release of the HDR+ Burst Photography Dataset[^HDRDataset], key improvements included transitioning to processing **raw images** directly from the sensor, which provided more data for the pipeline and improved image quality. Another crucial advancement was the elimination of **shutter lag**, ensuring the captured photo corresponded to the exact moment the shutter button was pressed. This was often achieved by utilizing frames already buffered by the camera system (Zero Shutter Lag \- ZSL). Processing times and power consumption were also optimized through implementation on specialized hardware accelerators like the Qualcomm Hexagon DSP and, later, Google's custom-designed Pixel Visual Core[^VisualCore]. 


![](https://i.imgur.com/mdtRThi.jpeg)


The introduction of **HDR+ with Bracketing**[^HDRBracket] addressed a limitation of the original HDR+ system: noisy shadows in very high dynamic range scenes due to all frames being underexposed.1 HDR+ with Bracketing strategically incorporates one or more longer exposure frames into the burst. For the default camera mode, this typically means capturing an additional long exposure frame *after* the shutter press (since ZSL uses pre-shutter frames). In Night Sight, multiple long exposures can be captured. The merging algorithm was updated to handle these bracketed exposures, choosing a short frame as the reference to avoid motion blur and clipped highlights from the long exposure. A sophisticated spatial merge algorithm, similar to that used in Super Res Zoom, performs deghosting to prevent artifacts from scene motion between frames of different exposures—a non-trivial task given differing noise characteristics and motion blur. This evolution resulted in improved shadow detail, more natural colors, better texture, and reduced noise, with the merging process also becoming 40% faster. Users of computational RAW also benefited from these enhancements, as the merging happens early in the pipeline using RAW data.

![](https://i.imgur.com/abxgoqR.jpeg)



### Live HDR+: From Bilateral Filtering to HDRNet

![](https://i.imgur.com/mDKmCqS.jpeg)



The Live HDR+[^LiveHDR] provides a **real-time preview** of the final result of HDR+, making HDR imaging more predictable. Thus, it is a fast approximation for multi-frame and bracket exposure process of HDR+ process, producing a HDR+ look at real-time for preview. They divide the input image into “tiles” of size roughly equal to the red patch in the figure below, and approximate HDR+ using a curve for each tile. Since these curves vary gradually, blending between curves is a good way to approximate the optimal curve at any pixel. The behind algorithm is the famous HDRNet[^HDRNet], which originates from the seminal **bilateral filter**.

![](https://i.imgur.com/AadSqTr.jpeg)


A bilateral filter is an image processing technique that smooths images while preserving sharp edges. Unlike traditional blurring filters that can soften important details, the bilateral filter selectively averages pixels, effectively reducing noise in flat areas without affecting the clarity of boundaries. For each pixel, it considers two factors:
- Spatial Distance: How far away the neighboring pixels are.
- Intensity Difference: How different the brightness or color of the neighboring pixels is.

A neighboring pixel is only given a high weight in the average if it is both physically close and has a similar color. This prevents pixels on one side of an edge from being averaged with pixels on the other side, thus keeping the edge sharp.


A naive implementation of the bilateral filter calculates the value of each output pixel by iterating over all of its neighbors within a specified spatial radius. For each neighbor, it computes a weight based on both spatial distance and intensity difference, then calculates a weighted average. This process is repeated for every pixel in the image, leading to a high computational complexity, especially with large filter kernels (i.e., a large spatial sigma).

![](https://i.imgur.com/2bZFkv9.jpeg)


The **bilateral grid**[^BiGrid2] accelerates this process by performing the filtering in a much smaller, lower-dimensional space. Instead of operating directly on the 2D image, it uses a 3D data structure that represents the image's two spatial dimensions (x, y) and its range dimension (intensity or color).



The first step is to "splat" the information from the original, full-resolution image onto the smaller, downsampled bilateral grid. This is analogous to creating a 3D histogram. For each pixel in the input image, its value is distributed among the nearest vertices in the 3D grid.

![](https://i.imgur.com/NkdvU8g.jpeg)

The second step is to apply a fast, simple 3D Gaussian blur directly on the bilateral grid. This is where the edge-preserving smoothing happens. In the grid, pixels that were close in both space and intensity in the original image are now close to each other in the 3D grid. The blur, therefore, averages their values together. Conversely, pixels that were on opposite sides of an edge (spatially close but with a large intensity difference) are far apart along the grid's intensity dimension and are not blurred together.

![](https://i.imgur.com/hZq2Qw3.jpeg)


The final step is to "slice" the blurred grid to produce the final, filtered output image. This step uses the original image as a "guidance map" to read the smoothed values back from the grid.

For each pixel in the original image at coordinates (x, y) with intensity z, we find its corresponding position in the now-blurred 3D grid. Since the original pixel's coordinates and intensity will not align perfectly with the grid's discrete vertices, we perform a tri-linear interpolation between the surrounding blurred grid cells. This interpolation retrieves the final, smoothed pixel value.

The act of looking up and interpolating a value from the grid for every pixel of the input image is what is referred to as "slicing."[^BiGrid]  It effectively creates a 2D "slice" of the 3D grid to form the output image. The final value is obtained by dividing the interpolated intensity sum by the interpolated weight sum from the grid.

The other origin concept of HDRNet comes from **Joint Bilateral Upsampling**, a combination of principles of both bilateral filtering and guided filtering.


![](https://i.imgur.com/EcFttaO.jpeg)

Instead of using a non-linear weighting scheme, the **guided filter**[^Guided] is based on a local linear model. It assumes that the filtered output image can be expressed as a linear transformation of a guidance image within any local window. This guidance image can be the input image itself or a different image


![](https://i.imgur.com/nIwPAVB.jpeg)

The primary purpose of Joint Bilateral Upsampling[^BiGuided] is to upsample a low-resolution image using a corresponding high-resolution image as a guide. It adapts the bilateral filter by decoupling the two kernels. When filtering the low-resolution input image:
- The Spatial Weight is calculated from the pixel coordinates in the high-resolution grid.
- The Range (Intensity) Weight is calculated using the pixel intensity values from the high-resolution guidance image.

![](https://i.imgur.com/7F2cDx7.jpeg)

HDRNet's architecture is fundamentally a two-stream design that mirrors the bilateral grid's logic:

- **A Low-Resolution "Processing" Path (The Grid)**: The input image is first downsampled significantly. This low-resolution preview is fed into a deep but lightweight CNN. This network does the heavy lifting, analyzing the image content and learning the desired enhancement (e.g., tone mapping, color correction, etc.). The output of this network is not an image, but a small 3D grid of affine transformation matrices (e.g., 16x16x8). Each 3x4 matrix in this grid represents the ideal color transformation for a specific spatial location and intensity level. This low-resolution grid of learned transformations is the bilateral grid. It's where the expensive computation happens efficiently.
- **A Full-Resolution "Guidance" Path (The Input Image)**: The original, full-resolution input image is kept aside and used as the "guidance map." It provides the crucial high-frequency edge information that must be preserved.


![](https://i.imgur.com/nZi4kXP.jpeg)

The magic of HDRNet lies in its custom "slicing" layer, which is a direct implementation of the joint bilateral upsampling principle. This layer's job is to apply the learned, low-resolution transformations to the full-resolution image without introducing artifacts like halos or blurred edges.

- Lookup: For every pixel in the full-resolution input image, the slicing layer performs a lookup into the low-resolution grid of affine matrices.
- Guidance: The lookup coordinates are determined by the pixel's properties:
    - Its spatial (x, y) position determines where to look in the grid's spatial dimensions.
    - Its intensity (brightness) value determines where to look along the grid's depth (intensity) dimension.
- Interpolation (Upsampling): Since the pixel's coordinates and intensity won't perfectly align with the grid's discrete points, the slicing layer performs a trilinear interpolation between the neighboring affine transformation matrices in the grid. This step effectively "upsamples" the learned transformations, creating a unique, custom affine matrix for every single pixel in the high-resolution image.
- Application: The newly interpolated, full-resolution affine matrix is then applied to the original pixel's color value to produce the final, enhanced output pixel.

![](https://i.imgur.com/aRXWy1F.jpeg)

In essence, HDRNet’s brilliance is in this combination based on a **10+ year** efforts by the academic community:

- It leverages the bilateral grid as a computational framework to perform complex, expensive learning tasks in a small, low-resolution space, which is the key to its real-time speed.
- It replaces the grid's simple blur with a powerful CNN that can learn any stylistic enhancement from data.
- It uses the principle of joint bilateral upsampling in its "slicing" layer to apply these learned, low-resolution enhancements back to the high-resolution image. The original image guides this upsampling process, ensuring that the final result has sharp, clean edges, perfectly preserving the structural integrity of the original while applying a sophisticated new look.


## **III\. Low Light: Night Sight and Astrophotography**

Building upon the multi-frame merging principles of HDR+, Google Camera introduced Night Sight[^NightSight], revolutionizing low-light mobile photography without requiring flash or a tripod. Night Sight aimed to solve the inherent challenges of low-light imaging: insufficient photons leading to noise, and long exposures leading to motion blur.

### Night Sight

![](https://i.imgur.com/L1pWIpq.jpeg)


The core of Night Sight involves capturing significantly more light than a standard shot by using longer effective exposure times, achieved by merging a burst of frames.

![](https://i.imgur.com/FpLvFtY.jpeg)


Key algorithmic components are:

* **Motion Metering and Adaptive Exposure Strategy:** Before capture, Night Sight measures natural hand shake and scene motion with the combination of motion estimation based on adjacement frames and angular rate measurements from the gyroscope[^NightSightPaper]. If the phone is stable and the scene is still, it uses fewer, longer exposures (up to 1 second per frame if on a tripod, or up to 333ms handheld with minimal motion). If motion is detected, it uses more, shorter exposures (e.g., 15 frames of 1/15s or less) to minimize motion blur. This adaptive strategy is crucial for balancing noise reduction (favoring longer exposures) and sharpness (favoring shorter exposures). 
![](https://i.imgur.com/Hq7vZvc.jpeg)

* **Multi-Frame Merging and Denoising:** The captured burst of dark but sharp frames is carefully aligned and merged. On Pixel 1 and 2, this utilized a modified HDR+ merging algorithm, retuned for very noisy scenes. Pixel 3 leveraged the Super Res Zoom merging algorithm, which also excels at noise reduction through averaging. This process significantly improves the signal-to-noise ratio (SNR).  
![](https://i.imgur.com/i3tzKMP.jpeg)

* **Learning-Based Auto White Balance (AWB):** Traditional AWB often fails in very low light. Night Sight introduced a learning-based AWB algorithm, based on FCCC[^FCCC], trained to recognize and correct color casts, ensuring natural color rendition even in challenging mixed lighting. This model was trained by manually correcting the white balance of numerous low-light scenes, with a newly introduced error metric for more accurate and balanced target.



### Astrophotography

![](https://i.imgur.com/6sfg2VF.jpeg)


* **Extended Multi-Frame Exposures:** To capture enough light from faint celestial objects, total exposure is split into a sequence of frames, each with an exposure time short enough (e.g., up to 16 seconds per frame) to render stars as points rather than trails caused by Earth's rotation.
* **Advanced Noise Reduction:**  
  * **Dark Current and Hot Pixel Correction:** Long exposures exacerbate sensor artifacts like dark current (spurious signal even with no light) and hot/warm pixels (pixels that incorrectly report high values). These are identified by comparing neighboring pixel values within a frame and across the sequence, and outliers are concealed by averaging neighbors[^Astrophotography].  
  * **Sky-Specific Denoising:** The algorithm recognizes that noise characteristics can differ between the sky and foreground.  
* **Sky Segmentation and Optimization:** An on-device Convolutional Neural Network (CNN), trained on over 100,000 manually labeled images, identifies sky regions in the photograph. This allows for selective processing, such as targeted contrast enhancement[^Sky] or darkening of the sky to counteract the tendency of low-light amplification to make the night sky appear unnaturally bright. This segmentation is crucial for realistic rendering.  



## IV\. Better Detail: Super Res Zoom

Smartphones traditionally struggled with zoom, as physical space constraints limit the inclusion of complex optical zoom lens systems found in DSLR cameras. Digital zoom, which typically involves cropping and upscaling a single image, results in significant loss of detail and often introduces artifacts. Google addressed this with Super Res Zoom[^SuperRes], a computational approach to achieve optical-like zoom quality without traditional optical zoom hardware (for modest zoom factors).

![](https://i.imgur.com/iA0yZbQ.jpeg)


The core algorithmic principle of Super Res Zoom is **multi-frame super-resolution**[^SuperRes]. Instead of relying on a single frame, it leverages the burst of frames captured by HDR+. The key insight is that natural hand tremor, even when imperceptible, causes slight shifts in the camera's viewpoint between successive frames in a burst. When combined with Optical Image Stabilization (OIS) that can actively introduce tiny, controlled sub-pixel shifts, each frame captures a slightly different sampling of the scene. By aligning these multiple, slightly offset low-resolution frames and merging them onto a higher-resolution grid, Super Res Zoom can reconstruct details that would be lost in any single frame.

![](https://i.imgur.com/8oGyqVz.jpeg)



This multi-frame approach is fundamentally different from single-frame upscaling techniques like RAISR (Rapid and Accurate Image Super-Resolution)[^RAISR], which Google also developed and uses for enhancing visual quality. While RAISR can improve the appearance of an already captured image, the primary resolution gain in Super Res Zoom (especially for modest zoom factors like 2-3x) comes from the multi-frame merging process itself. 

![](https://i.imgur.com/TIqaiJr.jpeg)


Implementing multi-frame super-resolution on a handheld device presents significant challenges:

* **Noise:** Single frames in a burst, especially if underexposed for HDR+, can be noisy. The algorithm must be robust to this noise and aim to produce a less noisy, higher-resolution result  
* **Complex Scene Motion:** Objects in the scene (leaves, water, people) can move independently of camera motion. Reliable alignment and merging in the presence of such complex, sometimes non-rigid, motion is difficult. The algorithm needs to work even with imperfect motion estimation and incorporate deghosting mechanisms.  
* **Irregular Data Spread:** Due to random hand motion and scene motion, the sampling of the high-resolution grid can be irregular – dense in some areas, sparse in others. This makes the interpolation problem complex.

![](https://i.imgur.com/0THZIPp.jpeg)


Super Res Zoom addresses these by integrating sophisticated alignment and merging algorithms that are aware of noise and can handle local misalignments to prevent ghosting, similar to the advanced merging techniques developed for HDR+ with Bracketing. The system intelligently selects and weights information from different frames to reconstruct the final zoomed image.

![](https://i.imgur.com/SA8u8cX.jpeg)

Like Live HDR+, the merging and interpolation algorithm of Super Res Zoom comes from a **10+ year** idea of kernel regression for image processing[^SKR]. Its primary contribution is to connect the statistical method of kernel regression to various image processing tasks. Kernel regression is a non-parametric technique used to estimate a function or value at a specific point by calculating a weighted average of its neighbors. The core idea is simple: closer points should have more influence. This influence is defined by a "kernel," which is a weighting function that decreases with distance.

![](https://i.imgur.com/LcOspTY.jpeg)

The framework's power is significantly enhanced by the use of adaptive kernels. Instead of using a fixed, symmetric kernel (like a simple circle or square), the method analyzes the local image structure to "steer" the kernel. This means the kernel's shape, orientation, and size are adapted on-the-fly:

- Near an edge, the kernel elongates and orients itself to lie parallel to the edge, thereby averaging pixels along the edge but not across it.
- In flat, textureless regions, the kernel remains more uniform and circular.

This data-adaptive approach allows for superior preservation of sharp details and textures compared to methods using fixed kernels.

In my works, [LeRF]({{ site.baseurl }}/publications/#learning steerable resampling) and [LeRF++]({{ site.baseurl }}/publications/#lerf:), we push this direction a step foward by introducing a **learning-based parametric CNN** for the prediction of kernel shapes, which is further accelerated by [**LUTs**]({{ site.baseurl }}/streaming-&-display/) to achieve adaptive and efficient interpolation.


## **V\. Depth, Portraits, and Semantic Understanding**

Google Camera's Portrait Mode, which simulates the shallow depth-of-field effect (bokeh) typically associated with DSLR cameras using wide-aperture lenses, has undergone significant algorithmic evolution, heavily relying on advancements in depth estimation and machine learning for semantic understanding.

### Depth Estimation, Segmentation, and Portrait Mode

![](https://i.imgur.com/C4Yfzp2.jpeg)


* **Pixel 2 (2017): Single Camera Depth via Dual-Pixel Auto-Focus (PDAF) and ML Segmentation:** The Pixel 2, despite having a single rear camera, introduced Portrait Mode by ingeniously using its dual-pixel auto-focus (PDAF) sensor. Each pixel on a PDAF sensor is split into two photodiodes, capturing two slightly different perspectives of the scene (a very short baseline stereo pair). The parallax (apparent shift) between these two sub-pixel views can be used to compute a depth map. This depth map, combined with a machine learning model trained to segment people from the background, allowed for the initial bokeh effect. For the front-facing camera, which initially lacked PDAF-based depth, segmentation was achieved purely through an ML model[^Portrait]. 
* **Pixel 3 (2018): ML-Enhanced Depth from PDAF:** The Pixel 3 improved upon this by using machine learning to directly predict depth from the PDAF pixel data[^Portrait2]. Instead of a traditional stereo algorithm, a Convolutional Neural Network (CNN) trained in TensorFlow took the two PDAF views as input and learned to predict a higher quality depth map[^SynDepth]. This ML approach was better at handling errors common with traditional stereo, such as those around repeating patterns or textureless surfaces, and could leverage semantic cues (e.g., recognizing a face to infer its distance). 


### More applications: Alpha Matting, Relighting

![](https://i.imgur.com/Q2w9LuP.jpeg)



While depth maps are crucial for determining the *amount* of blur, accurately separating the subject from the background, especially around fine details like hair, requires more than just depth. This is where semantic segmentation and alpha matting come into play.

* Early Semantic Segmentation: From its inception, Portrait Mode used ML-based semantic segmentation to identify people in the scene, creating a mask to distinguish foreground from background. This mask was then refined by the depth map.  
* Pixel 6 (2021): High-Resolution ML Alpha Matting[^Matting]: A major leap in subject separation for selfies came with the Pixel 6, which introduced a new ML-based approach for Portrait Matting to estimate a high-resolution and accurate alpha matte. An alpha matte specifies the opacity of each pixel, allowing for very fine-grained foreground-background compositing.  
  The Portrait Matting model is a fully convolutional neural network with a MobileNetV3 backbone and encoder-decoder blocks. It takes the color image and an initial coarse alpha matte (from a low-resolution person segmenter) as input. It first predicts a refined low-resolution matte, then a shallow encoder-decoder refines this to a high-resolution matte, focusing on structural features and fine details like individual hair strands. 
  This model was trained using a sophisticated dataset:  
  1. **Light Stage Data:** High-quality ground truth alpha mattes were generated using Google's Light Stage, a volumetric capture system with 331 LED lights and high-resolution cameras/depth sensors. This allowed for "ratio matting" (silhouetting against an illuminated background) to get precise mattes. These subjects were then relit and composited onto various backgrounds.  
  2. **In-the-Wild Portraits:** To improve generalization, pseudo-ground truth mattes were generated for a large dataset of in-the-wild Pixel photos using an ensemble of existing matting models and test-time augmentation. This high-quality alpha matte allows for much more accurate bokeh rendering around complex boundaries like hair, significantly reducing artifacts where the background might have remained sharp or the foreground was incorrectly blurred.



Further, they design a novel system[^Relight] for portrait relighting and background replacement, which maintains high-frequency boundary details and accurately synthesizes the subject’s appearance as lit by novel illumination, thereby producing realistic composite images for any desired scene. The key componenets include foreground estimation via alpha matting, relighting, and compositing. 

![](https://i.imgur.com/GFENBBl.jpeg)

The relighting module is divided into three sequential steps. A first Geometry Network estimates per-pixel surface normals from the input foreground. The surface normals and foreground 𝐹 are used to generate the albedo. The target HDR lighting environment is prefiltered using diffuse and specular convolution operations, and then these prefiltered maps are sampled using surface normals or reflection vectors, producing a per-pixel representation of diffuse and specular reflectance for the target illumination (light maps). Finally, a Shading Network produces the final relit foreground.

![](https://i.imgur.com/fbP6oqQ.jpeg)


## **VI. Other Topics**

### Video Stabilization

![](https://i.imgur.com/4ZoCoCj.jpeg)


Fused Video Stabilization is a hybrid approach that combines Optical Image Stabilization (OIS) with Electronic Image Stabilization (EIS) to produce smooth, professional-looking videos.

The core idea is to address common problems like camera shake and motion blur. The system uses the phone's gyroscope and accelerometer to precisely measure and predict the user's intended motion. The OIS hardware compensates for small, high-frequency jitters, while the EIS software handles larger motions and corrects for other distortions like rolling shutter ("jello" effect). By intelligently fusing these two methods, the technology delivers exceptionally stable video that mimics the look of footage shot with professional camera equipment.

### Denoise and Deblur

The denosing takes advantage of self-similarity of patches across the image to denoise with high fidelity. The general principle behind the seminal “non-local” denoising is that noisy pixels can be denoised by averaging pixels with similar local structure. However, these approaches typically incur high computational costs because they require a brute force search for pixels with similar local structure, making them impractical for on-device use. In the “pull-push” approach, the algorithmic complexity is decoupled from the size of filter footprints thanks to effective information propagation across spatial scales.

Instead of tackling severe motion blur, the deblur[^Deblur] function focuses on "mild" blur—the subtle loss of sharpness caused by minor camera shake, small focus errors, or lens optics. The proposed method, Polyblur[^Polyblur], is a two-stage process designed to be fast enough to run in a fraction of a second on mobile devices.

![](https://i.imgur.com/he7qCci.jpeg)



## **VII\. Final Remarks**

The Google Camera team, led by Prof. Marc Levoy[^Marc] and Prof. Peyman Milanfar[^Peymann] later, contributed a lot to the advancement in both academia and massive application in industry of computational photography technology, and had a deep influence to myself in terms of both research taste and ideas. Personally, I want to appreciate their openness.

Look back to the evolution of Google Camera, a clear trend is the **iterative enhancement of core techniques**, including HDR+, Night Sight, and Super Res Zoom. Another key is takeaway the synergy between **software, hardware, and data-driven approach**.


![](https://i.imgur.com/o3S4PcU.jpeg)

The ultimate goal of computatiobal photography goes beyond match the human vision in terms of spatial, temporal, sectral resolution.  From my point of view, future trends include intergration of other sensor modality, e.g., [multi-spectral]({{ site.baseurl }}/publications/#multi-spectral), take advantage of on-device generative AI for creative post-processing, e.g., applying personalized photographic style, and adaptation to novel capture devices, e.g., AR Glasses & Wearables.


**References**


[^HDRBracket]: [HDR+ with Bracketing on Pixel Phones](https://research.google/blog/hdr-with-bracketing-on-pixel-phones/)  

[^HDR]: [HDR+: Low Light and High Dynamic Range photography in the Google Camera App](https://research.google/blog/hdr-low-light-and-high-dynamic-range-photography-in-the-google-camera-app/)

[^HDRDataset]: [Introducing the HDR+ Burst Photography Dataset](https://research.google/blog/introducing-the-hdr-burst-photography-dataset/)  

[^HDRPaper]: [Burst photography for high dynamic range and low-light imaging on mobile cameras](https://dl.acm.org/doi/10.1145/2980179.2980254), in SIGGRAPH 2016

[^VisualCore]: [Pixel Visual Core: image processing and machine learning on Pixel 2](https://blog.google/products/pixel/pixel-visual-core-image-processing-and-machine-learning-pixel-2/)  


[^FCCC]: [Fast Fourier Color Constancy](https://research.google/pubs/fast-fourier-color-constancy/), in CVPR 2017

[^NightSight]: [Night Sight: Seeing in the Dark on Pixel Phones](https://research.google/blog/night-sight-seeing-in-the-dark-on-pixel-phones/)  

[^NightSightPaper]: [Handheld Mobile Photography in Very Low Light](https://dl.acm.org/doi/10.1145/3355089.3356508), in TOG 2019

[^Sky]: [Sky Optimization: Semantically aware image processing of skies in low-light photography](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Liba_Sky_Optimization_Semantically_Aware_Image_Processing_of_Skies_in_Low-Light_CVPRW_2020_paper.pdf), in CVPRW 2020


[^Astrophotography]: [Astrophotography with Night Sight on Pixel Phones](https://research.google/blog/astrophotography-with-night-sight-on-pixel-phones/)  


[^RAISR]: [Rapid and Accurate Image Super-Resolution](https://ieeexplore.ieee.org/iel7/6745852/6960042/07744595.pdf), in TCI 2017

[^SuperRes]: [See Better and Further with Super Res Zoom on the Pixel 3](https://research.google/blog/see-better-and-further-with-super-res-zoom-on-the-pixel-3/)  

[^SuperResPaper]: [Handheld multi-frame super-resolution](https://sites.google.com/view/handheld-super-res/), in SIGGRAPH 2019

[^SynDepth]: [Synthetic depth-of-field with a single-camera mobile phone](https://dl.acm.org/doi/10.1145/3197517.3201329), in TOG 2018

[^BiGrid]: [A Fast Approximation of the Bilateral Filter using a Signal Processing Approach](https://link.springer.com/chapter/10.1007/11744085_44), in ECCV 2006

[^BiGrid2]: [Real-time Edge-Aware Image Processing with the Bilateral Grid](https://dl.acm.org/doi/10.1145/1276377.1276506), in SIGGRAPH 2007

[^Guided]: [Guided Image Filtering](https://ieeexplore.ieee.org/document/6319316/), in T-PAMI 2013

[^BiGuided]: [Bilateral guided upsampling](https://dl.acm.org/doi/10.1145/2980179.2982423), in SIGGRAPH Asia 2016


[^SKR]: [Kernel Regression for Image Processing and Reconstruction](http://ieeexplore.ieee.org/document/4060955/), in T-IP 2007

[^HDRNet]: [Deep bilateral learning for real-time image enhancement](https://groups.csail.mit.edu/graphics/hdrnet/), in SIGGRPAH 2017

[^LiveHDR]: [Live HDR+ and Dual Exposure Controls on Pixel 4 and 4a](https://research.google/blog/live-hdr-and-dual-exposure-controls-on-pixel-4-and-4a/)


[^Portrait]: [Portrait mode on the Pixel 2 and Pixel 2 XL smartphones](https://research.google/blog/portrait-mode-on-the-pixel-2-and-pixel-2-xl-smartphones/)

[^Portrait2]: [Learning to Predict Depth on the Pixel 3 Phones](https://research.google/blog/learning-to-predict-depth-on-the-pixel-3-phones/)

[^Relight]: [Learning to Relight Portraits for Background Replacement](https://augmentedperception.github.io/total_relighting/), in SIGGRPAH 2021


[^Matting]: [Accurate Alpha Matting for Portrait Mode Selfies on Pixel 6](https://research.google/blog/accurate-alpha-matting-for-portrait-mode-selfies-on-pixel-6/)

[^Marc]: [Marc Levoy](https://graphics.stanford.edu/~levoy/)

[^Peymann]: [Peyman Milanfar](https://sites.google.com/view/milanfarhome/)

[^Stable]: [Fused Video Stabilization on the Pixel 2 and Pixel 2 XL](https://research.google/blog/fused-video-stabilization-on-the-pixel-2-and-pixel-2-xl/)


[^Deblur]: [Take All Your Pictures to the Cleaners, with Google Photos Noise and Blur Reduction](https://research.google/blog/take-all-your-pictures-to-the-cleaners-with-google-photos-noise-and-blur-reduction/)

[^Polyblur]: [Polyblur: Removing Mild Blur by Polynomial Reblurring](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9502555), in T-CI 2017