// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "dropdown-computational-photography",
              title: "Computational Photography",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/computational-photography/";
              },
            },{id: "dropdown-rendering-amp-genenerative-ai",
              title: "Rendering &amp; Genenerative AI",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/rendering-&-generative-ai/";
              },
            },{id: "dropdown-streaming-amp-display",
              title: "Streaming &amp; Display",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/streaming-&-display/";
              },
            },{id: "dropdown-intelligent-sensing",
              title: "Intelligent Sensing",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/intelligent-sensing/";
              },
            },{id: "post-nvidia-dlss-4",
        
          title: "NVIDIA DLSS 4",
        
        description: "An overview of the latest advancements in neural rendering with DLSS 4, featuring multi-frame generation, transformer-based ray reconstruction and super-resolution, and relex frame warp.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/DLSS4/";
          
        },
      },{id: "post-amd-fidelityfx-super-resolution",
        
          title: "AMD FidelityFX Super Resolution",
        
        description: "An overview of the latest advancements in neural rendering with AMD FSR 3.x.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/AMD-FSR/";
          
        },
      },{id: "post-the-real-time-rendering-pipeline",
        
          title: "The Real-Time Rendering Pipeline",
        
        description: "An introduction to the fundamental stages of the real-time rendering pipeline, highlighting how graphics engines like Unreal Engine and Unity utilize GPUs for efficient scene visualization.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/CG-Pipeline/";
          
        },
      },{id: "post-the-evolution-of-google-camera",
        
          title: "The Evolution of Google Camera",
        
        description: "A deep dive into the evolution of Google Camera&#39;s algorithms and features, exploring key milestones like HDR+, Night Sight, and Super Res Zoom.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Google-Camera/";
          
        },
      },{id: "post-image-and-video-compression",
        
          title: "Image and Video Compression",
        
        description: "An note of the fundamentals of image and video compression techniques.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/Image-Video-Codec/";
          
        },
      },{id: "post-image-signal-processing-isp-pipeline-and-3a-algorithms",
        
          title: "Image Signal Processing (ISP) Pipeline and 3A Algorithms",
        
        description: "A Comprehensive Guide to Understanding Image Signal Processing and Automatic Exposure, Focus, and White Balance Adjustment",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/ISP/";
          
        },
      },{id: "post-restoration-amp-generation-a-correspondence-perspective",
        
          title: "Restoration &amp; Generation: A Correspondence Perspective",
        
        description: "Slides for paper reading topic from a correspondence perspective, focusing on restoration and generation tasks in computer vision.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/Correspondence-Paper/";
          
        },
      },{id: "post-anchor-free-object-detection",
        
          title: "Anchor-Free Object Detection",
        
        description: "An overview and summary of anchor-free series detection works.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2020/Anchor-Free-Object-Detection/";
          
        },
      },{id: "post-deep-generative-models",
        
          title: "Deep Generative Models",
        
        description: "A brief introduction to deep generative models, including variational autoencoders (VAEs), generative adversarial networks (GANs), and other related topics.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2019/Generative-Models/";
          
        },
      },{id: "post-deep-learning-in-scientific-research-my-best-practices",
        
          title: "Deep Learning in Scientific Research: My Best Practices",
        
        description: "A guide to implementing deep learning models for scientific research, with a focus on reproducibility, data management, and model evaluation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2019/Best-Practice/";
          
        },
      },{id: "post-deep-learning-for-object-detection-fundamentals",
        
          title: "Deep Learning for Object Detection: Fundamentals",
        
        description: "A comprehensive guide to deep learning for object detection, covering the fundamentals of two-stage and one-stage methods.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2018/Object-Detection/";
          
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6A%63%6C%65%65@%6D%61%69%6C.%75%73%74%63.%65%64%75.%63%6E", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/ddlee-cn", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/ddleecc", "_blank");
        },
      },{
        id: 'social-dblp',
        title: 'DBLP',
        section: 'Socials',
        handler: () => {
          window.open("https://dblp.org/pid/18/5576-4.html", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=7cC0YysAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
