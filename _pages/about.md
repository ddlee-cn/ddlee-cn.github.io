---
layout: about
title: About
permalink: /
subtitle: Research Scientist @ <a href='https://research.sony/'>Sony Research</a>

profile:
  align: right
  image: portrait.jpg
  image_circular: true # crops the image to make it circular
  more_info: >

selected_papers: true # includes a list of papers marked as "selected={true}"
social: false # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts


display_categories: [Computational Photography, Rendering & Generative AI, Streaming & Display, Intelligent Sensing]
---

- Hi there, I am LI, Jiacheng (lee, jya-cherng), a Research Scientist with [Sony Research](https://research.sony/), based at [Tokyo](https://www.gotokyo.org/en/index.html), Japan.
- I received the Ph.D. degree from [University of Science and Technology of China (USTC)](http://en.ustc.edu.cn/), supervised by [Prof. Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/). During my times in [VIDAR Lab](https://vidar-ustc.github.io), I also worked closely with [Prof. Dong Liu](https://faculty.ustc.edu.cn/dongeliu/), under a bigger team led by [Prof. Feng Wu](https://scholar.google.com/citations?user=5bInRDEAAAAJ). Previously, I received the Bachelor degree from [Tongji University (Tongji)](https://en.tongji.edu.cn/p/#/).
- My core research interest lies in advancing the acquisition and utilization of visual data through innovative algorithms. From the moment of capture, I seek to elevate clarity, smoothness, and fidelity—leveraging the powers of [Computational Photography]({{ site.baseurl }}/computational-photography), [Rendering, and Generative AI]({{ site.baseurl }}/rendering-&-generative-ai) to push the boundaries of perception. Beyond capture, I also explore recreating scenes with breathtaking realism and efficiency on [Streaming and Display]({{ site.baseurl }}/streaming-&-display) platforms, and revealing the meanings hidden within pixels through [Intelligent Sensing]({{ site.baseurl }}/intelligent-sensing). 



> 呦呦鹿鳴， 食野之苹。 我有嘉賓， 鼓瑟吹笙。——《詩經·小雅·鹿鳴》


---

## Research Goal

My main research goal is to build **the next generation of visual computing systems** that not only capture the richness of visual experiences with unprecedented depth and nuance, but also distribute and reproduce these experiences seamlessly and universally, making them accessible across space and time. The pursuit of this vision is anchored by three indispensable principles: [faithfulness]({{ site.baseurl }}/publications/#faithfulnes), [efficiency]({{ site.baseurl }}/publications/#efficiency), and [creativity]({{ site.baseurl }}/publications/#creativity).

---

## Research Area

  <!-- Image-based category links -->
  <div class="category-links" style="margin-bottom: 2rem; text-align: center;">
    {% for category in page.display_categories %}
      <a href="{{ category | downcase | replace: ' ', '-' }}" style="display: inline-block; margin-right: 1rem; text-align: center;">
        <img src="{{ site.baseurl }}/assets/img/research/{{ category | downcase | replace: ' ', '-' }}.png" alt="{{ category }}" style="width: 400px; height: 400px; object-fit: cover; border-radius: 8px; display: block; margin: 0 auto;">
        <div style="margin-top: 0.5rem; margin-bottom: 0.5rem; font-weight: bold;">{{ category }}</div>
      </a>
    {% endfor %}
  </div>


---


<!-- Selected papers -->
{% if page.selected_papers %}
  <h2>
    <a href="{{ '/publications/' | relative_url }}" style="color: inherit">Selected Publications</a>
  </h2>
  {% include selected_papers.liquid %}
{% endif %}

---


## Services


Reviewer: T-PAMI, T-IP, T-CSVT, CVPR, ICCV, ECCV, ICLR, NeurIPS, etc.

Organizer: Advances in Image Manipulation workshop (Real-World Raw Denoising Challenge), ICCV 2025