# FRESCO - Official PyTorch Implementation


**FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
in CVPR 2024 <br>
[**Project Page**]() | [**Paper**]() | [**Supplementary Video**]() | [**Input Data and Video Results**]() <br>

**Abstract:** *The remarkable efficacy of text-to-image diffusion models has motivated extensive exploration of their potential application in video domains.
Zero-shot methods seek to extend image diffusion models to videos without necessitating model training.
Recent methods mainly focus on incorporating inter-frame correspondence into attention mechanisms. However, the soft constraint imposed on determining where to attend to valid features can sometimes be insufficient, resulting in temporal inconsistency.
In this paper, we introduce FRESCO, intra-frame correspondence alongside inter-frame correspondence to establish a more robust spatial-temporal constraint. This enhancement ensures a more consistent transformation of semantically similar content across frames. Beyond mere attention guidance, our approach involves an explicit update of features to achieve high spatial-temporal consistency with the input video, significantly improving the visual coherence of the resulting translated videos.
Extensive experiments demonstrate the effectiveness of our proposed framework in producing high-quality, coherent videos, marking a notable improvement over existing zero-shot methods.*

**Features**:<br>
- **Temporal consistency**: use robust intra-and inter-frame constraint with better consistency and coverage than optical flow alone.
- **Zero-shot**: no training or fine-tuning required.
- **Flexibility**: compatible with off-the-shelf models (e.g., [ControlNet](https://github.com/lllyasviel/ControlNet), [LoRA](https://civitai.com/)) for customized translation.


