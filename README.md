# FRESCO - Official PyTorch Implementation


**FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
in CVPR 2024 <br>
[**Project Page**]() | [**Paper**]() | [**Supplementary Video**]() | [**Input Data and Video Results**](https://drive.google.com/file/d/12BFx3hp8_jp9m0EmKpw-cus2SABPQx2Q/view?usp=sharing) <br>

**Abstract:** *The remarkable efficacy of text-to-image diffusion models has motivated extensive exploration of their potential application in video domains.
Zero-shot methods seek to extend image diffusion models to videos without necessitating model training.
Recent methods mainly focus on incorporating inter-frame correspondence into attention mechanisms. However, the soft constraint imposed on determining where to attend to valid features can sometimes be insufficient, resulting in temporal inconsistency.
In this paper, we introduce FRESCO, intra-frame correspondence alongside inter-frame correspondence to establish a more robust spatial-temporal constraint. This enhancement ensures a more consistent transformation of semantically similar content across frames. Beyond mere attention guidance, our approach involves an explicit update of features to achieve high spatial-temporal consistency with the input video, significantly improving the visual coherence of the resulting translated videos.
Extensive experiments demonstrate the effectiveness of our proposed framework in producing high-quality, coherent videos, marking a notable improvement over existing zero-shot methods.*

**Features**:<br>
- **Temporal consistency**: use robust intra-and inter-frame constraint with better consistency and coverage than optical flow alone.
    - Compared with our previous work [Rerender-A-Video](https://github.com/williamyang1991/Rerender_A_Video), FRESCO is more robust to large and quick motion.
- **Zero-shot**: no training or fine-tuning required.
- **Flexibility**: compatible with off-the-shelf models (e.g., [ControlNet](https://github.com/lllyasviel/ControlNet), [LoRA](https://civitai.com/)) for customized translation.



https://github.com/williamyang1991/FRESCO/assets/18130694/aad358af-4d27-4f18-b069-89a1abd94d38



## Updates
- [03/2023] Code is released.
- [03/2024] This website is created.

### TODO
- [x] Add webUI.
- [x] Upload paper to arXiv, release related material
- [x] Update readme

## Installation

1. Clone the repository. 

```shell
git clone https://github.com/williamyang1991/FRESCO.git
cd FRESCO
```

2. You can simply set up the environment with pip based on [requirements.txt](https://github.com/williamyang1991/FRESCO/blob/main/requirements.txt)
    - We have tested on torch 2.0.0/2.1.0 and diffusers 0.19.3
    - If you use new versions of diffusers, you need to modify [my_forward()](https://github.com/williamyang1991/FRESCO/blob/fb991262615665de88f7a8f2cc903d9539e1b234/src/diffusion_hacked.py#L496)

3. Run the installation script. The required models will be downloaded in `./model`, `./src/ControlNet/annotator` and `./src/ebsynth/deps/ebsynth/bin`.
    - Requires access to huggingface.co

```shell
python install.py
```

4. You can run the demo with `run_fresco.py`

```shell
python run_fresco.py ./config/config_music.yaml
```

5. For issues with Ebsynth, please refer to [issues](https://github.com/williamyang1991/Rerender_A_Video#issues)


## (1) Inference

### WebUI (recommended)

TBA

### Command Line

We provide a flexible script `run_fresco.py` to run our method.

Set the options via a config file. For example,
```shell
python run_fresco.py ./config/config_music.yaml
```
We provide some examples of the config in `config` directory.
Most options in the config is the same as those in WebUI.
Please check the explanations in the WebUI section.

## (2) Results

### Key frame translation

<table class="center">
<tr>
  <td><img src="https://github.com/williamyang1991/FRESCO/assets/18130694/e8d5776a-37c5-49ae-8ab4-15669df6f572" raw=true></td>
  <td><img src="https://github.com/williamyang1991/FRESCO/assets/18130694/8a792af6-555c-4e82-ac1e-5c2e1ee35fdb" raw=true></td>
  <td><img src="https://github.com/williamyang1991/FRESCO/assets/18130694/10f9a964-85ac-4433-84c5-1611a6c2c434" raw=true></td>
  <td><img src="https://github.com/williamyang1991/FRESCO/assets/18130694/0ec0fbf9-90dd-4d8b-964d-945b5f6687c2" raw=true></td>
</tr>
<tr>
  <td width=26.5% align="center">a red car turns in the winter</td>
  <td width=26.5% align="center">an African American boxer wearing black boxing gloves punches towards the camera, cartoon style</td>
  <td width=26.5% align="center">a cartoon spiderman in black suit, black shoes and white gloves is dancing</td>
  <td width=20.5% align="center">a beautiful woman holding her glasses in CG style</td>
</tr>
</table>


### Full video translation

https://github.com/williamyang1991/FRESCO/assets/18130694/bf8bfb82-5cb7-4b2f-8169-cf8dbf408b54

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang2024fresco,
 title = {FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation},
 author = {Yang, Shuai and Zhou, Yifan and Liu, Ziwei and and Loy, Chen Change},
 booktitle = {CVPR},
 year = {2024},
}
```

## Acknowledgments

The code is mainly developed based on [Rerender-A-Video](https://github.com/williamyang1991/Rerender_A_Video), [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [GMFlow](https://github.com/haofeixu/gmflow) and [Ebsynth](https://github.com/jamriska/ebsynth).


