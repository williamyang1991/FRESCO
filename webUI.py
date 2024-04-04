import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"

# uncomment the next line to use huggingface model in China
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import io
import gc
import yaml
import argparse
import torch
import torchvision
import diffusers
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, ControlNetModel
import gradio as gr
from enum import Enum
import imageio.v2 as imageio

from src.utils import *
from src.keyframe_selection import get_keyframe_ind
from src.diffusion_hacked import apply_FRESCO_attn, apply_FRESCO_opt, disable_FRESCO_opt
from src.diffusion_hacked import get_flow_and_interframe_paras, get_intraframe_paras
from src.pipe_FRESCO import inference
from src.free_lunch_utils import apply_freeu

import sys
sys.path.append("./src/ebsynth/deps/gmflow/")
sys.path.append("./src/EGNet/")
sys.path.append("./src/ControlNet/")

from gmflow.gmflow import GMFlow
from model import build_model
from annotator.hed import HEDdetector
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector


def get_models(config):
    # optical flow
    flow_model = GMFlow(feature_channels=128,
                        num_scales=1,
                        upsample_factor=8,
                        num_head=1,
                        attention_type='swin',
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        ).to('cuda')

    checkpoint = torch.load(
        config['gmflow_path'], map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval()

    # saliency detection
    sod_model = build_model('resnet')
    sod_model.load_state_dict(torch.load(config['sod_path']))
    sod_model.to("cuda").eval()

    # controlnet
    if config['controlnet_type'] not in ['hed', 'depth', 'canny']:
        config['controlnet_type'] = 'hed'
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-"+config['controlnet_type'],
                                                 torch_dtype=torch.float16)
    controlnet.to("cuda")
    if config['controlnet_type'] == 'depth':
        detector = MidasDetector()
    elif config['controlnet_type'] == 'canny':
        detector = CannyDetector()
    else:
        detector = HEDdetector()

    # diffusion model
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        config['sd_path'], vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(
        config['num_inference_steps'], device=pipe._execution_device)

    frescoProc = apply_FRESCO_attn(pipe)
    frescoProc.controller.disable_controller()
    apply_FRESCO_opt(pipe)

    for param in flow_model.parameters():
        param.requires_grad = False
    for param in sod_model.parameters():
        param.requires_grad = False
    for param in controlnet.parameters():
        param.requires_grad = False
    for param in pipe.unet.parameters():
        param.requires_grad = False

    return pipe, frescoProc, controlnet, detector, flow_model, sod_model


def apply_control(x, detector, control_type):
    if control_type == 'depth':
        detected_map, _ = detector(x)
    elif control_type == 'canny':
        detected_map = detector(x, 50, 100)
    else:
        detected_map = detector(x)
    return detected_map


class ProcessingState(Enum):
    NULL = 0
    KEY_IMGS = 1
    
    
def cfg_to_input(filename):

    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    use_constraints = [
                        'spatial-guided attention',
                        'cross-frame attention',
                        'temporal-guided attention',
                        'spatial-guided optimization',
                        'temporal-guided optimization',
                    ]
    
    a_prompt = 'RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'
    n_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
    frame_count = get_frame_count(cfg['file_path'])
    args = [
        cfg['file_path'], cfg['prompt'], cfg['sd_path'], cfg['seed'], 512, cfg['cond_scale'],
        1.0, cfg['controlnet_type'], 50, 100,
        cfg['num_inference_steps'], 7.5, a_prompt, n_prompt,
        frame_count, cfg['batch_size'], cfg['mininterv'], cfg['maxinterv'],
        use_constraints, True, True, 4,
        1, 1, 1, 1
    ]
    return args


class GlobalState:
    def __init__(self):
        config_path = 'config/config_dog.yaml'
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.sd_model = config['sd_path']
        self.control_type = config['controlnet_type']
        self.processing_state = ProcessingState.NULL
        pipe, frescoProc, controlnet, detector, flow_model, sod_model = get_models(
            config)
        self.pipe = pipe
        self.frescoProc = frescoProc
        self.controlnet = controlnet
        self.detector = detector
        self.flow_model = flow_model
        self.sod_model = sod_model
        self.keys = []

    def update_controlnet_model(self, control_type):
        if self.control_type == control_type:
            return
        self.control_type = control_type
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-"+control_type,
                                                          torch_dtype=torch.float16)
        self.controlnet.to("cuda")
        if control_type == 'depth':
            self.detector = MidasDetector()
        elif control_type == 'canny':
            self.detector = CannyDetector()
        else:
            self.detector = HEDdetector()
        torch.cuda.empty_cache()
        for param in self.controlnet.parameters():
            param.requires_grad = False

    def update_sd_model(self, sd_model):
        if self.sd_model == sd_model:
            return
        self.sd_model = sd_model
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model, vae=vae, torch_dtype=torch.float16)
        self.pipe.scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.to("cuda")
        self.frescoProc = apply_FRESCO_attn(self.pipe)
        self.frescoProc.controller.disable_controller()
        torch.cuda.empty_cache()
        for param in self.pipe.unet.parameters():
            param.requires_grad = False


@torch.no_grad()
def process(*args):
    keypath = process1(*args)
    fullpath = process2(*args)
    return keypath, fullpath


@torch.no_grad()
def process1(input_path, prompt, sd_model, seed, image_resolution, control_strength,
             x0_strength, control_type, low_threshold, high_threshold,
             ddpm_steps, scale, a_prompt, n_prompt,
             frame_count, batch_size, mininterv, maxinterv,
             use_constraints, bg_smooth, use_poisson, max_process,
             b1, b2, s1, s2):
    global global_state
    global_state.update_controlnet_model(control_type)
    global_state.update_sd_model(sd_model)
    apply_freeu(global_state.pipe, b1=b1, b2=b2, s1=s1, s2=s2)

    filename = os.path.splitext(os.path.basename(input_path))[0]
    save_path = os.path.join('output', filename)
    device = global_state.pipe._execution_device
    guidance_scale = scale
    do_classifier_free_guidance = True
    global_state.pipe.scheduler.set_timesteps(ddpm_steps, device=device)
    timesteps = global_state.pipe.scheduler.timesteps
    cond_scale = [control_strength] * ddpm_steps
    dilate = Dilate(device=device)

    base_prompt = prompt
    video_cap = cv2.VideoCapture(input_path)
    frame_num = min(frame_count, int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    if mininterv > maxinterv:
        mininterv = maxinterv

    keys = get_keyframe_ind(input_path, frame_num, mininterv, maxinterv)
    if len(keys) < 3:
        raise gr.Error('Too few (%d) keyframes detected!' % (len(keys)))
    global_state.keys = keys
    fps = max(int(fps * len(keys) / frame_num), 1)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'keys'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'video'), exist_ok=True)

    sublists = [keys[i:i+batch_size-2]
                for i in range(2, len(keys), batch_size-2)]
    sublists[0].insert(0, keys[0])
    sublists[0].insert(1, keys[1])
    if len(sublists) > 1 and len(sublists[-1]) < 3:
        add_num = 3 - len(sublists[-1])
        sublists[-1] = sublists[-2][-add_num:] + sublists[-1]
        sublists[-2] = sublists[-2][:-add_num]

    batch_ind = 0
    propagation_mode = batch_ind > 0
    imgs = []
    record_latents = []
    video_cap = cv2.VideoCapture(input_path)

    for i in range(frame_num):
        success, frame = video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, image_resolution)
        H, W, C = img.shape
        Image.fromarray(img).save(os.path.join(
            save_path, 'video/%04d.png' % (i)))
        if i not in sublists[batch_ind]:
            continue

        imgs += [img]
        if i != sublists[batch_ind][-1]:
            continue

        # prepare input
        batch_size = len(imgs)
        n_prompts = [n_prompt] * len(imgs)
        prompts = [base_prompt + a_prompt] * len(sublists[batch_ind])
        if propagation_mode:
            prompts = ref_prompt + prompts

        prompt_embeds = global_state.pipe._encode_prompt(
            prompts,
            device,
            1,
            do_classifier_free_guidance,
            n_prompts,
        )

        imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)

        edges = torch.cat([numpy2tensor(apply_control(img,
                                                      global_state.detector, control_type)[:, :, None]) for img in imgs], dim=0)
        edges = edges.repeat(1, 3, 1, 1).cuda() * 0.5 + 0.5
        edges = torch.cat([edges.to(global_state.pipe.unet.dtype)] * 2)

        if bg_smooth:
            saliency = get_saliency(imgs, global_state.sod_model, dilate)
        else:
            saliency = None

        # prepare parameters for inter-frame and intra-frame consistency
        flows, occs, attn_mask, interattn_paras = get_flow_and_interframe_paras(
            global_state.flow_model, imgs)
        correlation_matrix = get_intraframe_paras(global_state.pipe, imgs_torch, global_state.frescoProc,
                                                  prompt_embeds, seed=seed)

        global_state.frescoProc.controller.disable_controller()
        if 'spatial-guided attention' in use_constraints:
            global_state.frescoProc.controller.enable_intraattn()
        if 'temporal-guided attention' in use_constraints:
            global_state.frescoProc.controller.enable_interattn(
                interattn_paras)
        if 'cross-frame attention' in use_constraints:
            global_state.frescoProc.controller.enable_cfattn(attn_mask)

        global_state.frescoProc.controller.enable_controller(
            interattn_paras=interattn_paras, attn_mask=attn_mask)
        optimize_temporal = True
        if 'temporal-guided optimization' not in use_constraints:
            correlation_matrix = []
        if 'spatial-guided optimization' not in use_constraints:
            optimize_temporal = False
        apply_FRESCO_opt(global_state.pipe, steps=timesteps[:int(ddpm_steps*0.75)],
                         flows=flows, occs=occs, correlation_matrix=correlation_matrix,
                         saliency=saliency, optimize_temporal=optimize_temporal)

        gc.collect()
        torch.cuda.empty_cache()

        # run!
        latents = inference(global_state.pipe, global_state.controlnet, global_state.frescoProc,
                            imgs_torch, prompt_embeds, edges, timesteps,
                            cond_scale, ddpm_steps, int(
                                ddpm_steps*(1-x0_strength)),
                            True, seed, guidance_scale, True,
                            record_latents, propagation_mode,
                            flows=flows, occs=occs, saliency=saliency, repeat_noise=True)

        with torch.no_grad():
            image = global_state.pipe.vae.decode(
                latents / global_state.pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = torch.clamp(image, -1, 1)
            save_imgs = tensor2numpy(image)
            bias = 2 if propagation_mode else 0
            for ind, num in enumerate(sublists[batch_ind]):
                Image.fromarray(
                    save_imgs[ind+bias]).save(os.path.join(save_path, 'keys/%04d.png' % (num)))

        batch_ind += 1
        # current batch uses the last frame of the previous batch as ref
        ref_prompt = [prompts[0], prompts[-1]]
        imgs = [imgs[0], imgs[-1]]
        propagation_mode = batch_ind > 0
        if batch_ind == len(sublists):
            gc.collect()
            torch.cuda.empty_cache()
            break

    writer = imageio.get_writer(os.path.join(save_path, 'key.mp4'), fps=fps)
    file_list = sorted(os.listdir(os.path.join(save_path, 'keys')))
    for file_name in file_list:
        if not (file_name.endswith('jpg') or file_name.endswith('png')):
            continue
        fn = os.path.join(os.path.join(save_path, 'keys'), file_name)
        curImg = imageio.imread(fn)
        writer.append_data(curImg)
    writer.close()

    global_state.processing_state = ProcessingState.KEY_IMGS
    return os.path.join(save_path, 'key.mp4')


@torch.no_grad()
def process2(input_path, prompt, sd_model, seed, image_resolution, control_strength,
             x0_strength, control_type, low_threshold, high_threshold,
             ddpm_steps, scale, a_prompt, n_prompt,
             frame_count, batch_size, mininterv, maxinterv,
             use_constraints, bg_smooth, use_poisson, max_process,
             b1, b2, s1, s2):

    global global_state
    if global_state.processing_state != ProcessingState.KEY_IMGS:
        raise gr.Error('Please generate key images before propagation')

    # reset blend dir
    filename = os.path.splitext(os.path.basename(input_path))[0]
    blend_dir = os.path.join('output', filename)
    os.makedirs(blend_dir, exist_ok=True)

    video_cap = cv2.VideoCapture(input_path)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    o_video = os.path.join(blend_dir, 'blend.mp4')
    key_ind = io.StringIO()
    for k in global_state.keys:
        print('%d' % (k), end=' ', file=key_ind)
    ps = '-ps' if use_poisson else ''
    cmd = (
        f'python video_blend.py {blend_dir} --key keys '
        f'--key_ind {key_ind.getvalue()} --output {o_video} --fps {fps} '
        f'--n_proc {max_process} {ps}')
    print(cmd)
    os.system(cmd)
    return o_video


config_dir = 'config'
filenames = os.listdir(config_dir)
config_list = []
for filename in filenames:
    if filename.endswith('yaml'):
        config_list.append(f'{config_dir}/{filename}')

global_state = GlobalState()
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## FRESCO Video-to-Video Translation')
    with gr.Row():
        with gr.Column():
            input_path = gr.Video(label='Input Video',
                                  source='upload',
                                  format='mp4',
                                  visible=True)
            prompt = gr.Textbox(label='Prompt')
            sd_model = gr.Dropdown(['SG161222/Realistic_Vision_V2.0', 
                                    'runwayml/stable-diffusion-v1-5',
                                    'stablediffusionapi/rev-animated',
                                    'stablediffusionapi/flat-2d-animerge'],
                                   label='Base model',
                                   value='SG161222/Realistic_Vision_V2.0')
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=2147483647,
                             step=1,
                             value=0,
                             randomize=True)
            run_button = gr.Button(value='Run All')
            with gr.Row():
                run_button1 = gr.Button(value='Run Key Frames')
                run_button2 = gr.Button(value='Run Propagation (Ebsynth)')
            with gr.Accordion('Advanced options for single frame processing',
                              open=False):
                image_resolution = gr.Slider(label='Frame resolution',
                                             minimum=256,
                                             maximum=512,
                                             value=512,
                                             step=64)
                control_strength = gr.Slider(label='ControlNet strength',
                                             minimum=0.0,
                                             maximum=2.0,
                                             value=1.0,
                                             step=0.01)
                x0_strength = gr.Slider(
                    label='Denoising strength',
                    minimum=0.00,
                    maximum=1.05,
                    value=0.75,
                    step=0.05,
                    info=('0: fully recover the input.'
                          '1.05: fully redraw the input.'))
                with gr.Row():
                    control_type = gr.Dropdown(['hed', 'canny', 'depth'],
                                               label='Control type',
                                               value='hed')
                    low_threshold = gr.Slider(label='Canny low threshold',
                                              minimum=1,
                                              maximum=255,
                                              value=50,
                                              step=1)
                    high_threshold = gr.Slider(label='Canny high threshold',
                                               minimum=1,
                                               maximum=255,
                                               value=100,
                                               step=1)
                ddpm_steps = gr.Slider(label='Steps',
                                       minimum=20,
                                       maximum=100,
                                       value=20,
                                       step=20)
                scale = gr.Slider(label='CFG scale',
                                  minimum=1.1,
                                  maximum=30.0,
                                  value=7.5,
                                  step=0.1)
                a_prompt = gr.Textbox(label='Added prompt',
                                      value='best quality, extremely detailed')
                n_prompt = gr.Textbox(
                    label='Negative prompt',
                    value=('longbody, lowres, bad anatomy, bad hands, '
                           'missing fingers, extra digit, fewer digits, '
                           'cropped, worst quality, low quality'))
                with gr.Row():
                    b1 = gr.Slider(label='FreeU first-stage backbone factor',
                                   minimum=1,
                                   maximum=1.6,
                                   value=1,
                                   step=0.01,
                                   info='FreeU to enhance texture and color')
                    b2 = gr.Slider(label='FreeU second-stage backbone factor',
                                   minimum=1,
                                   maximum=1.6,
                                   value=1,
                                   step=0.01)
                with gr.Row():
                    s1 = gr.Slider(label='FreeU first-stage skip factor',
                                   minimum=0,
                                   maximum=1,
                                   value=1,
                                   step=0.01)
                    s2 = gr.Slider(label='FreeU second-stage skip factor',
                                   minimum=0,
                                   maximum=1,
                                   value=1,
                                   step=0.01)
            with gr.Accordion('Advanced options for FRESCO constraints',
                              open=False):
                frame_count = gr.Slider(
                    label='Number of frames',
                    minimum=8,
                    maximum=300,
                    value=100,
                    step=1)
                batch_size = gr.Slider(
                    label='Number of frames in a batch',
                    minimum=3,
                    maximum=8,
                    value=8,
                    step=1)
                mininterv = gr.Slider(label='Min keyframe interval',
                                      minimum=1,
                                      maximum=20,
                                      value=5,
                                      step=1)
                maxinterv = gr.Slider(label='Max keyframe interval',
                                      minimum=1,
                                      maximum=50,
                                      value=20,
                                      step=1)
                use_constraints = gr.CheckboxGroup(
                    [
                        'spatial-guided attention',
                        'cross-frame attention',
                        'temporal-guided attention',
                        'spatial-guided optimization',
                        'temporal-guided optimization',
                    ],
                    label='Select the FRESCO contraints to be used',
                    value=[
                        'spatial-guided attention',
                        'cross-frame attention',
                        'temporal-guided attention',
                        'spatial-guided optimization',
                        'temporal-guided optimization',
                    ]),
                bg_smooth = gr.Checkbox(
                    label='Background smoothing',
                    value=True,
                    info='Select to smooth background')

            with gr.Accordion(
                    'Advanced options for the full video translation',
                    open=False):
                use_poisson = gr.Checkbox(
                    label='Gradient blending',
                    value=True,
                    info=('Blend the output video in gradient, to reduce'
                          ' ghosting artifacts (but may increase flickers)'))
                max_process = gr.Slider(label='Number of parallel processes',
                                        minimum=1,
                                        maximum=16,
                                        value=4,
                                        step=1)

            with gr.Accordion('Example configs', open=True):
                
                
                
                example_list = [cfg_to_input(x) for x in config_list]

                ips = [
                    input_path, prompt, sd_model, seed, image_resolution, control_strength,
                    x0_strength, control_type, low_threshold, high_threshold,
                    ddpm_steps, scale, a_prompt, n_prompt,
                    frame_count, batch_size, mininterv, maxinterv,
                    use_constraints[0], bg_smooth, use_poisson, max_process,
                    b1, b2, s1, s2
                ]

                gr.Examples(
                    examples=example_list,
                    inputs=[*ips],
                )

        with gr.Column():
            result_keyframe = gr.Video(label='Output key frame video',
                                       format='mp4',
                                       interactive=False)
            result_video = gr.Video(label='Output full video',
                                    format='mp4',
                                    interactive=False)
            
    def input_changed(path):
        if path is None:
            return (gr.Slider.update(), gr.Slider.update(), gr.Slider.update())
        frame_count = get_frame_count(path)
        if frame_count == 0:
            return (gr.Slider.update(), gr.Slider.update(), gr.Slider.update())
        if frame_count <= 8:
            raise gr.Error('The input video is too short!'
                           'Please input another video.')
        min_interv_l = 1
        max_interv_l = 1
        min_interv_c = min(5, frame_count)
        max_interv_c = min(20, frame_count)
        min_interv_r = frame_count
        max_interv_r = frame_count
        return (gr.Slider.update(minimum=min_interv_l, 
                                value=min_interv_c,
                                maximum=min_interv_r),
            gr.Slider.update(minimum=max_interv_l, 
                                value=max_interv_c,
                                maximum=max_interv_r),
            gr.Slider.update(minimum=8, 
                                value=frame_count,
                                maximum=frame_count),
            )

    input_path.change(input_changed, input_path, [mininterv, maxinterv, frame_count])
    input_path.upload(input_changed, input_path, [mininterv, maxinterv, frame_count])

    run_button.click(fn=process,
                     inputs=ips,
                     outputs=[result_keyframe, result_video])
    run_button1.click(fn=process1, inputs=ips, outputs=[result_keyframe])
    run_button2.click(fn=process2, inputs=ips, outputs=[result_video])

block.launch()
