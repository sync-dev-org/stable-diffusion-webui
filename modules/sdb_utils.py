import argparse, os, sys, glob
from email.mime import base
import gradio as gr
import k_diffusion as K
import math
import mimetypes
import numpy as np
import pynvml
import random
import threading, asyncio
import time
import torch
import torch.nn as nn
import yaml
import glob
from typing import List, Union

from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

from modules.sdb_shared import opt

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\|?*\n'


GFPGAN_dir = opt.gfpgan_dir
RealESRGAN_dir = opt.realesrgan_dir


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def crash(e, s, device, model):
#    global model
#    global device

    print(s, '\n', e)

    del model
    del device

    print('exiting...calling os._exit(0)')
    t = threading.Timer(0.25, os._exit, args=[0])
    t.start()


class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = 0

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print(f"[{self.name}] Recording max memory usage...\n")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
#            asyncio.sleep(0.1)
        print(f"[{self.name}] Stopped recording.\n")
        pynvml.nvmlShutdown()

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None

def create_random_tensors(shape, seeds, device):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_GFPGAN():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

def load_RealESRGAN(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    RealESRGAN_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    }

    model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(model_name+".pth not found at path "+model_path)

    sys.path.append(os.path.abspath(RealESRGAN_dir))
    from realesrgan import RealESRGANer

    instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0, half=True)
    instance.model.name = model_name
    return instance

def try_loading_RealESRGAN(model_name: str):
    global RealESRGAN
    if os.path.exists(RealESRGAN_dir):
        try:
            RealESRGAN = load_RealESRGAN(model_name) # TODO: Should try to load both models before giving up
            print("Loaded RealESRGAN with model "+RealESRGAN.model.name)
        except Exception:
            import traceback
            print("Error loading RealESRGAN:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

def load_embeddings(fp, model):
    if fp is not None and hasattr(model, "embedding_manager"):
        model.embedding_manager.load(fp.name)

def image_grid(imgs, batch_size, round_down=False, force_n_rows=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = int(rows) if round_down else round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0,2**32)
    n = abs(int(s) if s.isdigit() else hash(s))
    while n > 2**32:
        n = n >> 32
    return n

def draw_prompt_matrix(im, width, height, all_prompts):
    def wrap(text, d, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if d.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)

    def draw_texts(pos, x, y, texts, sizes):
        for i, (text, size) in enumerate(zip(texts, sizes)):
            active = pos & (1 << i) != 0

            if not active:
                text = '\u0336'.join(text) + '\u0336'

            d.multiline_text((x, y + size[1] / 2), text, font=fnt, fill=color_active if active else color_inactive, anchor="mm", align="center")

            y += size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fonts = ["arial.ttf", "DejaVuSans.ttf"]
    for font_name in fonts:
        try:
            fnt = ImageFont.truetype(font_name, fontsize)
            break
        except OSError:
           pass
    else:
        # ImageFont.load_default() is practically unusable as it only supports
        # latin1, so raise an exception instead
        raise Exception(f"No usable font found (tried {', '.join(fonts)})")
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_top = height // 4
    pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

    cols = im.width // width
    rows = im.height // height

    prompts = all_prompts[1:]

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
    prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

    sizes_hor = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]]
    sizes_ver = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]]
    hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
    ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_height / 2

        draw_texts(col, x, y, prompts_horiz, sizes_hor)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_height / 2

        draw_texts(row, x, y, prompts_vert, sizes_ver)

    return result


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


def check_prompt_length(prompt, comments, model):
    """this function tests if prompt is too long, and if so, adds a message to comments"""

    tokenizer = model.cond_stage_model.tokenizer
    max_length = model.cond_stage_model.max_length

    info = model.cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")


async def process_images(
        outpath, func_init, func_sample, prompt, seed, sampler_name, skip_grid, skip_save, batch_size,
        n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, use_RealESRGAN, realesrgan_model_name,
        fp, ddim_eta=0.0, do_not_save_grid=False, normalize_prompt_weights=True, init_img=None, init_mask=None,
        keep_mask=False, denoising_strength=0.75, resize_mode=None, uses_loopback=False,
        uses_random_seed_loopback=False, sort_samples=True, write_info_files=True, jpg_sample=False, model=None, device=None, GFPGAN=None):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    assert prompt is not None
    torch_gc()
    # start time after garbage collection (or before?)
    start_time = time.time()

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()

    if hasattr(model, "embedding_manager"):
        load_embeddings(fp, model)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    grid_count = len(os.listdir(outpath)) - 1

    comments = []

    prompt_matrix_parts = []
    if prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            current = prompt_matrix_parts[0]

            for n, text in enumerate(prompt_matrix_parts[1:]):
                print('for n, text in enumerate(prompt_matrix_parts[1:]):')
                print(text)
                if combination_num & (2 ** n) > 0:
                    current += ("" if text.strip().startswith(",") else ", ") + text

            all_prompts.append(current)

        n_iter = math.ceil(len(all_prompts) / batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
    else:

        if not opt.no_verify_input:
            try:
                check_prompt_length(prompt, comments, model)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    stats = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        init_data = func_init()
        tic = time.time()

        for n in range(n_iter):
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            uc = model.get_learned_conditioning(len(prompts) * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            # split the prompt if it has : for weighting
            # TODO for speed it might help to have this occur when all_prompts filled??
            subprompts,weights = split_weighted_subprompts(prompts[0])
            # get total weight for normalizing, this gets weird if large negative values used
            totalPromptWeight = sum(weights)

            # sub-prompt weighting used if more than 1
            if len(subprompts) > 1:
                c = torch.zeros_like(uc) # i dont know if this is correct.. but it works
                for i in range(0,len(subprompts)): # normalize each prompt and add it
                    weight = weights[i]
                    if normalize_prompt_weights:
                        weight = weight / totalPromptWeight
                    #print(f"{subprompts[i]} {weight*100.0}%")
                    # note if alpha negative, it functions same as torch.sub
                    c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
            else: # just behave like usual
                c = model.get_learned_conditioning(prompts)

            shape = [opt_C, height // opt_f, width // opt_f]

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds, device=device)
            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc, sampler_name=sampler_name)


            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                x_sample = x_sample.astype(np.uint8)

                if use_GFPGAN and GFPGAN is not None:
                    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample[:,:,::-1], has_aligned=False, only_center_face=False, paste_back=True)
                    x_sample = restored_img[:,:,::-1]

                if use_RealESRGAN and RealESRGAN is not None:
                    if RealESRGAN.model.name != realesrgan_model_name:
                        try_loading_RealESRGAN(realesrgan_model_name)

                    output, img_mode = RealESRGAN.enhance(x_sample[:,:,::-1])
                    x_sample = output[:,:,::-1]

                image = Image.fromarray(x_sample)
                if init_mask:
                    #init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
                    init_mask = init_mask.filter(ImageFilter.GaussianBlur(3))
                    init_mask = init_mask.convert('L')
                    init_img = init_img.convert('RGB')
                    image = image.convert('RGB')

                    if use_RealESRGAN and RealESRGAN is not None:
                        if RealESRGAN.model.name != realesrgan_model_name:
                            try_loading_RealESRGAN(realesrgan_model_name)
                        output, img_mode = RealESRGAN.enhance(np.array(init_img, dtype=np.uint8))
                        init_img = Image.fromarray(output)
                        init_img = init_img.convert('RGB')

                        output, img_mode = RealESRGAN.enhance(np.array(init_mask, dtype=np.uint8))
                        init_mask = Image.fromarray(output)
                        init_mask = init_mask.convert('L')

                    image = Image.composite(init_img, image, init_mask)

                sanitized_prompt = prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})
                if sort_samples:
                    sanitized_prompt = sanitized_prompt[:128] #200 is too long
                    sample_path_i = os.path.join(sample_path, sanitized_prompt)
                    os.makedirs(sample_path_i, exist_ok=True)
                    base_count = len(glob.glob(f"{sample_path_i}/*.png"))
                    filename = f"{base_count:05}-{seeds[i]}"
                else:
                    sample_path_i = sample_path
                    base_count = len(glob.glob(f"{sample_path_i}/*.png"))
                    sanitized_prompt = sanitized_prompt
                    filename = f"{base_count:05}-{seeds[i]}_{sanitized_prompt}"[:128] #same as before
                if not skip_save:
                    filename_i = os.path.join(sample_path_i, filename)
                    if not jpg_sample:
                        image.save(f"{filename_i}.png")
                    else:
                        image.save(f"{filename_i}.jpg", 'jpeg', quality=100, optimize=True)
                    if write_info_files:
                        # toggles differ for txt2img vs. img2img:
                        offset = 0 if init_img is None else 2
                        toggles = []
                        if prompt_matrix:
                            toggles.append(0)
                        if normalize_prompt_weights:
                            toggles.append(1)
                        if init_img is not None:
                            if uses_loopback:
                                toggles.append(2)
                            if uses_random_seed_loopback:
                                toggles.append(3)
                        if not skip_save:
                            toggles.append(2 + offset)
                        if not skip_grid:
                            toggles.append(3 + offset)
                        if sort_samples:
                            toggles.append(4 + offset)
                        if write_info_files:
                            toggles.append(5 + offset)
                        if use_GFPGAN:
                            toggles.append(6 + offset)
                        info_dict = dict(
                            target="txt2img" if init_img is None else "img2img",
                            prompt=prompts[i], ddim_steps=steps, toggles=toggles, sampler_name=sampler_name,
                            ddim_eta=ddim_eta, n_iter=n_iter, batch_size=batch_size, cfg_scale=cfg_scale,
                            seed=seed, width=width, height=height
                        )
                        if init_img is not None:
                            # Not yet any use for these, but they bloat up the files:
                            #info_dict["init_img"] = init_img
                            #info_dict["init_mask"] = init_mask
                            info_dict["denoising_strength"] = denoising_strength
                            info_dict["resize_mode"] = resize_mode
                        with open(f"{filename_i}.yaml", "w", encoding="utf8") as f:
                            yaml.dump(info_dict, f)

                output_images.append(image)
                base_count += 1

        if (prompt_matrix or not skip_grid) and not do_not_save_grid:
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)

            if prompt_matrix:
                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)


            grid_file = f"grid-{grid_count:05}-{seed}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.jpg"
            grid.save(os.path.join(outpath, grid_file), 'jpeg', quality=100, optimize=True)
            grid_count += 1
        toc = time.time()

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time

    info = f"""
{prompt}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}{', '+realesrgan_model_name if use_RealESRGAN and RealESRGAN is not None else ''}{', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()
    stats = f'''
Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    #mem_mon.stop()
    #del mem_mon
    torch_gc()

    return output_images, seed, info, stats



#def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str,
#            ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
#            height: int, width: int, fp):
async def txt2img(prompt: str, ddim_steps: int, sampler_name: str, toggles: List[int], realesrgan_model_name: str,
            ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int, width: int, fp, model, device, GFPGAN):
    outpath = opt.outdir_txt2img or opt.outdir or "outputs/txt2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    normalize_prompt_weights = 1 in toggles
    skip_save = 2 not in toggles
    skip_grid = 3 not in toggles
    sort_samples = 4 in toggles
    write_info_files = 5 in toggles
    jpg_sample = 6 in toggles
    use_GFPGAN = 7 in toggles
    use_RealESRGAN = 8 in toggles

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(model)
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x)
        return samples_ddim

    try:
        output_images, seed, info, stats = await process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name=sampler_name,
            skip_save=skip_save,
            skip_grid=skip_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=prompt_matrix,
            use_GFPGAN=use_GFPGAN,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=realesrgan_model_name,
            fp=fp,
            ddim_eta=ddim_eta,
            normalize_prompt_weights=normalize_prompt_weights,
            sort_samples=sort_samples,
            write_info_files=write_info_files,
            jpg_sample=jpg_sample,
            model=model,
            device=device,
            GFPGAN=GFPGAN
        )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (txt2img)!!', device, model)


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function !! + images, seed, comment, stats !! NOTE: changes to UI output must be reflected here too
        prompt, ddim_steps, sampler_name, toggles, ddim_eta, n_iter, batch_size, cfg_scale, seed, height, width, fp, images, seed, comment, stats = flag_data

        filenames = []

        with open("log/log.csv", "a", encoding="utf8", newline='') as file:
            import time
            import base64

            at_start = file.tell() == 0
            writer = csv.writer(file)
            if at_start:
                writer.writerow(["sep=,"])
                writer.writerow(["prompt", "seed", "width", "height", "sampler", "toggles", "n_iter", "n_samples", "cfg_scale", "steps", "filename"])

            filename_base = str(int(time.time() * 1000))
            for i, filedata in enumerate(images):
                filename = "log/images/"+filename_base + ("" if len(images) == 1 else "-"+str(i+1)) + ".png"

                if filedata.startswith("data:image/png;base64,"):
                    filedata = filedata[len("data:image/png;base64,"):]

                with open(filename, "wb") as imgfile:
                    imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

                filenames.append(filename)

            writer.writerow([prompt, seed, width, height, sampler_name, toggles, n_iter, batch_size, cfg_scale, ddim_steps, filenames[0]])

        print("Logged:", filenames[0])


async def img2img(prompt: str, image_editor_mode: str, init_info, mask_mode: str, ddim_steps: int, sampler_name: str,
            toggles: List[int], realesrgan_model_name: str, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float,
            seed: int, height: int, width: int, resize_mode: int, fp, model, device, GFPGAN):
    outpath = opt.outdir_img2img or opt.outdir or "outputs/img2img-samples"
    err = False
    seed = seed_to_int(seed)

    prompt_matrix = 0 in toggles
    normalize_prompt_weights = 1 in toggles
    loopback = 2 in toggles
    random_seed_loopback = 3 in toggles
    skip_save = 4 not in toggles
    skip_grid = 5 not in toggles
    sort_samples = 6 in toggles
    write_info_files = 7 in toggles
    jpg_sample = 8 in toggles
    use_GFPGAN = 9 in toggles
    use_RealESRGAN = 10 in toggles

    if sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(model,'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(model,'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(model,'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(model,'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(model,'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(model,'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    if image_editor_mode == 'Mask':
        init_img = init_info["image"]
        init_img = init_img.convert("RGB")
        init_img = resize_image(resize_mode, init_img, width, height)
        init_mask = init_info["mask"]
        init_mask = init_mask.convert("RGB")
        init_mask = resize_image(resize_mode, init_mask, width, height)
        keep_mask = mask_mode == 0
        init_mask = init_mask if keep_mask else ImageOps.invert(init_mask)
    else:
        init_img = init_info
        init_mask = None
        keep_mask = False

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(denoising_strength * ddim_steps)

    def init():
        image = init_img.convert("RGB")
        image = resize_image(resize_mode, image, width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        return init_latent,

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        if sampler_name != 'DDIM':
            x0, = init_data

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc - 1]

            xi = x0 + noise
            sigma_sched = sigmas[ddim_steps - t_enc - 1:]
            model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
            samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
        else:
            x0, = init_data
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
            z_enc = sampler.stochastic_encode(x0, torch.tensor([t_enc]*batch_size).to(device))
                                # decode it
            samples_ddim = sampler.decode(z_enc, conditioning, t_enc,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=unconditional_conditioning,)
        return samples_ddim


    try:
        if loopback:
            output_images, info = None, None
            history = []
            initial_seed = None

            for i in range(n_iter):
                output_images, seed, info, stats = await process_images(
                    outpath=outpath,
                    func_init=init,
                    func_sample=sample,
                    prompt=prompt,
                    seed=seed,
                    sampler_name=sampler_name,
                    skip_save=skip_save,
                    skip_grid=skip_grid,
                    batch_size=1,
                    n_iter=1,
                    steps=ddim_steps,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    prompt_matrix=prompt_matrix,
                    use_GFPGAN=use_GFPGAN,
                    use_RealESRGAN=False, # Forcefully disable upscaling when using loopback
                    realesrgan_model_name=realesrgan_model_name,
                    fp=fp,
                    do_not_save_grid=True,
                    normalize_prompt_weights=normalize_prompt_weights,
                    init_img=init_img,
                    init_mask=init_mask,
                    keep_mask=keep_mask,
                    denoising_strength=denoising_strength,
                    resize_mode=resize_mode,
                    uses_loopback=loopback,
                    uses_random_seed_loopback=random_seed_loopback,
                    sort_samples=sort_samples,
                    write_info_files=write_info_files,
                    jpg_sample=jpg_sample,
                    model=model,
                    device=device,
                    GFPGAN=GFPGAN
                )

                if initial_seed is None:
                    initial_seed = seed

                init_img = output_images[0]
                if not random_seed_loopback:
                    seed = seed + 1
                else:
                    seed = seed_to_int(None)
                denoising_strength = max(denoising_strength * 0.95, 0.1)
                history.append(init_img)

            if not skip_grid:
                grid_count = len(os.listdir(outpath)) - 1
                grid = image_grid(history, batch_size, force_n_rows=1)
                grid_file = f"grid-{grid_count:05}-{seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.jpg"
                grid.save(os.path.join(outpath, grid_file), 'jpeg', quality=100, optimize=True)


            output_images = history
            seed = initial_seed

        else:
            output_images, seed, info, stats = await process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                skip_save=skip_save,
                skip_grid=skip_grid,
                batch_size=batch_size,
                n_iter=n_iter,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=prompt_matrix,
                use_GFPGAN=use_GFPGAN,
                use_RealESRGAN=use_RealESRGAN,
                realesrgan_model_name=realesrgan_model_name,
                fp=fp,
                normalize_prompt_weights=normalize_prompt_weights,
                init_img=init_img,
                init_mask=init_mask,
                keep_mask=keep_mask,
                denoising_strength=denoising_strength,
                resize_mode=resize_mode,
                uses_loopback=loopback,
                sort_samples=sort_samples,
                write_info_files=write_info_files,
                jpg_sample=jpg_sample,
                model=model,
                device=device,
                GFPGAN=GFPGAN
            )

        del sampler

        return output_images, seed, info, stats
    except RuntimeError as e:
        err = e
        err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
        stats = err_msg
        return [], seed, 'err', stats
    finally:
        if err:
            crash(err, '!!Runtime error (img2img)!!', device, model)


# grabs all text up to the first occurrence of ':' as sub-prompt
# takes the value following ':' as weight
# if ':' has no value defined, defaults to 1.0
# repeats until no text remaining
# TODO this could probably be done with less code
def split_weighted_subprompts(text):
    print('Start split_weighted_subprompts')
    print('remove |')
    text = text.replace('|', "").replace('\n', ' ').replace(' ,', ',').replace(' ,', ',').replace(' ,', ',').replace(',,', ',').replace('  ', ' ').replace('  ', ' ')
    print('arg text')
    print(text)
    print('--------')
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        print('--------')
        print(f'while {remaining} > 0:')
#        if ":" in text:
        if "::" in text:
            print(':: found')
            idx = text.index("::") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            print('prompt')
            print(prompt)
            remaining -= idx
            # remove from main text
            text = text[idx+2:]
            # find value for weight, assume it is followed by a space or comma
            idx = len(text) # default is read to end of text
            if " " in text:
                idx = min(idx,text.index(" ")) # want the closer idx
            if "," in text:
                idx = min(idx,text.index(",")) # want the closer idx
            if "\n" in text:
                idx = min(idx,text.index("\n")) # want the closer idx
            if idx != 0:
                try:
                    print('find value for weight, assume it is followed by a space or comma')
                    weight = float(text[:idx])
                    print(weight)
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space or comma after a value?")
                    weight = 1.0
                    print(weight)
            else: # no value found
                print('no value found')
                weight = 1.0
                print(weight)
            # remove from main text
            remaining -= idx
            text = text[idx+2:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                print('no :: found')
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights

def run_GFPGAN(image, strength):
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)

    return res

def run_RealESRGAN(image, model_name: str):
    if RealESRGAN.model.name != model_name:
            try_loading_RealESRGAN(model_name)

    image = image.convert("RGB")

    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    res = Image.fromarray(output)

    return res


def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.update(value=resized_cropped_image)

def copy_img_to_input(selected=1, imgs = []):
    try:
        idx = int(0 if selected - 1 < 0 else selected - 1)
        image_data = re.sub('^data:image/.+;base64,', '', imgs[idx])
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return [processed_image, processed_image]
    except IndexError:
        return [None, None]

class SyncDiffusionWorker():
    def __init__(self, ckpt):
        self.ckpt = ckpt
        GFPGAN = None
        if os.path.exists(GFPGAN_dir):
            try:
                GFPGAN = load_GFPGAN()
                print("Loaded GFPGAN")
            except Exception:
                import traceback
                print("Error loading GFPGAN:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        RealESRGAN = None
        try_loading_RealESRGAN('RealESRGAN_x4plus')

        config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")

        model = load_model_from_config(config, self.ckpt['path'])

        gpu_ids = [0, 1]

        print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
        print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

        device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
        model = (model if opt.no_half else model.half()).to(device)


        if opt.defaults is not None and os.path.isfile(opt.defaults):
            try:
                with open(opt.defaults, "r", encoding="utf8") as f:
                    user_defaults = yaml.safe_load(f)
            except (OSError, yaml.YAMLError) as e:
                print(f"Error loading defaults file {opt.defaults}:", e)
                print("Falling back to program defaults.")
                user_defaults = {}
        else:
            user_defaults = {}

        # make sure these indicies line up at the top of txt2img()
        txt2img_toggles = [
            'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
            'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
            'Save individual images',
            'Save grid',
            'Sort samples by prompt',
            'Write sample info files',
            'jpg samples',
        ]
        if GFPGAN is not None:
            txt2img_toggles.append('Fix faces using GFPGAN')
        if RealESRGAN is not None:
            txt2img_toggles.append('Upscale images using RealESRGAN')

        txt2img_defaults = {
            'prompt': '',
            'ddim_steps': 50,
            'toggles': [1, 2, 3],
            'sampler_name': 'k_lms',
            'ddim_eta': 0.0,
            'n_iter': 1,
            'batch_size': 1,
            'cfg_scale': 7.5,
            'seed': '',
            'height': 512,
            'width': 512,
            'fp': None,
        }

        if 'txt2img' in user_defaults:
            txt2img_defaults.update(user_defaults['txt2img'])

        txt2img_toggle_defaults = [txt2img_toggles[i] for i in txt2img_defaults['toggles']]

        sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
        sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

        # make sure these indicies line up at the top of img2img()
        img2img_toggles = [
            'Create prompt matrix (separate multiple prompts using |, and get all combinations of them)',
            'Normalize Prompt Weights (ensure sum of weights add up to 1.0)',
            'Loopback (use images from previous batch when creating next batch)',
            'Random loopback seed',
            'Save individual images',
            'Save grid',
            'Sort samples by prompt',
            'Write sample info files',
            'jpg samples',
        ]
        if GFPGAN is not None:
            img2img_toggles.append('Fix faces using GFPGAN')
        if RealESRGAN is not None:
            img2img_toggles.append('Upscale images using RealESRGAN')

        img2img_mask_modes = [
            "Keep masked area",
            "Regenerate only masked area",
        ]

        img2img_resize_modes = [
            "Just resize",
            "Crop and resize",
            "Resize and fill",
        ]

        img2img_defaults = {
            'prompt': '',
            'ddim_steps': 50,
            'toggles': [1, 4, 5],
            'sampler_name': 'k_lms',
            'ddim_eta': 0.0,
            'n_iter': 1,
            'batch_size': 1,
            'cfg_scale': 5.0,
            'denoising_strength': 0.75,
            'mask_mode': 0,
            'resize_mode': 0,
            'seed': '',
            'height': 512,
            'width': 512,
            'fp': None,
        }

        if 'img2img' in user_defaults:
            img2img_defaults.update(user_defaults['img2img'])

        img2img_toggle_defaults = [img2img_toggles[i] for i in img2img_defaults['toggles']]
        img2img_image_mode = 'sketch'

        self.model = model
        self.device = device
        self.GFPGAN = GFPGAN


    async def dreaming(self, dream):
        print(dream)
        response = await txt2img(*dream[1], self.model, self.device, self.GFPGAN)
        # response =  output_images, seed, info, stats
        print(response)
        return response

    async def upsclaing(self, filename):
        img = Image.open(filename)
        model_name = 'RealESRGAN_x4plus'
        response = run_RealESRGAN(img, model_name)
        print(response)
        return response

