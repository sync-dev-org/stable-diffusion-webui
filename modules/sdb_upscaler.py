import os, sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from modules.sdb_shared import opt

RealESRGAN_dir = opt.realesrgan_dir


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

def run_RealESRGAN(image, model_name: str):
    if RealESRGAN.model.name != model_name:
            try_loading_RealESRGAN(model_name)

    image = image.convert("RGB")

    output, img_mode = RealESRGAN.enhance(np.array(image, dtype=np.uint8))
    res = Image.fromarray(output)

    return res

class SyncDiffusionUpscaler():
    def __init__(self):
        self.model_name = 'RealESRGAN_x4plus'
        self.response = None
        try_loading_RealESRGAN('RealESRGAN_x4plus')

    async def set_job(self, source_filename, source_filepath, target_filename, target_filepath):
        self.source_filename = source_filename
        self.source_filepath = source_filepath
        self.target_filename = target_filename
        self.target_filepath = target_filepath

    async def set_model(self, model_name):
        self.model_name = model_name

    async def run(self):
        image = Image.open(self.source_filepath)
        self.response = run_RealESRGAN(image, self.model_name)

    async def get_response(self):
        return self.response
