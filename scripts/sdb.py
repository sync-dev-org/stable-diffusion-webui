import argparse, os
import asyncio
from pyexpat import model
from random import randint
import uuid
import time
import math
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Process, Value, Queue
from threading import Thread

from modules.sdb_shared import opt, processes
from modules.sdb_utils import SyncDiffusionWorker

import discord
from discord.ext import commands, tasks

from dotenv import load_dotenv
load_dotenv()

import questionary





def get_resolution(ar, basesize):
    aspect = ar.split(':')
    if float(aspect[0]) > float(aspect[1]):
      m = float(aspect[0]) / float(aspect[1])
      w = int(float(basesize) * m // 64 * 64)
      h = int(float(basesize) / float(aspect[0]) * float(aspect[1]) * m // 64 * 64)
    else:
      m = float(aspect[1]) / float(aspect[0])
      h = int(float(basesize) * m // 64 * 64)
      w = int(float(basesize) / float(aspect[1]) * float(aspect[0]) * m // 64 * 64)

    resolution = w * h
    max_resolution = 1024 ** 2
    if resolution > max_resolution:
      ratio = math.sqrt(float(resolution / max_resolution))
      w = int(w / ratio // 64 * 64)
      h = int(h / ratio // 64 * 64)
    
    return w, h

class ResultButtons(discord.ui.View):
    def __init__(self, *, timeout=180):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="UpScale", style=discord.ButtonStyle.gray)
    async def upscale_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            button.style = discord.ButtonStyle.green
            await interaction.response.send_message(content=f"UpScale!")
        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise

class InfoButtons(discord.ui.View):
    def __init__(self, instance, timeout=180):
        super().__init__(timeout=timeout)
        self.instance = instance

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.gray)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            button.style = discord.ButtonStyle.green
            while not self.instance.dream_queue.empty():
                self.instance.dream_queue.get()
            await interaction.response.send_message(content=f"Send Cancel Dreaming!")
        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise


class SyncDiffusionJob():
    def __init__(self, id, interaction, payload, seed, total_itr, current_itr, dream_queue, awaken_queue, message_queue, out_dir):
        self.id = id
        self.interaction = interaction
        self.status = 'queued'
        self.payload = payload
        self.seed = seed
        self.total_itr = total_itr
        self.current_itr = current_itr
        self.dream_queue = dream_queue
        self.awaken_queue = awaken_queue
        self.message_queue = message_queue
        self.out_dir = out_dir

    async def dreaming(self):
        try:
            self.status = 'dreaming'
            dream = [self.id, self.payload]
            self.dream_queue.put(dream)
        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise

    async def terminate(self, awaken):
        try:
            print(f'bot: SyncDiffusionJob(): terminate()')
            await asyncio.sleep(randint(0, 2))
            print(f'bot: SyncDiffusionJob(): terminate(): await asyncio.sleep')
            print(awaken)
            response_message = f'```seed:  {self.seed}\nitr: {self.current_itr} / {self.total_itr}\nfilename: {awaken[2]}\nuser: {self.interaction.user.name} ({self.interaction.user.id}){awaken[3]}```'
            print(f'bot: SyncDiffusionJob(): terminate(): response_message: {response_message}')
            fp = f'{self.out_dir}\{awaken[2]}'
            print(f'bot: SyncDiffusionJob(): terminate(): fp: {fp}')
            original_message = await self.interaction.original_response()
            print(f'bot: SyncDiffusionJob(): terminate(): original_message: {original_message}')
            await original_message.reply(content=response_message, file=discord.File(fp=fp, filename=awaken[2]), view=ResultButtons())
            self.status = 'terminated'
            print(f'bot: SyncDiffusionJob(): terminate(): status: {self.status}')
        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise


class SyncDiffusionBot(commands.Bot):
    def __init__(self, dream_queue, awaken_queue, message_queue):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.dream_queue = dream_queue
        self.awaken_queue = awaken_queue
        self.message_queue = message_queue

    async def on_ready(self):
        print(f'Logged in as {self.user}#{self.user.id}')
        await self.tree.sync()
        print('Synced!')

class SyncDiffusionCog(commands.Cog):
    def __init__(self, bot: commands.Bot, dream_queue, awaken_queue, message_queue, out_dir) -> None:
        self.loop = asyncio.get_event_loop()
        self.bot = bot
        self.dream_queue = dream_queue
        self.awaken_queue = awaken_queue
        self.message_queue = message_queue
        self.index = 0
        self.jobs = dict()
        self.out_dir = out_dir

    async def setup_hook(self) -> None:
        print(f'bot: setup_hook(self)')
        # create the background task and run it in the background

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'bot: on_ready')
        await self.bot_loop.start()

    @discord.app_commands.command(name="info")
    async def show_info(self, interaction: discord.Interaction):
        try:
            print(f'bot: show_info')
            worker_number = len(processes)
            await interaction.response.send_message(content=f"```job_queue_number: {self.dream_queue.qsize()}\nworker_number: {worker_number}```", view=InfoButtons(self))
        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise


    @discord.app_commands.command(name="dream")
    async def recieve_dream(self, interaction: discord.Interaction, prompt: str, seed: int = None, itr: int = 1, ar: str = '1:1', 
        basesize: int = 512, ddim_steps: int = 25, cfg_scale: float = 7.5, sampler_name: str = 'k_lms', matrix: bool = False, normalize: bool = True, 
        gfpgan: bool = True, realesrgan: bool = False, realesrgan_anime:bool = False):
        try: 
            print(f'bot: recieve_dream() / dream_queue: {self.dream_queue.qsize()}')

            username = interaction.user.name
            userid = interaction.user.id

            ddim_eta = 0.0
            n_iter = 1
            batch_size = 1
            
            width = get_resolution(ar, basesize)[0]
            height = get_resolution(ar, basesize)[1]

            toggles = []
            if matrix:
                toggles.append(0)
                ddim_steps = int(ddim_steps / 3 * 2)
            if normalize:
                toggles.append(1)
            if gfpgan:
                toggles.append(7)
            if realesrgan:
                toggles.append(8)

            realesrgan_model_name = 'RealESRGAN_x4plus'
            if realesrgan_anime:
                realesrgan_model_name = 'RealESRGAN_x4plus_anime_6B'

            if seed == None:
                random_seed = True
                seed_message = ''
            else:
                random_seed = False
                seed_message = f'seed: {seed} '

            fp = None
    
            message = f'{prompt}\n```itr:{itr} ar:{ar} basesize:{basesize} \nddim_steps:{ddim_steps} cfg_scale:{cfg_scale} sampler_name:{sampler_name} \nmatrix:{matrix} normalize:{normalize} gfpgan:{gfpgan} realesrgan:{realesrgan} realesrgan_anime:{realesrgan_anime}\nuser: {username} ({userid})```'
            await interaction.response.send_message(content=message, view=InfoButtons(self))
            print(itr)

            for i in range(itr):
                total_itr = itr
                current_itr = i + 1
                print(f'bot: recieve_dream: current_itr: {current_itr}')
                id = str(uuid.uuid4())
                if random_seed:
                    seed = randint(0, 9999999999)
                payload = [prompt, ddim_steps, sampler_name, toggles, realesrgan_model_name, ddim_eta, n_iter, batch_size, cfg_scale, seed, height, width, fp]
                job = SyncDiffusionJob(id, interaction, payload, seed, total_itr, current_itr, self.dream_queue, self.awaken_queue, self.message_queue, self.out_dir)
                self.jobs[id] = job
                await job.dreaming()

        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise


    @tasks.loop(seconds=0.5)  # task runs every 60 seconds
    async def bot_loop(self):
#        print(f'bot: bot_loop: self.awaken_queue / {self.awaken_queue.qsize()}')
        try:
            if not self.message_queue.empty():
                message = self.message_queue.get()
                print(f'bot: force_send_message: message: {message}')
                await discord.interaction.response.send_message(content=message)

            elif not self.awaken_queue.empty():
                print(f'bot: bot_loop: self.awaken_queue found! / {self.awaken_queue.qsize()}')
                awaken = self.awaken_queue.get()
                print(f'bot: bot_loop: self.awaken_queue: awaken: {awaken}')
                job = self.jobs[awaken[0]]
                await job.terminate(awaken)
                del self.jobs[awaken[0]]
                del job
                print(f'bot: bot_loop: complete! / {self.awaken_queue.qsize()}')

        except Exception as e:
            await discord.interaction.response.send_message(content=str(e))
            raise


def bot_launch(dream_queue, awaken_queue, message_queue, out_dir):
    retry_count = 0
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print('bot_launch: start bot')
            bot = SyncDiffusionBot(dream_queue, awaken_queue, message_queue)
            
            loop.run_until_complete(bot.add_cog(SyncDiffusionCog(bot, dream_queue, awaken_queue, message_queue, out_dir)))
            loop.run_until_complete(bot.start(token=TOKEN))
            raise Exception('bot_launch: stop bot')
        except Exception as e:
            retry_count += 1
            message = f'bot: retry: {retry_count} / has error: {e}'
            print(message)
            message_queue.put(message)
            del bot
            if retry_count > 100:
                break
            time.sleep(retry_count * 2 ** 3)

async def save_file(response, out_dir):
    try:
        print(f'response: {str(response)}')
        now = int(time.time())

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        filename = f'{now}_{response[1]}'
        image = response[0][0]
        fileio = BytesIO()
        image.save(fileio, 'PNG')
        file_size = fileio.tell()
        fileio.seek(0)
        if file_size < 6000000:
            filename = filename + '.png'   
            image.save(f'{out_dir}\{filename}', 'PNG')
        else:
            filename = filename + '.jpg'   
            image.save(f'{out_dir}\{filename}', 'JPEG')

        return filename

    except Exception as e:
        message = f'worker: save_file: has error: {e}'
        print(message)
        message_queue.put(message)



async def worker_loop(dream_queue, awaken_queue, message_queue, worker, out_dir, dream=None):
    dream = None
    while True:
        try:
#            print(f'Worker: worker_loop: dream_queue / {dream_queue.qsize()}')
            if not dream_queue.empty():
                print(f'Worker: worker_loop: dream_queue is found! / {dream_queue.qsize()}')
                dream = dream_queue.get()
                response = await worker.dreaming(dream)
                filename = await save_file(response, out_dir)
                dream.append(filename)
                dream.append(response[3])
                awaken = dream
                awaken_queue.put(awaken)
            else:
    #            print('Dream Queue is empty!')
                await asyncio.sleep(0.5)
        except Exception as e:
            message = f'worker: worker_loop: has error: {e}'
            print(message)
            message_queue.put(message)
            raise

def worker_launch(dream_queue, awaken_queue, message_queue, ckpt, out_dir):
    retry_count = 0
    dream = None
    while True:
        try:
            loop = asyncio.get_event_loop()
            print('start worker')
            worker = SyncDiffusionWorker(ckpt)
            loop.run_until_complete(worker_loop(dream_queue, awaken_queue, message_queue, worker, out_dir, dream))
        except Exception as e:
            retry_count += 1
            message = f'worker: retry: {retry_count} / has error: {e}'
            print(message)
            message_queue.put(message)
            del worker
            if retry_count > 3:
                break
            time.sleep(retry_count * 2 ** 3)



if __name__ == "__main__":
    if opt.modeltype == None:
        modeltype = questionary.select(
            'Model?',
            choices=['normal', 'waifu', 'trinart2-min', 'trinart2-max']
        ).ask()
    else:
        modeltype = opt.modeltype
    if opt.workers == None:
        workers = int(questionary.text('How many workers?').ask())
    else:
        workers = opt.workers

    if modeltype == 'normal':
        ckpt = 'models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
        TOKEN = os.getenv('DISCORD_TOKEN_NORMAL')
    elif modeltype == 'waifu':
        ckpt = 'models/ldm/stable-diffusion-v1/wd-v1-2-full-ema.ckpt'
        TOKEN = os.getenv('DISCORD_TOKEN_WAIFU')
    elif modeltype == 'trinart2-min':
        ckpt = 'models/ldm/stable-diffusion-v1/trinart2_step60000.ckpt'
        TOKEN = os.getenv('DISCORD_TOKEN_TRINART2_MIN')
    elif modeltype == 'trinart2-max':
        ckpt = 'models/ldm/stable-diffusion-v1/trinart2_step115000.ckpt'
        TOKEN = os.getenv('DISCORD_TOKEN_TRINART2_MAX')
    else:
        raise Exception('ckpt is not found!')

    out_dir = f'outputs\{opt.modeltype}'



    dream_queue = Queue()
    awaken_queue = Queue()
    message_queue = Queue()

    bot_thread = Thread(target=bot_launch, args=(dream_queue, awaken_queue, message_queue, out_dir))
    bot_thread.start()

    for i in range(workers):
        p = Process(target=worker_launch, args=(dream_queue, awaken_queue, message_queue, ckpt, out_dir))
        p.start()
        processes.append(p)

    bot_thread.join()
    for p in processes:
        p.join()
