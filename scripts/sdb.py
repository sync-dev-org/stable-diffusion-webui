from operator import ne
import os
import asyncio
import time
from io import BytesIO
from multiprocessing import Process, Queue
from threading import Thread

from modules.sdb_shared import opt, processes, cpkts
from modules.sdb_upscaler import SyncDiffusionUpscaler
from modules.sdb_utils import SyncDiffusionWorker
from modules.sdb_discord import SyncDiffusionBot, SyncDiffusionCog

from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps

import discord
from discord.ext import commands, tasks

import questionary
from dotenv import load_dotenv
load_dotenv()



def bot_launch(dream_queue, awaken_queue, message_queue, upscale_queue, out_dir, restart_queue):
    retry_count = 0
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print('bot_launch: start bot')
            bot = SyncDiffusionBot(dream_queue, awaken_queue, message_queue)
            
            loop.run_until_complete(bot.add_cog(SyncDiffusionCog(bot, dream_queue, awaken_queue, message_queue, upscale_queue, out_dir, restart_queue)))
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
        print(f'save_file: response: {str(response)}')
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
#                print(f'Worker {worker.ckpt["name"]}: worker_loop: dream_queue is found! / {dream_queue.qsize()}')
                dream = dream_queue.get()
#                print(f'----------------------')
                response = await worker.dreaming(dream)
                filename = await save_file(response, out_dir)
                dream.append(filename)
                dream.append(response[3])
                awaken = dream
                awaken_queue.put(awaken)
            else:
    #            print('Dream Queue is empty!')
                await asyncio.sleep(0.2)
        except Exception as e:
            message = f'Worker {worker.ckpt["name"]}: worker_loop: has error: {e}'
            print(message)
            message_queue.put(message)
            raise

def worker_launch(dream_queue, awaken_queue, message_queue, ckpt, out_dir):
    retry_count = 0
    dream = None
    while True:
        try:
            loop = asyncio.get_event_loop()
            print('Worker: worker_launch: start worker')
            worker = SyncDiffusionWorker(ckpt)
            print(f'Worker: worker_launch: {worker.ckpt["name"]} launched')
            loop.run_until_complete(worker_loop(dream_queue, awaken_queue, message_queue, worker, out_dir, dream))
        except Exception as e:
            retry_count += 1
            message = f'Worker: worker_launch: retry: {retry_count} / has error: {e}'
            print(message)
            message_queue.put(message)
            del worker
            if retry_count > 3:
                break
            time.sleep(retry_count * 2 ** 3)


async def upscaler_loop(upscale_queue, out_dir):
    upscaler = SyncDiffusionUpscaler()
    while True:
        try:
            if not upscale_queue.empty():
                job = upscale_queue.get()
                if job[1] == 'queue':
                    print(job)
                    await upscaler.set_job(*job[2])
                    await upscaler.run()
                    image = await upscaler.get_response()

                    filename = job[2][2]
                    fileio = BytesIO()
                    image.save(fileio, 'PNG')
                    file_size = fileio.tell()
                    fileio.seek(0)
                    if file_size < 6000000:
                        filename = filename + '.upscale.png'   
                        image.save(f'{out_dir}\{filename}', 'PNG')
                    else:
                        filename = filename + '.upscale.jpg'
                        image.save(f'{out_dir}\{filename}', 'JPEG')

                    next_job = [job[0], 'done', job[2]]
                    upscale_queue.put(next_job)
                    print(next_job)
                else:
                    upscale_queue.put(job)
                    print(f'upscale_queue found, but not status is queue: {job}')
                    await asyncio.sleep(0.8)

            else:
                await asyncio.sleep(0.5)
        except Exception as e:
            message = f'upscaler_loop: has error: {e}'
            print(message)
            raise

def upscaler_launch(upscale_queue, out_dir):
    print('1')
    retry_count = 0
    while True:
        try:
            loop = asyncio.get_event_loop()
            print('start upscaler')
            loop.run_until_complete(upscaler_loop(upscale_queue, out_dir))
        except Exception as e:
            retry_count += 1
            message = f'upscaler: retry: {retry_count} / has error: {e}'
            print(message)
            if retry_count > 3:
                break
            time.sleep(retry_count * 2 ** 3)


if __name__ == "__main__":

    TOKEN = os.getenv('DISCORD_TOKEN')

    out_dir = f'outputs\gevanni'


    restart_queue = Queue()
    dream_queue = Queue()
    awaken_queue = Queue()
    message_queue = Queue()
    upscale_queue = Queue()

    bot_thread = Thread(target=bot_launch, args=(dream_queue, awaken_queue, message_queue, upscale_queue, out_dir, restart_queue))
    bot_thread.start()

    upscaler_process = Process(target=upscaler_launch, args=(upscale_queue, out_dir))
    upscaler_process.start()


    search_ckpt = list(filter(lambda item : item['name'] == 'sd1.5', cpkts))
    default_ckpt = search_ckpt[0]

    p = Process(target=worker_launch, args=(dream_queue, awaken_queue, message_queue, default_ckpt, out_dir))
    p.start()
    processes.append(p)


    '''
    for i in range(len(cpkts)):
        p = Process(target=worker_launch, args=(dream_queue, awaken_queue, message_queue, cpkts[i], out_dir))
        p.start()
        processes.append(p)
    '''


    while True:
        if not restart_queue.empty:
            print('restart_queue is found!')




    '''
    bot_thread.join()
    for p in processes:
        p.join()
    '''
