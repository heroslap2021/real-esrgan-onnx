import os
import sys
import cv2
import numpy as np
import subprocess
import platform

from concurrent.futures import ThreadPoolExecutor

from argparse import ArgumentParser
from tqdm import tqdm

import onnxruntime as rt
rt.set_default_logger_severity(3)

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

parser = ArgumentParser()
parser.add_argument("--source", help="path to source video")
parser.add_argument("--result", help="path to result video")
parser.add_argument("--audio", default=False, action="store_true", help="Keep audio")
parser.add_argument("--model", default='RealESRGAN_x2_fp16', help="model name")
opt = parser.parse_args()

from RealEsrganONNX.esrganONNX import RealESRGAN_ONNX
enhancer = RealESRGAN_ONNX(model_path=f"RealEsrganONNX/{opt.model}.onnx", device="cuda")

def ffmpeg_merge_frames(sequence_directory, pattern, destination, fps=30, crf=18, lowest_bitrate=None):
    pass1 = ['-pass','1','-an','-f','mp4','/dev/null']
    pass2 = ['-pass','2',destination]
    crf_bitrate, val_crf_bitrate = (
        '-b:v', f"{lowest_bitrate}k"
        ) if lowest_bitrate else (
            '-crf', str(crf)
        )

    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-hwaccel', 'cuda',
        '-r', str(fps),
        '-i', os.path.join(sequence_directory, pattern),
        '-c:v', 'h264_nvenc',
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        crf_bitrate, val_crf_bitrate,
        '-y'
    ]
    process = subprocess.Popen(cmd + pass1)
    process.communicate()
    if process.returncode != 0:
        print(f"Error: Failed to merge image sequence.")
        return None
    process = subprocess.Popen(cmd + pass2)
    process.communicate()
    if process.returncode == 0:
        return destination
    print(f"Error: Failed to merge image sequence.")
    return None

video = cv2.VideoCapture(opt.source)

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))    
fps = video.get(cv2.CAP_PROP_FPS)

temp_path = os.path.join("/", "temp_frames")
os.makedirs(temp_path, exist_ok=True)

cmd = ['ffmpeg', '-loglevel', 'error', '-hwaccel', 'cuda', '-c:v', 'h264_cuvid', '-i', opt.source, '-vf', f'select=1,setpts=N/(FRAME_RATE*TB),fps={fps}', '-vsync', 'vfr', '-y', os.path.join(temp_path, 'frame_%08d.png')]
process = subprocess.Popen(cmd)
sys.stdout.flush()
process.communicate()
if process.returncode != 0:
    print(f"Error: Failed to extract video.")
    sys.exit(1)

fp16 = True if "fp16" in opt.model else False

def upsampler_idx(idx):
    img_path = os.path.join(temp_path, f"frame_{idx:08d}.png")
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    output = enhancer.enhance_fp16(frame) if fp16 else enhancer.enhance(frame)
    cv2.imwrite(img_path, output)

with ThreadPoolExecutor(max_workers=24) as executor:
    with tqdm(total=n_frames, desc="[ Processing ]", ncols=70, colour="green") as pbar:
        xy = 1
        for _ in executor.map(upsampler_idx, range(1,n_frames+1)):
            pbar.update(1)
            progress((xy/n_frames), desc="Processing")
            xy += 1

video.release()
cv2.destroyAllWindows()

destination = ffmpeg_merge_frames(temp_path, 'frame_%08d.png', opt.result, fps=fps, crf=18, lowest_bitrate=None)

if opt.audio:
    extracted_audio_path = os.path.join('/', 'extracted_audio.aac')
    cmd1 = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', source,
        '-vn',
        '-c:a', 'aac',
        '-y',
        extracted_audio_path
    ]
    process = subprocess.Popen(cmd1)
    process.communicate()
    if process.returncode != 0:
        print(f"Error: Failed to extract audio.")
        sys.exit(1)
    cmd2 = [
        'ffmpeg',
        '-loglevel', 'error',
        '-hwaccel', 'cuda',
        '-i', destination,
        '-i', extracted_audio_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', destination.replace('.mp4', '_audio.mp4')
    ]
    process = subprocess.Popen(cmd2)
    process.communicate()
    if process.returncode == 0:
        if os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)
    else:
        print(f"Error: Failed to mux audio.")
        sys.exit(1)

print("Done")
