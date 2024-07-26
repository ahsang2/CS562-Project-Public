from train_dreambooth import dreambooth_main
import torch
import argparse
from argparse import Namespace  
import numpy as np
from PIL import Image
import glob
import subprocess, json, os
import pandas as pd
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel
from torch import autocast
import gc

def runPGD(real_image_dir, cloaked_output_dir):
    subprocess.run('python3 diffuser_attack.py --real_image_dir=' + real_image_dir + ' --cloaked_output_dir=' + cloaked_output_dir, shell=True)
    return

def create_concept_list(image_path, malicious_activity="shooting gun"):
    concepts_list = [
        {
            "instance_prompt":      "photo of sks person",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    image_path,
            "class_data_dir":       "person"
        },
        {
            "instance_prompt":      f"photo of person {malicious_activity}",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    f"{malicious_activity.replace(' ', '_')}",
            "class_data_dir":       "person"
        }
    ]

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

def run_dreambooth(img_path, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1", 
                   pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse", 
                   output_dir= "dreambooth-weights", 
                   revision="fp16", 
                   with_prior_preservation=True, 
                   prior_loss_weight=1.0, 
                   resolution=512, 
                   train_batch_size=1, 
                   train_text_encoder=True, 
                   mixed_precision="fp16", 
                   use_8bit_adam=True, 
                   gradient_accumulation_steps=1, 
                   learning_rate=1e-6, 
                   lr_scheduler="constant", 
                   lr_warmup_steps=0, 
                   num_class_images=50, 
                   sample_batch_size=4, 
                   max_train_steps=800, 
                   save_interval=10000, 
                   save_sample_prompt="photo of sks person", 
                   concepts_list_path="concepts_list.json"):
    create_concept_list(img_path)
    os.system(f"accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path {pretrained_model_name_or_path} \
        --pretrained_vae_name_or_path {pretrained_vae_name_or_path} \
        --output_dir {output_dir} \
        --with_prior_preservation \
        --prior_loss_weight {prior_loss_weight} \
        --resolution {resolution} \
        --train_batch_size {train_batch_size} \
        --train_text_encoder \
        --mixed_precision {mixed_precision} \
        --use_8bit_adam \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --learning_rate {learning_rate} \
        --lr_scheduler {lr_scheduler} \
        --lr_warmup_steps {lr_warmup_steps} \
        --num_class_images {num_class_images} \
        --sample_batch_size {sample_batch_size} \
        --max_train_steps {max_train_steps} \
        --save_interval {save_interval} \
        --concepts_list {concepts_list_path}")

def trainAndRunDreambooth(input_images_directory, output_images_directory, celeb_index):
    print("Starting model training...")
    run_dreambooth(input_images_directory) # trains and saves model in "dreambooth-weights" 
    print("Completed model training")    
    print("Starting image generation using model...")
    text_encoder = CLIPTextModel.from_pretrained("dreambooth-weights/800/text_encoder")
    pipe = StableDiffusionPipeline.from_pretrained("dreambooth-weights/800", text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    g_cuda = None
    
    prompt = "photo of sks person shooting gun, full body, studio lighting, masterpiece, 4k, realistic, ultra detailed, sharp focus, 8k, high definition, insanely detailed, intricate:1. 1)" 
    negative_prompt = "text, b&w, illustration, painting, cartoon, 3d, bad art, poorly drawn, close up, blurry, missing fingers, extra fingers, ugly fingers, long fingers, picture of gun, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck" 
    num_samples = 10 
    guidance_scale = 7.5 
    num_inference_steps = 50 #100
    height = 512 
    width = 512 

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for i, img in enumerate(images):
        img.save(output_images_directory + "/img" + str(i) + ".png")
        
    print("Completed image generation using model")  
    
def clear_memory():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

def run_experiments():
    for celeb_directory in os.listdir("celebrity_training_data"):
        experiment_directory = "base_experiments/experiment_" + celeb_directory
        
        # if not os.path.exists(experiment_directory + "/output-real-p0"):
        #     os.makedirs(experiment_directory + "/output-real-p0")
            
        if not os.path.exists(experiment_directory + "/input-base"):
            os.makedirs(experiment_directory + "/input-base")
        
        if not os.path.exists(experiment_directory + "/output-base-p0"):
            os.makedirs(experiment_directory + "/output-base-p0")
        
        print("Starting celeb number...", celeb_directory)
        
        # trainAndRunDreambooth("CelebA/" + celeb_directory, experiment_directory + "/output-real-p0", int(celeb_directory)) 
        
        print("Starting diffuser attack...")
        clear_memory()
        runPGD("celebrity_training_data" + celeb_directory, experiment_directory + "/input-base")
        trainAndRunDreambooth(experiment_directory + "/input-base", experiment_directory + "/output-base-p0", int(celeb_directory)) 
        print("Finished diffuser attack...")
        
        print("Starting metric evaluation...")
        clear_memory()
        subprocess.run('python3 clip_evaluations.py --celeb=' + str(celeb_directory), shell=True)
        print("Completed metric evaluation")
        
        print("Completed celeb number...", celeb_directory)    
        
def add_huggingface_token():
    print("Adding huggingface token...")
    os.system('echo -n REDACTED > /root/.cache/huggingface/token')
    
if __name__ == "__main__":
    add_huggingface_token()
    run_experiments()