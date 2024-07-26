from current_experiments.attacks import adv_loss
import torch
import argparse
import numpy as np
from PIL import Image
import glob
import os, json
import pandas as pd
from dataloader import load_celeba
from current_experiments.metrics import CLIP_similarity_metric, CLIP_prompt_metric
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch import autocast

#measure performance
def measure_performance(uncloaked_path, cloaked_path, original_path="", method="CLIP_similarity"):
    og_images = getImageTensorsFrom("CelebA/" + str(args.celeb))
    uncloaked_images = getImageTensorsFrom(experiment_directory + "/output-real-p0")
    cloaked_images = getImageTensorsFrom(experiment_directory + "/output-pgd-p0")
    og_features = getFeatureTensorsFrom("CelebA/" + str(args.celeb))
    uncloaked_features = getFeatureTensorsFrom(experiment_directory + "/output-real-p0")
    cloaked_features = getFeatureTensorsFrom(experiment_directory + "/output-pgd-p0")
    np.set_printoptions(suppress=True)
    
    uncloaked_metrics, _ = getCLIPPromptScores(uncloaked_images)
    uncloaked_metrics2, _ = getCLIPSimilarityScores(og_features, uncloaked_features)
    csv_file = experiment_directory + '/clip_metrics_uncloaked_p0.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in uncloaked_metrics.items():
            writer.writerow([key, value])
        for key, value in uncloaked_metrics2.items():
            writer.writerow([key, value])
    
    cloaked_metrics, _ = getCLIPPromptScores(cloaked_images)
    cloaked_metrics2, _ = getCLIPSimilarityScores(og_features, cloaked_features) 
    csv_file = experiment_directory + '/clip_metrics_pgd_p0.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in cloaked_metrics.items():
            writer.writerow([key, value])
        for key, value in cloaked_metrics2.items():
            writer.writerow([key, value])
            
    print("Unloaked metrics:", uncloaked_metrics)
    print("Cloaked metrics:", cloaked_metrics)
    print("Unloaked metrics:", uncloaked_metrics2)
    print("Cloaked metrics:", cloaked_metrics2)
    
#generate attacked faces using specified attack
def generate_attacked_faces(img_path, model, attack, save_path):
    #load images from img_path
    valid_path = "{}/*.*".format(img_path)
    images = map(Image.open, glob.glob(valid_path))
    for image in images:
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.0 * image - 1.0
        target = model.vae.encode(torch.zeros_like(image).half().cuda()).latent_dist.mean
        adv_image = adv_loss(image.half().cuda(), target, model.vae.encode, eps=16/255, step_size=0.02, steps=1000, attack=attack)
        adv_image = (adv_image / 2 + 0.5).clamp(0, 1)
        #save adv_image to img_path
        adv_image = adv_image[0].detach().cpu().numpy()
        img = Image.fromarray((adv_image * 255).astype(np.uint8))
        #if folder doesn't exist, create it
        if not os.path.isdir(save_path): os.mkdir(save_path)
        img.save(save_path)
        
def create_concept_list(image_path, malicious_activity="holding gun"):
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
            "instance_data_dir":    f"training-photos/{malicious_activity.replace(' ', '_')}",
            "class_data_dir":       "person"
        }
    ]
    # `class_data_dir` contains regularization images
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

def inference(output_images_directory, model_path="dreambooth-weights/800"):
    print("Starting image generation using model...")
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    g_cuda = None
    
    prompt = "photo of sks person holding a gun, studio lighting, masterpiece, 4k, ultra detailed, sharp focus, 8k, high definition, insanely detailed, intricate:1. 1)" 
    negative_prompt = "text, b&w, illustration, painting, cartoon, 3d, bad art, poorly drawn, close up, blurry,missing fingers, extra fingers, ugly fingers, long fingers" 
    num_samples = 10 
    guidance_scale = 7.5 
    num_inference_steps = 30 #100
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
    
def run_dreambooth(img_path, pretrained_model_name_or_path="SG161222/Realistic_Vision_V1.4", 
                   pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse", 
                   output_dir= "dreambooth-weights", 
                   revision="fp16", 
                   with_prior_preservation=True, 
                   prior_loss_weight=1.0, 
                   seed=1337, 
                   resolution=512, 
                   train_batch_size=1, 
                   train_text_encoder=True, 
                   mixed_precision="fp16", 
                   use_8bit_adam=True, 
                   gradient_accumulation_steps=1, 
                   learning_rate=1e-6, 
                   lr_scheduler="constant", 
                   lr_warmup_steps=0, 
                   num_class_images=100, 
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
        --revision {revision} \
        --with_prior_preservation {with_prior_preservation} \
        --prior_loss_weight {prior_loss_weight} \
        --seed {seed} \
        --resolution {resolution} \
        --train_batch_size {train_batch_size} \
        --train_text_encoder {train_text_encoder} \
        --mixed_precision {mixed_precision} \
        --use_8bit_adam {use_8bit_adam} \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --learning_rate {learning_rate} \
        --lr_scheduler {lr_scheduler} \
        --lr_warmup_steps {lr_warmup_steps} \
        --num_class_images {num_class_images} \
        --sample_batch_size {sample_batch_size} \
        --max_train_steps {max_train_steps} \
        --save_interval {save_interval} \
        --save_sample_prompt {save_sample_prompt} \
        --concepts_list_path {concepts_list_path}")
    #todo: add inferencing here

def clear_memory():
    torch.cuda.empty_cache()

def add_huggingface_token():
    print("Adding huggingface token...")
    os.system('mkdir -p ~/.huggingface')
    os.system('echo -n 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 > ~/.huggingface/token')

def setup():
  add_huggingface_token()
  load_celeba()
  
def full_attack(normal_image_path, attacked_image_path, model_path):
    for img_path in os.listdir(normal_image_path):
        experiment_directory = "experiments/experiment_" + img_path
        
        if not os.path.exists(experiment_directory + "/output-real-p0"):
            os.makedirs(experiment_directory + "/output-real-p0")
            
        if not os.path.exists(experiment_directory + "/input-pgd"):
            os.makedirs(experiment_directory + "/input-pgd")
        
        if not os.path.exists(experiment_directory + "/output-pgd-p0"):
            os.makedirs(experiment_directory + "/output-pgd-p0")
        
        print("Starting celeb number...", img_path)
        
        run_dreambooth(img_path=img_path)
        clear_memory()
        
        pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        generate_attacked_faces(img_path=img_path, model=pipe, attack="pgd", save_path=attacked_image_path)
        
        clear_memory()
        run_dreambooth(attacked_image_path)
        

def parse_args(args):
    parser = argparse.ArgumentParser(description="Dreambooth Attack")
    parser.add_argument("--setup", action="store_true", help="Setup Dreambooth Attack")
    parser.add_argument("--model_path", type=str, help="Path to Dreambooth model") #TODO: make required?
    parser.add_argument("--normal_image_path", type=str, help="Path to normal images", default="CelebA")
    parser.add_argument("--attacked_image_path", type=str, help="Path to attacked images", default="CelebA-attacked")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()
    if args.setup:
        setup()
    full_attack(args.normal_image_path, args.attacked_image_path, args.model_path)