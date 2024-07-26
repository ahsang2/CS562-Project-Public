import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torchvision.transforms as T
from PIL import Image
import argparse
from transformers import CLIPTextModel

totensor = T.ToTensor()
topil = T.ToPILImage()

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean).norm()

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv], retain_graph=False, create_graph=False)
        
        with torch.no_grad():
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None

        if mask is not None:
            X_adv.data *= mask

    return X_adv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CLIP metrics')
    parser.add_argument('--real_image_dir', type=str, help='Description of the argument')
    parser.add_argument('--cloaked_output_dir', type=str, help='Description of the argument')
    parser.add_argument('--base', action='store_true', help='Description of the argument')
    args = parser.parse_args()
    real_image_dir = args.real_image_dir
    cloaked_output_dir = args.cloaked_output_dir
    
    if args.base:
        model_id_or_path = "stabilityai/stable-diffusion-2-1"
    else:
        model_id_or_path = "dreambooth-weights/800"
        text_encoder = CLIPTextModel.from_pretrained("dreambooth-weights/800/text_encoder")
    pipe_img2img = StableDiffusionPipeline.from_pretrained(model_id_or_path, text_encoder=text_encoder, revision="fp16",torch_dtype=torch.float16).to("cuda")

    for img in os.listdir(real_image_dir):
        real_path = real_image_dir + "/" + img
        cloaked_path = cloaked_output_dir + "/" + img

        init_image = Image.open((real_path)).convert("RGB")
        resize = T.transforms.Resize(512)
        center_crop = T.transforms.CenterCrop(512)
        init_image = center_crop(resize(init_image))

        with torch.autocast('cuda'):
            X = preprocess(init_image).half().cuda()
            adv_X = pgd(X,
                        model=pipe_img2img.vae.encode,
                        clamp_min=-1,
                        clamp_max=1,
                        eps=0.06, # The higher, the less imperceptible the attack is
                        step_size=0.02, # Set smaller than eps
                        iters=1000, # The higher, the stronger your attack will be
                    )

            # convert pixels back to [0,1] range
            adv_X = (adv_X / 2 + 0.5).clamp(0, 1)

            adv_image = topil(adv_X[0]).convert("RGB")
            adv_image.save(cloaked_path)
            
        with torch.no_grad():
            torch.cuda.empty_cache()