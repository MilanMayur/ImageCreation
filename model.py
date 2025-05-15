import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import login

login(token='your_token')

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"

# Prompt for image generation
prompt = "A futuristic cityscape at sunset, with flying cars and glowing neon lights"

# Models to compare
models = {
            "Stable Diffusion v1-4": "CompVis/stable-diffusion-v1-4",
            "DreamShaper v8": "Lykon/dreamshaper-8",
            "Realistic Vision v5.0": "SG161222/RealVisXL_V5.0",
            "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0"
        }

# Store images for comparison
generated_images = {}

for model_name, model_id in models.items():
    print(f"Generating image using {model_name}...")
    
    try:
        if "xl" in model_id.lower():
            pipe_cls = StableDiffusionXLPipeline
        else:
            pipe_cls = StableDiffusionPipeline

        if use_fp16:
            pipe = pipe_cls.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            pipe = pipe_cls.from_pretrained(model_id)

        pipe = pipe.to(device)
        
        # Memory optimization
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print("xformers not enabled:", e)

        print(f"Generating with {model_name}...")

        # Smaller size for SDXL
        if isinstance(pipe, StableDiffusionXLPipeline):
            image = pipe( 
                            prompt=prompt,
                            height=768,
                            width=512,
                            num_inference_steps=30,
                            guidance_scale=7.5
                        ).images[0]
        else:
            image = pipe(
                            prompt,
                            num_inference_steps=30,
                            guidance_scale=7.5
                        ).images[0]

        generated_images[model_name] = image
        
        # Save image
        safe_name = model_name.replace(" ", "_") + ".png"
        image.save(safe_name)
        print(f"Saved: {safe_name}")

        # Clear memory
        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Plot comparison
fig, axs = plt.subplots(1, len(generated_images), figsize=(20, 5))
for i, (model_name, img) in enumerate(generated_images.items()):
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(model_name, fontsize=10)
plt.tight_layout()
plt.show()
