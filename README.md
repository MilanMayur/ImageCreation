# Image Generation Model Comparison using Stable Diffusion
This project compares multiple state-of-the-art text-to-image diffusion models by generating images from the same prompt and visually analyzing their outputs. The goal is to evaluate differences in image quality, realism, style, and creativity across models.

# Project Overview
Given a single text prompt, the system generates images using multiple diffusion pipelines and saves the results for side-by-side comparison.

# Prompt Used
```
A futuristic cityscape at sunset, with flying cars and glowing neon lights
```
![Screenshot (5)](https://github.com/user-attachments/assets/df8f53d8-bde6-48e0-9a86-69fa515d6ad0)

# Tech Stack
- Python 3.9+
- PyTorch
- Diffusers (Hugging Face)
- Stable Diffusion & SDXL Pipelines
- CUDA (GPU acceleration)
- Matplotlib
- PIL
- Hugging Face Hub

# Features
- Automatic GPU detection with FP16 optimization
- Supports both Stable Diffusion and SDXL pipelines
- Memory-efficient techniques:
  - Attention slicing
  - CPU offloading
- xFormers (if available)
- Saves generated images locally
- Displays side-by-side visual comparison

# Models Compared
| Model Name            | Hugging Face Model ID                      | Pipeline Used             | Key Characteristics                        |
| --------------------- | ------------------------------------------ | ------------------------- | ------------------------------------------ |
| Stable Diffusion v1-4 | `CompVis/stable-diffusion-v1-4`            | StableDiffusionPipeline   | Baseline diffusion model, balanced quality |
| DreamShaper v8        | `Lykon/dreamshaper-8`                      | StableDiffusionPipeline   | Artistic, stylized outputs                 |
| Realistic Vision v5.0 | `SG161222/RealVisXL_V5.0`                  | StableDiffusionXLPipeline | Photorealistic, SDXL-based                 |
| SDXL Base 1.0         | `stabilityai/stable-diffusion-xl-base-1.0` | StableDiffusionXLPipeline | High-resolution, advanced composition      |

# Setup & Installation
1- Clone the Repository
```
git clone https://github.com/your-username/image-generation-comparison.git
cd image-generation-comparison
```
2️- Install Dependencies
```
pip install torch diffusers transformers accelerate matplotlib pillow huggingface_hub
```

## Hugging Face Authentication
```
from huggingface_hub import login
login(token="your_huggingface_token")
```

3- How to Run
```
python model.py
```

The script will:
- Load each model
- Generate an image using the same prompt
- Save the output as .png
- Display a visual comparison using Matplotlib
