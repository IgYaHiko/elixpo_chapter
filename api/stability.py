import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

device = "cuda"

# Load SDXL Refiner
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

refiner.enable_xformers_memory_efficient_attention()

def refine_sdxl(
        input_path,
        output_path="refined.png",
        prompt="high detail, photorealistic, crisp textures",
        strength=0.25,
        steps=30,
        guidance=3.0
    ):

    # Load image
    img = Image.open(input_path).convert("RGB")

    # Run SDXL Refiner
    out = refiner(
        prompt=prompt,
        image=img,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]

    # Save
    out.save(output_path)
    print(f"[SDXL Refiner] Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    refine_sdxl("esrgan_upscaled.png", "sdxl_refined.png")
