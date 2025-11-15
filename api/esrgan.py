import cv2
from PIL import Image
import torch
from realesrgan import RealESRGAN

def upscale_realesrgan(input_path, output_path, scale=2, model="RealESRGAN_x4plus"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RealESRGAN(device, scale=scale, model_name=model)
    model.load_weights(f"{model}.pth", download=True)

    # Read image
    img = Image.open(input_path).convert("RGB")

    # Inference
    out = model.predict(img)

    # Save
    out.save(output_path)
    print(f"[RealESRGAN] Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    upscale_realesrgan("input.png", "esrgan_upscaled.png", scale=2)
