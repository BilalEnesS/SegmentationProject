"""
BackgroundReplacer class for image segmentation and background replacement.
"""
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image, ImageFilter, ImageDraw
import torch
import numpy as np
import cv2
import requests
import io
import openai
from openai import OpenAI
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

class BackgroundReplacer:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the BackgroundReplacer with optional OpenAI API key.
        """
        print("Loading CLIPSeg model...")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("OpenAI API initialized successfully!")
        else:
            self.openai_client = None
            print("No OpenAI API key provided. Will use placeholder backgrounds.")
        print("Model loaded successfully!")

    def create_mask(self, image: Image.Image, prompt: str, threshold: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a segmentation mask for the given prompt.
        """
        inputs = self.processor(
            text=[prompt], 
            images=[image], 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits
        mask = torch.sigmoid(preds[0]).numpy()
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        return binary_mask, mask

    def refine_mask(self, mask: np.ndarray, kernel_size: int = 5, blur_radius: int = 2) -> np.ndarray:
        """
        Refine the mask by smoothing edges and reducing noise.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        return mask

    def extract_subject(self, image: Image.Image, mask: np.ndarray, save_path: Optional[str] = None) -> Image.Image:
        """
        Extract the subject from the image using the mask, make the rest transparent.
        """
        if mask.shape[:2] != image.size[::-1]:
            mask_img = Image.fromarray(mask).resize(image.size, Image.NEAREST)
            mask = np.array(mask_img)
        img_array = np.array(image)
        if len(mask.shape) == 2:
            mask_3ch = np.stack([mask] * 3, axis=-1)
        else:
            mask_3ch = mask
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array
        rgba_array[:, :, 3] = (mask_normalized[:, :, 0] * 255).astype(np.uint8)
        extracted_image = Image.fromarray(rgba_array, 'RGBA')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            extracted_image.save(save_path)
            print(f"Extracted subject saved to: {save_path}")
        return extracted_image

    def create_white_background_version(self, image: Image.Image, mask: np.ndarray, save_path: Optional[str] = None) -> Image.Image:
        """
        Place the segmented subject on a white background.
        """
        if mask.shape[:2] != image.size[::-1]:
            mask_img = Image.fromarray(mask).resize(image.size, Image.NEAREST)
            mask = np.array(mask_img)
        img_array = np.array(image)
        white_bg = np.ones_like(img_array) * 255
        mask_normalized = mask.astype(np.float32) / 255.0
        if len(mask_normalized.shape) == 2:
            mask_normalized = np.stack([mask_normalized] * 3, axis=-1)
        result = white_bg * (1 - mask_normalized) + img_array * mask_normalized
        result = result.astype(np.uint8)
        result_image = Image.fromarray(result)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result_image.save(save_path)
            print(f"White background version saved to: {save_path}")
        return result_image

    def generate_background_with_api(self, prompt: str, width: int = 512, height: int = 512, model: str = "dall-e-3", quality: str = "hd") -> Image.Image:
        """
        Generate a new background using OpenAI DALL-E API.
        """
        if not self.openai_client:
            print("OpenAI API not configured. Creating placeholder background...")
            return self._create_placeholder_background(width, height, prompt)
        print(f"Generating background with DALL-E: '{prompt}'")
        try:
            size = self._get_dalle_size(width, height, model)
            enhanced_prompt = f"High-quality professional background image: {prompt}. Clean, detailed, photorealistic style, suitable as background image. No people, no text, no watermarks."
            response = self.openai_client.images.generate(
                model=model,
                prompt=enhanced_prompt,
                size=size,
                quality=quality,
                n=1,
                response_format="url"
            )
            image_url = response.data[0].url
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            print(f"Background generated successfully! Size: {image.size}")
            return image
        except Exception as e:
            print(f"Error generating background with DALL-E: {str(e)}")
            print("Falling back to placeholder background...")
            return self._create_placeholder_background(width, height, prompt)

    def _get_dalle_size(self, width: int, height: int, model: str) -> str:
        """
        Get supported DALL-E size string for the given width and height.
        """
        if model == "dall-e-3":
            if width == height:
                return "1024x1024"
            elif width > height:
                return "1792x1024"
            else:
                return "1024x1792"
        else:
            if max(width, height) <= 256:
                return "256x256"
            elif max(width, height) <= 512:
                return "512x512"
            else:
                return "1024x1024"

    def _create_placeholder_background(self, width: int, height: int, prompt: str) -> Image.Image:
        """
        Create a placeholder background if API is not available or fails.
        """
        colors = {
            'sunset': ('#FF6B6B', '#4ECDC4'),
            'ocean': ('#0074D9', '#7FDBFF'),
            'forest': ('#2ECC40', '#01FF70'),
            'mountain': ('#B10DC9', '#F012BE'),
            'city': ('#111111', '#85144b'),
            'desert': ('#FF851B', '#FFDC00'),
            'space': ('#001f3f', '#7FDBFF'),
            'default': ('#74B9FF', '#A29BFE')
        }
        color_key = 'default'
        for key in colors.keys():
            if key in prompt.lower():
                color_key = key
                break
        start_color, end_color = colors[color_key]
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        for y in range(height):
            ratio = y / height
            r = int(int(start_color[1:3], 16) * (1 - ratio) + int(end_color[1:3], 16) * ratio)
            g = int(int(start_color[3:5], 16) * (1 - ratio) + int(end_color[3:5], 16) * ratio)
            b = int(int(start_color[5:7], 16) * (1 - ratio) + int(end_color[5:7], 16) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        return img

    def composite_subject_on_background(self, subject_rgba: Image.Image, background: Image.Image, save_path: Optional[str] = None) -> Image.Image:
        """
        Composite the segmented subject onto a new background.
        """
        subject_size = subject_rgba.size
        background_resized = background.resize(subject_size, Image.Resampling.LANCZOS)
        if background_resized.mode != 'RGBA':
            background_resized = background_resized.convert('RGBA')
        result = Image.alpha_composite(background_resized, subject_rgba)
        result_rgb = result.convert('RGB')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result_rgb.save(save_path, quality=95)
            print(f"Final result saved to: {save_path}")
        return result_rgb

    def process_full_pipeline(self, image_path: str, subject_prompt: str, background_prompt: str, threshold: float = 0.4, output_dir: str = "output") -> Dict[str, Any]:
        """
        Full pipeline: segmentation -> extraction -> new background -> composition.
        """
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image.size}")
        print(f"Creating mask for: '{subject_prompt}'")
        binary_mask, raw_mask = self.create_mask(image, subject_prompt, threshold)
        refined_mask = self.refine_mask(binary_mask)
        if refined_mask.shape[:2] != image.size[::-1]:
            mask_img = Image.fromarray(refined_mask).resize(image.size, Image.NEAREST)
            refined_mask = np.array(mask_img)
        extracted_subject = self.extract_subject(
            image, refined_mask, 
            save_path=f"{output_dir}/extracted_subject.png"
        )
        white_bg_version = self.create_white_background_version(
            image, refined_mask,
            save_path=f"{output_dir}/white_background.jpg"
        )
        new_background = self.generate_background_with_api(
            background_prompt, 
            width=image.size[0], 
            height=image.size[1]
        )
        final_result = self.composite_subject_on_background(
            extracted_subject, new_background,
            save_path=f"{output_dir}/final_result.jpg"
        )
        self.visualize_results(image, binary_mask, refined_mask, 
                              extracted_subject, new_background, final_result)
        return {
            'original': image,
            'mask': refined_mask,
            'extracted': extracted_subject,
            'white_bg': white_bg_version,
            'new_background': new_background,
            'final': final_result
        }

    def visualize_results(self, original: Image.Image, mask: np.ndarray, refined_mask: np.ndarray, extracted: Image.Image, new_bg: Image.Image, final: Image.Image) -> None:
        """
        Visualize the results of the pipeline.
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title("Binary Mask")
        axes[0, 1].axis('off')
        axes[0, 2].imshow(refined_mask, cmap='gray')
        axes[0, 2].set_title("Refined Mask")
        axes[0, 2].axis('off')
        axes[1, 0].imshow(extracted)
        axes[1, 0].set_title("Extracted Subject")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(new_bg)
        axes[1, 1].set_title("New Background")
        axes[1, 1].axis('off')
        axes[1, 2].imshow(final)
        axes[1, 2].set_title("Final Result")
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show() 