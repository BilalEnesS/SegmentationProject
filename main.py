from segment.background_replacer import BackgroundReplacer
from PIL import Image
import os

def main():
    """
    Example usage for BackgroundReplacer pipeline.
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    bg_replacer = BackgroundReplacer(openai_api_key=api_key)

    # Single background replacement
    results = bg_replacer.process_full_pipeline(
        image_path="image1.JPG",
        subject_prompt="human",
        background_prompt="beautiful tropical beach with palm trees and turquoise water",
        threshold=0.4,
        output_dir="output"
    )
    print("All processes completed successfully!")

if __name__ == "__main__":
    main()