# SegmentasyonProject

This project provides a pipeline for image segmentation and background replacement using CLIPSeg and OpenAI DALL-E. It allows you to extract a subject from an image and place it on a new, AI-generated or placeholder background.

## Features
- Segment objects in images using CLIPSeg
- Refine segmentation masks
- Extract subjects with transparent backgrounds
- Generate new backgrounds with OpenAI DALL-E or create placeholder backgrounds
- Composite subjects onto new backgrounds
- Visualize all steps of the pipeline

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd SegmentasyonProject
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the main script:
```bash
python main.py
```

Edit `main.py` to change the image path, prompts, or output directory as needed.

## Project Structure
```
SegmentasyonProject/
├── main.py
├── requirements.txt
├── README.md
├── segment/
│   ├── __init__.py
│   └── background_replacer.py
├── tests/
│   └── test_background_replacer.py
├── output/
```

## License
MIT 