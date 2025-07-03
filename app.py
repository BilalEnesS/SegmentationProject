import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from segment.background_replacer import BackgroundReplacer
from PIL import Image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load API key from environment
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
bg_replacer = BackgroundReplacer(openai_api_key=API_KEY)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type')
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        subject_prompt = request.form.get('subject_prompt', 'human')
        background_prompt = request.form.get('background_prompt', 'beautiful beach')
        # Çıktı dosya isimleri
        output_dir = app.config['OUTPUT_FOLDER']
        results = bg_replacer.process_full_pipeline(
            image_path=upload_path,
            subject_prompt=subject_prompt,
            background_prompt=background_prompt,
            threshold=0.4,
            output_dir=output_dir
        )
        # Çıktı dosya yolları
        output_files = {
            'original': upload_path,
            'mask': os.path.join(output_dir, 'refined_mask.png'),
            'extracted': os.path.join(output_dir, 'extracted_subject.png'),
            'white_bg': os.path.join(output_dir, 'white_background.jpg'),
            'new_background': os.path.join(output_dir, 'new_background.jpg'),
            'final': os.path.join(output_dir, 'final_result.jpg')
        }
        # Maskı kaydet (çünkü pipeline'da sadece refined_mask array olarak dönüyor)
        Image.fromarray(results['mask']).save(output_files['mask'])
        # Yeni arka planı kaydet
        results['new_background'].save(output_files['new_background'])
        return render_template('result.html', files=output_files)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 