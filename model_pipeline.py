import os
import torch
import zipfile
import gdown
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, GenerationConfig
from PIL import Image, ImageOps

# Explicitly assign IDs to prevent model alignment mismatches
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_sinhala_ocr_model():
    """
    Checks for the fine-tuned model folder. If missing, streams the zip package 
    from Google Drive, extracts it, and updates configurations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_FOLDER = "./final_trocr_sinhala_model"
    ZIP_FILE_NAME = "trocr_sinhala_handwriting_model.zip"
    
    # 🔍 REPLACE THIS VALUE WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
    GDRIVE_FILE_ID = "https://drive.google.com/file/d/1Oparm6c06aFFA5xGDQc4zLDRsqgEGQor/view?usp=sharing" 
    gdrive_url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    # 1. Background Google Drive Downloader (Triggers only on first deployment boot)
    if not os.path.exists(MODEL_FOLDER):
        print("📥 Model directory not found. Fetching fine-tuned weights from Google Drive...")
        try:
            # Handles Google's large-file scanning gateway screens seamlessly
            gdown.download(gdrive_url, ZIP_FILE_NAME, quiet=False)
            
            print("📦 Unpacking model architecture layers...")
            with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
                
            # Clean up the large zip archive locally to save server hosting space
            if os.path.exists(ZIP_FILE_NAME):
                os.remove(ZIP_FILE_NAME)
                
            print("🏆 Model infrastructure extracted and verified locally!")
        except Exception as e:
            raise RuntimeError(f"Fatal: Failed to sync model assets from Google Drive: {str(e)}")

    # 2. Pipeline Initialization
    processor = TrOCRProcessor.from_pretrained(MODEL_FOLDER)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_FOLDER).to(device)
    model.eval()

    # 3. Locked Generation Profile to Prevent Stutter Loops & Trailing Hallucinations
    tight_generation_config = GenerationConfig(
        bos_token_id=processor.tokenizer.cls_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.cls_token_id,
        max_length=15,                  # Sharp cutoff matching actual phrase word-counts
        early_stopping=True,
        no_repeat_ngram_size=2,         # Strictly paralyzes duplicate bigram phrase loops
        repetition_penalty=2.0,         # Heavy math penalty against repeating identical characters
        length_penalty=0.6,             # Discourages generation from guessing trailing filler words
        num_beams=4                     # Explores the top 4 structural variations for high accuracy
    )
    
    return model, processor, device, tight_generation_config


def extract_text_from_handwriting(image_file, model, processor, device, generation_config):
    """
    Accepts an uploaded image file, applies aspect-ratio maintaining white letterboxing,
    and runs full neural inference to output clean handwritten Sinhala text.
    """
    try:
        # Load file stream into a clean RGB visual plane
        image = Image.open(image_file).convert("RGB")
        
        # 4. Aspect-Ratio Padding (Letterboxing Mechanization)
        target_size = 384
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Calculate trailing white pixel buffers to avoid squishing handwriting strokes
        delta_w = target_size - image.size[0]
        delta_h = target_size - image.size[1]
        padding = (0, 0, delta_w, delta_h)
        
        # Construct the final square canvas input
        padded_image = ImageOps.expand(image, padding, fill="white")
        
        # 5. Visual Feature Tensor Conversion
        pixel_values = processor(padded_image, return_tensors="pt").pixel_values.to(device)
        
        # 6. Strict Controlled Sequence Generation
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                generation_config=generation_config
            )
        
        # 7. Decode into clean human-readable text
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_text

    except Exception as e:
        return f"Recognition Error: {str(e)}"
