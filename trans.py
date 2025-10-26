#!/usr/bin/env python3
"""
Bato.to Manga Translator
Downloads images from a chapter, extracts text, translates (AI or manual), cleans and replaces text.
Supports any target language via Gemini AI or manual input.
"""

import requests
import json
import re
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
import google.generativeai as genai
from deep_translator import GoogleTranslator  # Fallback translator

class BatoTranslator:
    def __init__(self, gemini_api_key=None, target_language='marathi', manual_mode=False):
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://bato.to/',
            'DNT': '1',
        })
        
        # AI Setup
        self.gemini = None
        if gemini_api_key and not manual_mode:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai.GenerativeModel('gemini-pro')
        
        self.target_language = target_language.lower()
        self.manual_mode = manual_mode
        self.translation_context = []
        self.base_output_dir = Path("translated_chapters")
        self.base_output_dir.mkdir(exist_ok=True)
    
    def extract_chapter_id(self, url):
        """Extract chapter ID from URL"""
        match = re.search(r'/chapter/(\d+)', url)
        return match.group(1) if match else None
    
    def get_all_image_urls(self, url):
        """Get ALL image URLs from the page using improved extraction"""
        print(f"\n🔍 Fetching page: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            html = response.text
            print(f"✅ Page loaded ({len(html)} bytes)")
            
            # Multiple extraction methods
            all_urls = set()
            
            # Method 1: Find image tags
            img_tags = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            all_urls.update(img_tags)
            
            # Method 2: Find data-src attributes
            data_src = re.findall(r'data-src=["\']([^"\']+)["\']', html, re.IGNORECASE)
            all_urls.update(data_src)
            
            # Method 3: Find all image URLs in JavaScript
            js_images = re.findall(r'["\']https?://[^"\']*\.(?:jpg|jpeg|png|webp|gif)[^"\']*["\']', html, re.IGNORECASE)
            for match in js_images:
                url_clean = match.strip('"\'')
                all_urls.add(url_clean)
            
            # Method 4: Find URLs without quotes
            direct_urls = re.findall(r'https?://[^\s<>"\']+\.(?:jpg|jpeg|png|webp|gif)[^\s<>"\']*', html, re.IGNORECASE)
            all_urls.update(direct_urls)
            
            # Filter for manga images
            manga_images = []
            for img_url in all_urls:
                if not img_url.startswith('http'):
                    continue
                    
                # Filter out icons, logos, tiny images
                if any(skip in img_url.lower() for skip in ['icon', 'logo', 'avatar', 'banner', 'button', 'thumb']):
                    continue
                
                # Must be reasonably long URL (manga CDN urls are long)
                if len(img_url) > 40:
                    manga_images.append(img_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_images = [url for url in manga_images if not (url in seen or seen.add(url))]
            
            print(f"✅ Found {len(unique_images)} manga images")
            
            # Debug: show first few URLs
            if unique_images:
                print("\n📋 Sample URLs:")
                for i, url in enumerate(unique_images[:3], 1):
                    print(f"  {i}. {url[:80]}...")
            
            return unique_images
            
        except Exception as e:
            print(f"❌ Error fetching page: {e}")
            return []
    
    def download_image(self, img_url, save_path):
        """Download a single image"""
        try:
            headers = self.session.headers.copy()
            headers['Referer'] = 'https://bato.to/'
            
            response = self.session.get(img_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"  ❌ HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def extract_text_tesseract(self, image_path):
        """Extract text regions using Tesseract OCR with improved preprocessing"""
        img = cv2.imread(str(image_path))
        if img is None:
            return [], None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing methods for better OCR
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        text_regions = []
        for thresh in [thresh1, thresh2, gray]:
            try:
                data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    conf = int(data['conf'][i])
                    text = data['text'][i].strip()
                    if conf > 20 and len(text) > 0:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        if w > 10 and h > 10:
                            # Avoid duplicates
                            duplicate = any(
                                abs(existing['x'] - x) < 10 and abs(existing['y'] - y) < 10 and existing['text'] == text
                                for existing in text_regions
                            )
                            if not duplicate:
                                text_regions.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h})
            except:
                continue
        
        return text_regions, img
    
    def translate_batch(self, texts):
        """Translate batch using AI or manual input"""
        if not texts:
            return []
        
        if self.manual_mode:
            print("\n📝 Manual translation mode. Enter translations for each text:")
            translations = []
            for text in texts:
                print(f"Original: {text}")
                trans = input("Translation: ").strip()
                translations.append(trans or text)  # Fallback to original if empty
            return translations
        
        if self.gemini:
            return self.translate_batch_gemini(texts)
        else:
            return self.translate_batch_fallback(texts)
    
    def translate_batch_gemini(self, texts):
        """Translate using Gemini AI"""
        context = "\n".join(self.translation_context[-10:]) if self.translation_context else ""
        
        prompt = f"""You are a professional manga translator. Translate these English texts to {self.target_language.capitalize()}.

Previous context:
{context}

Rules:
- Keep dialogue natural and conversational
- Preserve character personality
- For sound effects, use equivalents or transliterate
- Maintain emotional tone

English texts:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Return ONLY {self.target_language.capitalize()} translations, one per line, same order. No numbering, no explanations."""
        
        try:
            response = self.gemini.generate_content(prompt)
            result = response.text.strip()
            translations = []
            for line in result.split('\n'):
                line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                line = re.sub(r'^[-*]\s*', '', line)
                if line:
                    translations.append(line)
            
            for eng, trans in zip(texts, translations):
                self.translation_context.append(f"{eng}→{trans}")
            
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])
            
            return translations[:len(texts)]
            
        except Exception as e:
            print(f"    ⚠️ Gemini error: {e}, using fallback")
            return self.translate_batch_fallback(texts)
    
    def translate_batch_fallback(self, texts):
        """Fallback translation using Google Translate"""
        lang_code = {'marathi': 'mr', 'hindi': 'hi', 'spanish': 'es', 'french': 'fr'}.get(self.target_language, 'mr')
        translator = GoogleTranslator(source='en', target=lang_code)
        translations = []
        for t in texts:
            try:
                translations.append(translator.translate(t))
            except:
                translations.append(t)
            time.sleep(0.5)
        return translations
    
    def get_font(self, size):
        """Get suitable font for target language (platform-agnostic)"""
        fonts = [
            "NirmalaS.ttf", "Nirmala.ttf", "mangal.ttf",
            "NotoSansDevanagari-Bold.ttf", "NotoSansDevanagari-Regular.ttf",
            "DevanagariSangamMN.ttc",
        ]
        font_paths = [
            Path("/usr/share/fonts/truetype/noto/"),
            Path("C:\\Windows\\Fonts\\"),
            Path("/System/Library/Fonts/Supplemental/"),
        ]
        
        for base in font_paths:
            for font in fonts:
                font_path = base / font
                if font_path.exists():
                    try:
                        return ImageFont.truetype(str(font_path), size)
                    except:
                        pass
        
        print("⚠️ Using default font (may not support all languages)")
        return ImageFont.load_default()
    
    def clean_and_replace(self, img, text_regions, translations):
        """Clean original text and replace with translated text"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        replaced_count = 0
        
        for region, translation in zip(text_regions, translations):
            try:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Expand cleaning area
                padding = 8
                clean_x1 = max(0, x - padding)
                clean_y1 = max(0, y - padding)
                clean_x2 = min(img_pil.width, x + w + padding)
                clean_y2 = min(img_pil.height, y + h + padding)
                
                # Sample background color
                try:
                    sample_region = img_pil.crop((clean_x1, clean_y1, clean_x2, clean_y2))
                    pixels = np.array(sample_region)
                    bg_color = tuple(np.median(pixels.reshape(-1, 3), axis=0).astype(int))
                except:
                    bg_color = (255, 255, 255)
                
                # Clean area
                draw.rectangle([clean_x1, clean_y1, clean_x2, clean_y2], fill=bg_color)
                
                # Font setup
                font_size = max(int(h * 0.75), 14)
                font = self.get_font(font_size)
                
                # Text color based on background
                brightness = sum(bg_color) / 3
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                
                # Draw text
                text_x = x + 3
                text_y = y + 2
                draw.text((text_x, text_y), translation, font=font, fill=text_color)
                replaced_count += 1
                
            except Exception as e:
                print(f"    ⚠️ Error replacing text: {e}")
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), replaced_count
    
    def process_image(self, img_path, index, total):
        """Process a single image"""
        print(f"\n📄 [{index}/{total}] Processing: {img_path.name}")
        
        # OCR
        print("  🔍 Extracting text...")
        text_regions, img = self.extract_text_tesseract(img_path)
        
        if not text_regions:
            print("  ⚠️ No text found")
            return img, 0
        
        print(f"  ✅ Found {len(text_regions)} text regions")
        
        texts = [r['text'] for r in text_regions]
        print(f"  📝 Sample text: {texts[0] if texts else 'none'}")
        
        # Translate
        print(f"  🌐 Translating to {self.target_language.capitalize()}...")
        translations = self.translate_batch(texts)
        
        if translations:
            print(f"  ✅ Sample translation: {translations[0]}")
        
        # Replace
        print("  🎨 Replacing text...")
        processed_img, replaced_count = self.clean_and_replace(img, text_regions, translations)
        
        print(f"  ✅ Replaced {replaced_count} text regions")
        
        return processed_img, replaced_count
    
    def run(self, bato_url):
        """Main processing pipeline"""
        print("\n" + "="*70)
        print(f"🌸 BATO.TO → {self.target_language.upper()} TRANSLATOR {'(Manual Mode)' if self.manual_mode else '(AI Powered)'} 🌸")
        print("="*70)
        
        chapter_id = self.extract_chapter_id(bato_url)
        if not chapter_id:
            print("❌ Invalid URL")
            return
        
        print(f"\n📖 Chapter ID: {chapter_id}")
        
        chapter_dir = self.base_output_dir / f"chapter_{chapter_id}"
        chapter_dir.mkdir(exist_ok=True)
        print(f"📁 Output: {chapter_dir}")
        
        image_urls = self.get_all_image_urls(bato_url)
        
        if not image_urls:
            print("\n❌ No images found!")
            print("💡 Manual mode: Paste image URLs (one per line, empty line to finish):")
            manual = []
            while True:
                url = input().strip()
                if not url:
                    break
                if url.startswith('http'):
                    manual.append(url)
            image_urls = manual
        
        if not image_urls:
            print("❌ No images!")
            return
        
        print(f"\n🚀 Processing {len(image_urls)} images...")
        
        # Download
        downloaded = []
        print("\n⬇️ Downloading images...")
        for i, url in enumerate(image_urls, 1):
            print(f"  [{i}/{len(image_urls)}] Downloading...", end=' ')
            save_path = chapter_dir / f"original_{i:03d}.jpg"
            if self.download_image(url, save_path):
                print("✅")
                downloaded.append(save_path)
            else:
                print("❌")
            time.sleep(0.5)
        
        print(f"\n✅ Downloaded {len(downloaded)}/{len(image_urls)} images")
        
        if not downloaded:
            print("❌ No images downloaded!")
            return
        
        # Process
        print("\n🔄 Processing images...")
        processed_count = 0
        total_replacements = 0
        
        for i, img_path in enumerate(downloaded, 1):
            try:
                processed_img, replacements = self.process_image(img_path, i, len(downloaded))
                output_path = chapter_dir / f"translated_{i:03d}.jpg"
                cv2.imwrite(str(output_path), processed_img)
                processed_count += 1
                total_replacements += replacements
                time.sleep(1 if self.gemini else 0.5)
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print(f"📊 Processed: {processed_count}/{len(downloaded)} images")
        print(f"📝 Total text replaced: {total_replacements}")
        print(f"📁 Output: {chapter_dir.absolute()}")
        print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Bato.to Manga Translator")
    parser.add_argument("url", nargs="?", help="Bato.to chapter URL")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--lang", default="marathi", help="Target language (default: marathi)")
    parser.add_argument("--manual", action="store_true", help="Manual translation input mode")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not args.manual and not api_key:
        api_key = input("🔑 Gemini API key: ").strip()
    
    url = args.url or input("🔗 Bato.to URL: ").strip()
    
    translator = BatoTranslator(api_key, target_language=args.lang, manual_mode=args.manual)
    translator.run(url)