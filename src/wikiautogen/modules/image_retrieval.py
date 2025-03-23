import hashlib
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
import json
from PIL import Image
import io
import torch
import clip
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import base64
from ...utils import read_article_file, read_input_file

class ImageRetriever:
    """A class to process articles and insert relevant images based on queries."""

    def __init__(self, openai_api_key, save_folder="downloaded_images", num_images=3):
        """Initialize with OpenAI API key and configuration."""
        self.openai_kwargs = {
            'api_key': openai_api_key,
            'temperature': 0.7,
            'top_p': 0.9
        }
        self.save_folder = save_folder
        self.num_images = num_images
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        os.makedirs(save_folder, exist_ok=True)
        self.serper_api_key = 'YOUR_Serper_KEY'

    def _fetch_web_images(self, urls):
        """Fetch image URLs from a list of webpage URLs."""
        image_urls = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.find_all('img'):
                    img_url = img.get('src')
                    if img_url and not img_url.lower().endswith('.svg'):
                        img_url = urljoin(url, img_url) if not img_url.startswith('http') else img_url
                        image_urls.append(img_url)
            except Exception as e:
                print(f"Error fetching images from {url}: {e}")
        return image_urls

    def _search_images(self, query):
        """Search images using Google Serper API."""
        url = "https://google.serper.dev/images"
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json().get('images', [])
            return [r['imageUrl'] for r in results[:self.num_images] 
                    if r.get('imageUrl') and not r['imageUrl'].lower().endswith('.svg')]
        except Exception as e:
            print(f"Error searching images for '{query}': {e}")
            return []

    def _fetch_wikipedia_images(self, query):
        """Retrieve and download images from Wikipedia."""
        base_url = "https://en.wikipedia.org/w/index.php"
        try:
            response = requests.get(base_url, params={"search": query}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            wiki_urls = ["https://en.wikipedia.org" + link.get('href') 
                        for link in soup.select('.mw-search-result-heading a')[:5]]
        except Exception as e:
            print(f"Error fetching Wikipedia results for '{query}': {e}")
            return []

        image_paths = []
        for url in wiki_urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.find('div', class_='mw-parser-output')
                if content:
                    for img in content.find_all('img'):
                        img_url = img.get('src')
                        if img_url and not img_url.lower().endswith('.svg'):
                            img_url = urljoin(url, img_url) if not img_url.startswith('http') else img_url
                            file_ext = '.' + img_url.split('.')[-1].lower()
                            path = os.path.join(self.save_folder, f"{url.split('/')[-1]}{file_ext}")
                            with open(path, 'wb') as f:
                                for chunk in requests.get(img_url, timeout=10).iter_content(100000):
                                    f.write(chunk)
                            image_paths.append(path)
            except Exception as e:
                print(f"Error processing Wikipedia page {url}: {e}")
        return image_paths

    def _download_image(self, url):
        """Download an image from a URL to a temporary file."""
        file_ext = os.path.splitext(url.split('?')[0])[1] or '.jpg'
        temp_path = os.path.join(self.save_folder, f"temp_{hashlib.sha256(url.encode()).hexdigest()[:10]}{file_ext}")
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return temp_path
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def _filter_images(self, image_sources, query):
        """Filter images by relevance using CLIP."""
        text = clip.tokenize([query]).to(self.device)
        image_scores = []
        temp_files = []

        for source in image_sources:
            path = source if not source.startswith('http') else self._download_image(source)
            if not path or not os.path.exists(path):
                continue
            if source.startswith('http'):
                temp_files.append(path)

            try:
                image = Image.open(path).convert("RGB")
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).item()
                    image_scores.append((path, similarity))
            except Exception as e:
                print(f"Error processing image {path}: {e}")

        return sorted(image_scores, key=lambda x: x[1], reverse=True), temp_files

    def _gpt_judge(self, query, image_paths):
        """Use GPT to judge the best image among up to three options."""
        client = OpenAI(api_key=self.openai_kwargs['api_key'])

        def path_to_base64(path):
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                image = Image.open(io.BytesIO(content))
                mime_types = {'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif', 
                            'bmp': 'image/bmp', 'webp': 'image/webp'}
                mime_type = mime_types.get(image.format.lower())
                if not mime_type:
                    print(f"Unsupported format: {image.format}")
                    return None
                encoded = base64.b64encode(content).decode('utf-8')
                return f'data:{mime_type};base64,{encoded}'
            except Exception as e:
                print(f"Error encoding {path}: {e}")
                return None

        valid_paths = [p for p in image_paths if p]  # Filter out None values
        if not valid_paths:
            return None

        images_base64 = [path_to_base64(p) for p in valid_paths]
        if not any(images_base64):
            return None

        num_images = len(valid_paths)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"I'm writing an article about '{query}'. I have {num_images} images. "
                    "Which one is more suitable to insert into the article, and is an image or video more suitable in general? "
                    "Assess helpfulness for understanding (information supplementation, visual reinforcement, theme relevance, emotional resonance). "
                    "Suggest if a better image is needed or if the current image should be repositioned to a specific sentence in the original text.\n"
                    "Format:\nBest medium for insertion: [Image 1/Image 2{'/Image 3' if num_images == 3 else ''}/Video]\n"
                    "Helpfulness assessment: [Description]\nNeed new image: [Yes/No]\nNew image description: [Description]\n"
                    "Need position change: [Yes/No]\nNew position: [Sentence]"
                )},
                *[{"type": "image_url", "image_url": {"url": img}} for img in images_base64 if img]
            ]
        }]

        try:
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=500, **self.openai_kwargs
            )
            result = response.choices[0].message.content.split('\n')
            return {
                "best_medium": next((l.split(": ")[1].strip() for l in result if l.startswith("Best medium")), "Image 1"),
                "need_new_image": any(l.split(": ")[1].strip() == "Yes" for l in result if l.startswith("Need new image")),
                "new_image_description": next((l.split(": ")[1].strip() for l in result if l.startswith("New image description")), ""),
                "need_position_change": any(l.split(": ")[1].strip() == "Yes" for l in result if l.startswith("Need position change")),
                "new_position": next((l.split(": ")[1].strip() for l in result if l.startswith("New position")), "")
            }
        except Exception as e:
            print(f"GPT error: {e}")
            return None

    def process_query(self, query, title, urls):
        """Process a query to find and select the best image."""
        print(f"Processing query: {query}, title: {title}")
        image_sources = (self._fetch_web_images(urls) + self._search_images(query) + 
                        self._fetch_wikipedia_images(query))
        if not image_sources:
            print(f"No images found for '{query}'")
            return None, None, False, None

        # Filter images
        sorted_images, temp_files = self._filter_images(image_sources, query)
        best_images = [img[0] for img in sorted_images[:3]] if sorted_images else []
        all_paths = {img[0] for img in sorted_images}

        if not best_images:
            print(f"No suitable images after filtering for '{query}'")
            return None, None, False, None

        # GPT judgment with safe slicing
        num_images = len(best_images)
        images_to_judge = best_images[:min(num_images, 3)]  # Take up to 3 images safely
        result = self._gpt_judge(query, images_to_judge)
        best_image = best_images[0] if best_images else None
        need_position_change = False
        new_position = None

        if result:
            image_map = {"Image 1": 0, "Image 2": 1, "Image 3": 2}
            best_idx = image_map.get(result["best_medium"], 0)
            best_image = best_images[best_idx] if best_idx < len(best_images) and best_images[best_idx] else best_images[0]
            if result["need_new_image"]:
                print(f"GPT suggests new image: {result['new_image_description']}")
                new_sources = self._search_images(result["new_image_description"])
                new_sorted, new_temp = self._filter_images(new_sources, result["new_image_description"])
                if new_sorted:
                    best_image = new_sorted[0][0]
                    temp_files.extend(new_temp)
                    all_paths.update(img[0] for img in new_sorted)
            need_position_change = result["need_position_change"]
            new_position = result["new_position"]
        else:
            print(f"GPT judgment failed for '{query}'. Using first image if available.")

        # Cleanup
        if best_image:
            for path in all_paths | set(temp_files):
                if path != best_image and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"Deleted: {path}")
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")

        return title, best_image, need_position_change, new_position

    def _insert_image_placeholders(self, article, target_sentences, image_info_list, position_changes):
        """Insert image placeholders into the article."""
        for i, (sentence, (name, path)) in enumerate(zip(target_sentences, image_info_list)):
            if name and path:
                target = position_changes[i] if position_changes[i] else sentence
                index = article.find(target)
                if index != -1:
                    placeholder = f"[IMAGE: {name}, PATH: {path}]"
                    article = article[:index + len(target)] + f"\n{placeholder}" + article[index + len(target):]
                else:
                    print(f"Position '{target}' not found in article.")
        return article

    def process_article(self, input_file_path, article_file_path, output_folder):
        """Process input file and insert images into the article."""
        os.makedirs(output_folder, exist_ok=True)
        save_folder = os.path.join(output_folder, "downloaded_images")
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

        queries, target_sentences, titles, url_lists = read_input_file(input_file_path)
        article = read_article_file(article_file_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_query, q, t, u) 
                    for q, t, u in zip(queries, titles, url_lists)]
            results = [f.result() for f in tqdm(concurrent.futures.as_completed(futures), 
                                                total=len(futures), desc="Processing queries")]

        image_info_list = [(r[0], r[1]) for r in results]
        position_changes = [r[3] if r[2] else None for r in results]
        ordered_image_info = [None] * len(target_sentences)
        for i, title in enumerate(titles):
            for info in image_info_list:
                if info[0] == title:
                    ordered_image_info[i] = info
                    break

        new_article = self._insert_image_placeholders(article, target_sentences, ordered_image_info, position_changes)
        output_path = os.path.join(output_folder, "multi_article.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_article)
        return output_path
