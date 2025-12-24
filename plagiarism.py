import faiss
import sys
import os
import pickle
import time
import random
import re
import io
import numpy as np
import requests
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from copydetect import CodeFingerprint

# --- Configuration ---
DATA_DIR = r"d:\Project\aistudytools\AI and Plagiarism Detect\data"
HASHES_FILE = os.path.join(DATA_DIR, "winnow_hashes.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
CORPUS_FILE = os.path.join(DATA_DIR, "corpus_data.pkl")
CACHE_FILE = os.path.join(DATA_DIR, "search_cache.pkl") 

MODEL_NAME = "all-MiniLM-L6-v2"
WINNOW_K = 15
WINNOW_W = 10

# API Configuration
SERPAPI_KEY = "serpapikey here" 

# --- Global Cache Variable ---
SEARCH_CACHE = {}

# --- UI Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ==========================================
# 1. CACHING & UTILITY FUNCTIONS
# ==========================================

def load_cache():
    """Loads previous search results to save API credits."""
    global SEARCH_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                SEARCH_CACHE = pickle.load(f)
            print(f"{Colors.CYAN}[System] Loaded {len(SEARCH_CACHE)} cached search queries.{Colors.RESET}")
        except Exception:
            SEARCH_CACHE = {}

def save_cache():
    """Saves new search results to disk."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(SEARCH_CACHE, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def generate_winnow_hashes(text):
    try:
        if not text or len(text) < WINNOW_K: return set()
        
        # Suppress stdout strictly for this function
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        stream = io.StringIO(text)
        fp = CodeFingerprint(
            file="dummy.txt",
            k=WINNOW_K,
            win_size=WINNOW_W,
            fp=stream,
            filter=False
        )
        sys.stdout = original_stdout 
        return fp.hashes
    except Exception:
        sys.stdout = sys.__stdout__
        return set()

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def sliding_window(text, windowsize=15, step=10):
    words = text.split()
    chunks = []
    if len(words) < windowsize:
        return [" ".join(words)]
    for i in range(0, len(words) - windowsize + 1, step):
        chunk = " ".join(words[i : i + windowsize])
        chunks.append(chunk)
    return chunks

def calculate_token_overlap(chunk, snippet):
    chunk_tokens = set(tokenize(chunk))
    snippet_tokens = set(tokenize(snippet))
    if not chunk_tokens: return 0.0
    common_tokens = chunk_tokens.intersection(snippet_tokens)
    return len(common_tokens) / len(chunk_tokens)

# ==========================================
# 2. ADVANCED WEB FILTER LOGIC
# ==========================================

def calculate_text_entropy(chunks):
    if not chunks: return []
    if len(chunks) == 1: return chunks
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)
        dense_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        ranked_indices = dense_scores.argsort()[::-1]
        return [chunks[i] for i in ranked_indices]
    except Exception:
        return sorted(chunks, key=len, reverse=True)

def fetch_and_verify_url(url, user_chunk_hashes, user_chunk, model):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=4)
        
        if response.status_code != 200: return 0.0, f"HTTP {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header", "iframe"]):
            script.decompose()
            
        web_text = soup.get_text(separator=' ')
        web_text = ' '.join(web_text.split())
        
        if not web_text or len(web_text) < 50: return 0.0, "Empty Content"
        
        # --- Filter 1: Winnowing ---
        web_hashes = generate_winnow_hashes(web_text)
        winnow_score = 0.0
        if web_hashes and user_chunk_hashes:
            common = user_chunk_hashes.intersection(web_hashes)
            winnow_score = len(common) / len(user_chunk_hashes)

        # --- Filter 2: Semantic ---
        semantic_score = 0.0
        if model:
            web_chunks = sliding_window(web_text)[:250] 
            if web_chunks and len(web_chunks[0].strip()) > 0:
                user_emb = model.encode([user_chunk], convert_to_numpy=True)
                web_embs = model.encode(web_chunks, convert_to_numpy=True)
                
                user_norm = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-9)
                web_norm = web_embs / (np.linalg.norm(web_embs, axis=1, keepdims=True) + 1e-9)
                
                similarities = np.dot(web_norm, user_norm.T).flatten()
                if len(similarities) > 0:
                    semantic_score = float(np.max(similarities))
                
        if winnow_score > 0.15: 
            return winnow_score, "Deep Scan (Winnowing Match)"
        elif semantic_score > 0.70:
            return semantic_score, "Deep Scan (Semantic Match)"
        else:
            return max(winnow_score, semantic_score), "No Match"
        
    except requests.exceptions.Timeout:
        return 0.0, "Timeout"
    except Exception as e:
        return 0.0, "Error"

# ==========================================
# 3. INTERNET SEARCH WITH CACHING
# ==========================================

def filter_3_internet_match(chunk, model):
    if SERPAPI_KEY == "serpapikey here":
        print(f"{Colors.RED} [!] SerpApi Key missing.{Colors.RESET}")
        return None

    if chunk in SEARCH_CACHE:
        return SEARCH_CACHE[chunk]

    try:
        time.sleep(random.uniform(1.0, 1.5))

        params = {
            "engine": "google",
            "q": chunk,
            "api_key": SERPAPI_KEY,
            "num": 3 
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
             print(f"{Colors.RED}API Error: {results['error']}{Colors.RESET}")
             return None

        if "organic_results" not in results:
            SEARCH_CACHE[chunk] = None
            return None
            
        top_results = results["organic_results"]
        user_hashes = generate_winnow_hashes(chunk)
        found_links = []
        final_result = None
        
        for result in top_results:
            link = result.get('link')
            snippet = result.get('snippet', '')
            found_links.append(link)
            
            overlap_score = calculate_token_overlap(chunk, snippet)
            
            if overlap_score > 0.65:
                final_result = {
                    "type": "WEB-SNIPPET",
                    "score": overlap_score,
                    "matched_text": snippet,
                    "source": link,
                    "Flag": True
                }
                break
            
            if overlap_score > 0.25: 
                score, method_name = fetch_and_verify_url(link, user_hashes, chunk, model)
                
                if "Match" in method_name: 
                    final_result = {
                        "type": f"WEB-VERIFIED ({method_name})",
                        "score": score,
                        "matched_text": "Deep scan confirmed match.",
                        "source": link,
                        "Flag": True
                    }
                    break
        
        if not final_result and found_links:
            final_result = {
                "source": ", ".join(found_links),
                "Flag": False
            }
        
        SEARCH_CACHE[chunk] = final_result
        return final_result
                
    except Exception as e:
        print(f"Web Filter Error: {e}")
        return None

# ==========================================
# 4. EXISTING FILTERS (Local)
# ==========================================

def filter_1_exact_match(chunk, corpus_hash_set):
    chunk_hashes = generate_winnow_hashes(chunk)
    if not chunk_hashes: return None
    
    common_hashes = chunk_hashes.intersection(corpus_hash_set)
    if not common_hashes: return None 
    
    match_ratio = len(common_hashes)/len(chunk_hashes)
    if match_ratio > 0.40:
        return {
            "type": "EXACT MATCH (Local)",
            "score": match_ratio,
            "matched_text": chunk,
            "source": f"Local DB ({len(common_hashes)} hashes)"
        }

def filter_2_semantic_match(chunk, index, model, corpus, threshold=0.75):
    if not index or not model: return None

    query_vec = model.encode([chunk], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k=1)
    
    idx = indices[0][0]
    if idx >= len(corpus): return None
    
    matched_text = corpus[idx]
    match_vec = model.encode([matched_text], convert_to_numpy=True).astype("float32")
    
    cosine_score = np.dot(query_vec[0], match_vec[0]) / (np.linalg.norm(query_vec[0]) * np.linalg.norm(match_vec[0]))
    char_score = SequenceMatcher(None, chunk, matched_text).ratio()
    
    final_score = (cosine_score * 0.6) + (char_score * 0.4)

    if final_score > threshold:
        return {
            "type": "SEMANTIC MATCH",
            "score": float(final_score),
            "matched_text": matched_text,
            "source": f"Local Database (ID: {idx})"
        }
    return None

# ==========================================
# 5. SYSTEM SETUP & SCAN LOOP
# ==========================================

def load_or_create_corpus(source="wikitext"):
    if source == "wikitext":
        print("Loading 'wikitext' corpus...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            corpus = [line.strip() for line in dataset["text"] if len(line.split()) > 10]
            print(f"Corpus loaded: {len(corpus)} documents.")
            return corpus
        except Exception as e:
            print(f"{Colors.RED}Error loading wikitext: {e}{Colors.RESET}")
    return []

def build_index(corpus, model):
    print(f"Encoding {len(corpus)} sentences...")
    embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")
    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, 1024, 32, 8)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10
    return index

def load_system(model):
    if os.path.exists(INDEX_FILE) and os.path.exists(CORPUS_FILE):
        try:
            print(f"{Colors.CYAN}Loading system files...{Colors.RESET}")
            index = faiss.read_index(INDEX_FILE)
            with open(CORPUS_FILE, 'rb') as f: corpus = pickle.load(f)
            
            if os.path.exists(HASHES_FILE):
                with open(HASHES_FILE, 'rb') as f: corpus_hash_set = pickle.load(f)
            else:
                corpus_hash_set = set()
            
            load_cache()
            return index, corpus, corpus_hash_set
        except Exception: pass
    return None, None, None

def scan_document(filepath, index, model, corpus, corpus_set):
    """
    Main entry point for scanning. 
    NOW RETURNS: list of detected_matches
    """
    try:
        from read_data import extract_structured_data
        paragraph_data = extract_structured_data(filepath)
    except Exception:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        paragraph_data = [{'text': p} for p in text.split('\n\n') if len(p) > 20]

    print(f"\nScanning {len(paragraph_data)} paragraphs...\n")
    detected_matches = []
    
    for i, p_obj in enumerate(paragraph_data):
        text = p_obj['text']
        print(f"Processing Paragraph {i+1}...", end="\r")
        
        chunks = sliding_window(text)
        if not chunks: continue
        
        suspicious_web_candidates = []
        paragraph_has_match = False

        for chunk in chunks:
            # 1. Exact Match
            res = filter_1_exact_match(chunk, corpus_set)
            if res:
                print(f"\n  {Colors.RED}⚡ INSTANT MATCH: {chunk[:40]}...{Colors.RESET}")
                res['user_chunk'] = chunk
                res['paragraph_index'] = i 
                detected_matches.append(res)
                paragraph_has_match = True
                continue 

            # 2. Semantic Match
            res = filter_2_semantic_match(chunk, index, model, corpus)
            if res:
                print(f"\n  {Colors.YELLOW}🧠 AI MATCH ({res['score']:.2f}): {chunk[:40]}...{Colors.RESET}")
                res['user_chunk'] = chunk
                res['paragraph_index'] = i
                detected_matches.append(res)
                paragraph_has_match = True
            else:
                suspicious_web_candidates.append(chunk)

        # 3. Internet Match
        if not paragraph_has_match and suspicious_web_candidates:
            ranked_candidates = calculate_text_entropy(suspicious_web_candidates)
            if ranked_candidates:
                top_candidate = ranked_candidates[0]
                if len(top_candidate.split()) > 10:
                    print(f"  {Colors.BLUE}🌐 Checking Web (Top 1): {top_candidate[:30]}...{Colors.RESET}")
                    
                    result = filter_3_internet_match(top_candidate, model)
                    
                    if result and result.get("Flag") == True:
                        print(f"\n  {Colors.RED}❌ WEB MATCH ({result['score']:.2f}): {result['source']}{Colors.RESET}")
                        result['user_chunk'] = top_candidate
                        result['paragraph_index'] = i
                        detected_matches.append(result)
                    elif result:
                        print(f"\n  {Colors.GREEN}✔ PASSED. Sources checked: {result['source']}{Colors.RESET}")
                    else:
                        print(f"\n  {Colors.CYAN}ℹ No search results found on Google.{Colors.RESET}")

    save_cache()

    print(f"\n\n{Colors.BOLD}=== PLAGIARISM DETAILS ==={Colors.RESET}")
    if detected_matches:
        print(f"Found {len(detected_matches)} suspicious segments.")
        for m in detected_matches:
            print(f"\nType: {m['type']} | Score: {m['score']:.2f}")
            print(f"Source: {m['source']}")
            print(f"Text: \"{m['user_chunk'][:100]}...\"")
    else:
        print(f"{Colors.GREEN}No plagiarism detected.{Colors.RESET}")

    # --- RETURN THE DATA FOR INTEGRATOR ---
    return detected_matches

if __name__ == "__main__":
    try:
        print(f"{Colors.CYAN}Initializing Engine...{Colors.RESET}")
        model = SentenceTransformer(MODEL_NAME)
        index, corpus, corpus_set = load_system(model)

        if index is None:
            print(f"{Colors.YELLOW}System not found. Building...{Colors.RESET}")
            corpus = load_or_create_corpus('wikitext')
            if not corpus: sys.exit(1)
            index = build_index(corpus, model)
            faiss.write_index(index, INDEX_FILE)
            with open(CORPUS_FILE, 'wb') as f: pickle.dump(corpus, f)
            print("System built. Restart to cache hashes.")
            sys.exit(0)

        while True:
            path = input(f"\n{Colors.BLUE}Enter file path (or 'exit'): {Colors.RESET}")
            if path.lower() == 'exit': break
            # Note: When running directly, we just ignore the return value
            scan_document(path, index, model, corpus, corpus_set)

    except KeyboardInterrupt:
        print("\nExiting...")
        save_cache()