import requests
import time
import re
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
import plagiarism as detector  
from read_data import extract_structured_data

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# --- Configuration ---
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_bibliography(text):
    """
    Scans bottom-up to find References and extracts citation titles.
    """
    lines = text.split('\n')
    total_lines = len(lines)
    target_headers = ["References", "Bibliography", "Works Cited", "Literature Cited"]
    
    start_index = -1
    found_header = ""
    stop_limit = int(total_lines * 0.2) 

    print(f"\n{Colors.BLUE}Scanning for references (Bottom-up)...{Colors.RESET}")

    # 1. Find Header (Bottom-up scan)
    for i in range(total_lines - 1, stop_limit, -1):
        line = lines[i].strip()
        if len(line) > 40 or len(line) < 5: continue
        
        for target in target_headers:
            if fuzz.token_set_ratio(line.lower(), target.lower()) >= 90:
                if " see " not in line.lower() and " in " not in line.lower():
                    start_index = i
                    found_header = line
                    break
        if start_index != -1: break

    if start_index == -1: 
        print(f"{Colors.YELLOW}Could not fuzzy-detect 'References' section.{Colors.RESET}")
        return []

    print(f"{Colors.GREEN}✔ Found Header: '{found_header}' (Line {start_index}){Colors.RESET}")

    # 2. Extract Lines
    raw_bib_lines = lines[start_index+1:]
    citation_titles = []
    
    for line in raw_bib_lines:
        clean = line.strip()
        if len(clean) < 10 or "page" in clean.lower(): continue
        
        # Clean "[1]" or "1."
        if clean[0] == '[' and ']' in clean:
             clean = clean.split(']', 1)[1].strip()
        elif clean[0].isdigit():
             parts = clean.split(' ', 1)
             if len(parts) > 1 and parts[0].endswith('.'):
                 clean = parts[1].strip()
        
        citation_titles.append(clean)
            
    return citation_titles

def fetch_citation_network(titles, deep_scan=False):
    """
    Fetches abstracts.
    - If deep_scan=False (Layer 1): Direct citations only.
    - If deep_scan=True (Layer 2): Direct citations + papers CITED BY them.
    """
    corpus = {}
    print(f"{Colors.BLUE}   -> Fetching citation network for {len(titles)} papers...{Colors.RESET}")
    
    headers = { "User-Agent": "CitationAudit/2.0" }

    for i, raw_title in enumerate(titles):
        if i > 14: break # Safety limit for Level 1 to save time
        
        # 1. Clean Title for Search
        clean_query = raw_title
        year_match = re.search(r'\(\d{4}\)\.\s*', raw_title)
        if year_match:
            potential_title = raw_title[year_match.end():]
            clean_query = potential_title.split('.')[0] if '.' in potential_title else potential_title
        
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        clean_query = re.sub(r'^[^a-zA-Z0-9]+', '', clean_query)
        if len(clean_query) > 80: clean_query = clean_query[:80]

        try:
            # 2. Prepare API Fields
            # We ask for 'references.title' and 'references.abstract' ONLY if deep_scan is True
            fields = "title,abstract"
            if deep_scan:
                fields += ",references.title,references.abstract"

            params = {
                "query": clean_query,
                "limit": 1,
                "fields": fields
            }
            
            # Rate limit politeness
            time.sleep(1.2) 
            
            r = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=10)
            
            if r.status_code == 200:
                data = r.json()
                if data.get('data'):
                    paper = data['data'][0]
                    found_title = paper.get('title', 'Unknown')
                    abstract = paper.get('abstract')
                    
                    # Store Level 1 (Direct Citation)
                    if abstract:
                        corpus[f"[DIRECT] {found_title}"] = abstract
                        print(f"     ✔ Found: {found_title[:40]}...")
                    
                    # Store Level 2 (The Second Layer)
                    if deep_scan and paper.get('references'):
                        refs = paper['references']
                        count = 0
                        for ref in refs:
                            if count >= 5: break # Limit to top 5 most relevant secondary refs
                            if ref.get('abstract') and ref.get('title'):
                                key = f"[SECONDARY - Cited by {found_title[:15]}...] {ref['title']}"
                                corpus[key] = ref['abstract']
                                count += 1
                        if count > 0:
                            print(f"       + Added {count} secondary citations from this paper.")
                                
        except Exception as e:
            # print(f"Error fetching {clean_query}: {e}")
            pass
            
    return corpus

def audit_paper_content(paragraph_data, citations, model=None, deep_scan=False):
    issues_list = []
    """
    Main logic to check user text against Level 1 and Level 2 abstracts.
    """
    # 1. Fetch Data
    cited_corpus = fetch_citation_network(citations, deep_scan=deep_scan)
    
    if not cited_corpus:
        print(f"{Colors.YELLOW}   -> No abstracts found.{Colors.RESET}")
        return 0, 0, []

    print(f"\n{Colors.CYAN}   -> Comparing content against {len(cited_corpus)} abstracts...{Colors.RESET}")
    
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    
    issues_found = 0
    
    # 2. Batch Encode Corpus (Optimization)
    corpus_keys = list(cited_corpus.keys())
    corpus_texts = list(cited_corpus.values())
    
    if not corpus_texts: return

    try:
        emb_corpus = model.encode(corpus_texts, convert_to_numpy=True)
        # Normalize
        emb_corpus = emb_corpus / (np.linalg.norm(emb_corpus, axis=1, keepdims=True) + 1e-9)
    except Exception as e:
        print(f"Encoding Error: {e}")
        return

    # 3. Compare User Paragraphs
    for i, p_obj in enumerate(paragraph_data):
        user_para = p_obj['text']
        if len(user_para) < 50: continue

        # Encode user para
        emb_user = model.encode([user_para], convert_to_numpy=True)
        emb_user = emb_user / (np.linalg.norm(emb_user, axis=1, keepdims=True) + 1e-9)

        # Vectorized Dot Product
        scores = np.dot(emb_user, emb_corpus.T)[0]
        
        # Check results
        for idx, score in enumerate(scores):
            title = corpus_keys[idx]
            
            # A. Exact Match (Winnowing)
            u_hashes = detector.generate_winnow_hashes(user_para)
            a_hashes = detector.generate_winnow_hashes(corpus_texts[idx])
            
            if u_hashes and a_hashes:
                 overlap = len(u_hashes.intersection(a_hashes)) / len(u_hashes)
                 if overlap > 0.15:
                      print(f"\n{Colors.RED}[CRITICAL] Text overlap in Para {i+1}{Colors.RESET}")
                      print(f"  Source: {title}")
                      issues_found += 1
                      issues_list.append({
                            "paragraph_index": i,
                            "type": "text-overlap",
                            "source_title": title,
                            "similarity": overlap
                      })
                      break

            # B. Semantic Match (Paraphrasing)
            if score > 0.75: # Similarity Threshold
                print(f"\n{Colors.RED}[MATCH DETECTED] Para {i+1} matches academic source.{Colors.RESET}")
                print(f"   Source: {title}")
                print(f"   Similarity: {score:.2f}")
                
                # Check if it is a Second Layer match
                if "[SECONDARY" in title:
                     print(f"   {Colors.YELLOW}⚠ NOTE: This is a SECONDARY source (cited by one of your refs).")
                     print(f"   Did you read the original paper, or just the citation?{Colors.RESET}")
                
                issues_found += 1 
                issues_list.append({
                    "paragraph_index": i, 
                    "type": "semantic",
                    "source_title": title,
                    "similarity": float(score)
                })
                break 

    if issues_found == 0:
        print(f"\n{Colors.GREEN}✔ Citation Audit Passed.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}Found {issues_found} potential citation issues.{Colors.RESET}")

    return issues_found, len(cited_corpus), issues_list
# Wrapper for direct execution if running this file alone
def audit_paper(filepath):
    data = extract_structured_data(filepath)
    full_text = "\n".join([d['text'] for d in data])
    citations = extract_bibliography(full_text)
    
    if citations:
        print(f"\n{Colors.HEADER}Select Scan Depth:{Colors.RESET}")
        print("1. Layer 1 (Direct Citations Only)")
        print("2. Layer 2 (Deep Scan - Includes Citations of Citations)")
        choice = input("Enter choice (1/2): ").strip()
        deep = (choice == '2')
        
        audit_paper_content(data, citations, deep_scan=deep)

if __name__ == "__main__":
    path = input("Enter research paper path: ")
    audit_paper(path)