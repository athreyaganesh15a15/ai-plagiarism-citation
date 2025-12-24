import re
import math
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForSequenceClassification, AutoTokenizer
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
    UNDERLINE = '\033[4m'

# --- 1. Load Models GLOBALLY (Done once, not every loop) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{Colors.CYAN}Initializing AI Detection Models on {device}...{Colors.RESET}")

# --- Model A: Perplexity (GPT-2) ---
print("  Loading GPT-2 (Perplexity)...")
ppl_model_id = 'gpt2'
try:
    ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
except Exception as e:
    print(f"{Colors.RED}Error loading GPT-2: {e}{Colors.RESET}")
    exit()

# --- Model B: Classifier (RoBERTa) ---
# Note: We use the community hosted version of the OpenAI detector
print("  Loading RoBERTa (Classifier)...")
cls_model_id = 'openai-community/roberta-base-openai-detector'
try:
    cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_id).to(device)
    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_id)
except Exception as e:
    print(f"{Colors.RED}Error loading RoBERTa: {e}{Colors.RESET}")
    exit()

def calculate_perplexity(text):
    """
    Calculates how 'surprised' GPT-2 is by the text.
    Low Perplexity (<60) = Predicted easily = Likely AI.
    """
    encodings = ppl_tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)
    
    max_length = ppl_model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100 

        with torch.no_grad():
            outputs = ppl_model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    if not nlls: 
        return 0

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def get_roberta_probability(text):
    """
    Returns the probability (0.0 to 1.0) that the text is 'Fake' (AI-generated).
    """
    # Tokenize and truncate to 512 tokens (model max)
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        logits = cls_model(**inputs).logits
    
    # Apply Softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # Label 0 = Real, Label 1 = Fake (usually, depends on specific model config)
    # For openai-detector: Index 1 is usually 'Fake'
    fake_prob = probabilities[0][1].item()
    return fake_prob

def analyse(paragraph_data):
    scores = []
    print(f"Scanning {len(paragraph_data)} paragraphs...")

    for i, para in enumerate(paragraph_data):
        text = para['text']
        formatted = para['formatted_text']
        
        # Skip tiny snippets
        if len(text.split()) < 5: continue

        score = 0
        reasons = []

        # --- 1. Perplexity Check ---
        ppl = calculate_perplexity(text)
        ppl_flag = False
        
        if ppl < 60:
            score += 40
            ppl_flag = True
            reasons.append(f"Low Perplexity ({ppl:.1f})")
        elif ppl < 90:
            score += 20
            reasons.append(f"Medium Perplexity ({ppl:.1f})")

        # --- 2. RoBERTa Classifier Check ---
        fake_prob = get_roberta_probability(text)
        cls_flag = False
        
        # Convert to percentage for display
        fake_percent = fake_prob * 100
        
        if fake_percent > 90:
            score += 50
            cls_flag = True
            reasons.append(f"RoBERTa classifier 90%+ confident ({fake_percent:.1f}%)")
        elif fake_percent > 70:
            score += 30
            reasons.append(f"RoBERTa suspicious ({fake_percent:.1f}%)")

        # --- 3. ENSEMBLE LOGIC (The "And" Gate) ---
        # If Perplexity is Low AND Classifier is High, we are very sure.
        if ppl_flag and cls_flag:
            score = 99 # Force max score
            reasons.insert(0, f"{Colors.RED}★ ENSEMBLE MATCH (PPL + Classifier){Colors.RESET}")
        
        # --- 4. Heuristics (Formatting) ---
        if re.search(r'\*\*.*?\*\*:', formatted):
            score += 10
            reasons.append("Structured bold pattern")

        final_score = min(100, score)
        
        scores.append({
            "paragraph_index": i,
            "ai_probability": final_score,
            "reasons": reasons,
            "snippet": text[:60] + "...",
            "perplexity": ppl,
            "roberta_score": fake_percent
        })

    return scores

if __name__ == "__main__":
    while True:
        try:
            filepath = input(f"{Colors.BLUE}(Type 'exit' to quit) Enter file path: {Colors.RESET}")
            if filepath == 'exit': break
            
            paragraph_data = extract_structured_data(filepath)
            
            print(f"\n{Colors.YELLOW}--- Running Ensemble AI Detector ---{Colors.RESET}")
            ai_report = analyse(paragraph_data)
            
            suspicious = [r for r in ai_report if r['ai_probability'] > 50]
            
            if suspicious:
                print(f"{Colors.RED}\nWARNING: {len(suspicious)} paragraphs look AI-generated.{Colors.RESET}")
                for entry in suspicious:
                    print(f"\n{Colors.BOLD}Paragraph #{entry['paragraph_index'] + 1} (AI Score: {entry['ai_probability']:.1f}%){Colors.RESET}")
                    print(f"  Ensemble Metrics: PPL={entry['perplexity']:.1f} | RoBERTa={entry['roberta_score']:.1f}%")
                    print(f"  Reasons: {', '.join(entry['reasons'])}")
                    print(f"  Snippet: \"{entry['snippet']}\"")
            else:
                print(f"\n{Colors.GREEN}✔ No strong AI patterns detected.{Colors.RESET}")
                
        except ValueError as e:
            print(f"{Colors.RED}File Error: {e}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")