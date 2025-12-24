import sys
import torch
from sentence_transformers import SentenceTransformer

# Import modules
import plagiarism as general_detector
import research
import isresearchpaper
from ai_detect import analyse 
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
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    # Highlighting backgrounds
    HIGHLIGHT_PLAG = '\033[43m\033[30m' # Yellow BG, Black Text
    HIGHLIGHT_AI = '\033[31m'           # Red Text

MODEL_NAME = "all-MiniLM-L6-v2"

def initialize_models():
    print(f"{Colors.HEADER}=== SYSTEM INITIALIZATION ==={Colors.RESET}")
    print(f"{Colors.BLUE}[1/2] Loading Embedding Model...{Colors.RESET}")
    emb_model = SentenceTransformer(MODEL_NAME)
    
    print(f"{Colors.BLUE}[2/2] Loading Plagiarism Database...{Colors.RESET}")
    index, corpus, corpus_set = general_detector.load_system(emb_model)
    
    return emb_model, index, corpus, corpus_set

def visualize_document(paragraph_data, ai_report, plagiarism_matches):
    """
    Reconstructs the document text and applies color highlighting.
    AI = Red Text
    Plagiarism = Yellow Background
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*50)
    print(f"       VISUAL DOCUMENT ANALYSIS       ")
    print("="*50 + f"{Colors.RESET}\n")
    print(f"KEY: {Colors.HIGHLIGHT_AI}Red Text = AI Generated{Colors.RESET} | {Colors.HIGHLIGHT_PLAG}Yellow Highlight = Plagiarized{Colors.RESET}\n")

    # 1. Map AI scores to paragraph indices
    ai_map = {item['paragraph_index']: item['ai_probability'] for item in ai_report}
    
    # 2. Map Plagiarism chunks to paragraph indices
    plag_map = {}
    for item in plagiarism_matches:
        idx = item.get('paragraph_index')
        if idx is not None:
            if idx not in plag_map: plag_map[idx] = []
            plag_map[idx].append(item['user_chunk'])

    # 3. Iterate and Print
    for i, p in enumerate(paragraph_data):
        text = p['text']
        ai_prob = ai_map.get(i, 0)
        
        # Base color for the paragraph
        para_color = Colors.RESET
        if ai_prob > 50:
            para_color = Colors.HIGHLIGHT_AI
        
        # Apply Base Color
        display_text = f"{para_color}{text}{Colors.RESET}"
        
        # Highlighting Plagiarism (String replacement within the colored text)
        # We assume the chunks match exactly. 
        if i in plag_map:
            for chunk in plag_map[i]:
                # We wrap the chunk in the Highlight code, then reset to the Paragraph's base color
                replacement = f"{Colors.RESET}{Colors.HIGHLIGHT_PLAG}{chunk}{Colors.RESET}{para_color}"
                display_text = display_text.replace(chunk, replacement)

        print(f"{Colors.BOLD}[P{i+1}]{Colors.RESET} {display_text}\n")

def print_final_dashboard(ai_score, plag_score, citscore, is_paper, plagchecked, citation_count, issues_count=0):
    while True:
        try:
            print("\n--- Set Thresholds for Final Verdict ---")
            aiscorethresh = int(input(f"{Colors.CYAN} AI Score threshold (default 75): {Colors.RESET}") or 75)
            plagscorethresh = int(input(f"{Colors.CYAN} Plagiarism Score threshold (default 25): {Colors.RESET}") or 25)
            citscorethresh = int(input(f"{Colors.CYAN} Citation Score threshold (default 25): {Colors.RESET}") or 25)
            break
        except ValueError:
            print(f"{Colors.BG_RED} INVALID INPUT - Please enter numbers {Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.HEADER}" + "="*50)
    print(f"       FINAL DOCUMENT REPORT       ")
    print("="*50 + f"{Colors.RESET}\n")

    # 1. AI Score Visualization
    ai_color = Colors.RED if ai_score > 50 else Colors.YELLOW if ai_score > 20 else Colors.GREEN
    print(f"{Colors.BOLD}🤖 AI GENERATED CONTENT SCORE: {ai_color}{ai_score:.1f}%{Colors.RESET}")
    
    # 2. Plagiarism Score Visualization
    if plagchecked:
        plag_color = Colors.RED if plag_score > 20 else Colors.YELLOW if plag_score > 10 else Colors.GREEN
        print(f"{Colors.BOLD}📝 PLAGIARISM SCORE:          {plag_color}{plag_score:.1f}%{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}PLAGIARISM CHECK SKIPPED{Colors.RESET}")
        
    # 3. Citation Score Visualisation
    if is_paper:
        cit_color = Colors.RED if citscore > 20 else Colors.YELLOW if citscore > 10 else Colors.GREEN
        print(f"{Colors.BOLD}📚 CITATION SCORE:            {cit_color}{citscore:.1f}%{Colors.RESET}")

    # 4. Context
    print(f"\n{Colors.BOLD}📄 DOCUMENT CONTEXT:{Colors.RESET}")
    print(f"   - Type: {'Research Paper' if is_paper else 'General Document'}")
    
    if is_paper:
        cit_count_color = Colors.GREEN if citation_count > 5 else Colors.YELLOW
        print(f"   - Citations Found: {cit_count_color}{citation_count}{Colors.RESET}")
        audit_color = Colors.RED if issues_count > 0 else Colors.GREEN
        print(f"   - Citation Issues: {audit_color}{issues_count}{Colors.RESET}")

    # 5. Overall Verdict
    print(f"\n{Colors.BOLD}🔍 VERDICT:{Colors.RESET}")
    
    # AI Verdict
    if ai_score > aiscorethresh:
        print(f"   {Colors.BG_RED} HIGH PROBABILITY OF AI GENERATION {Colors.RESET}")
    else:
        print(f"   {Colors.BG_GREEN} HUMAN WRITTEN (Likely) {Colors.RESET}")
        
    # Plagiarism Verdict
    if plagchecked:
        if plag_score > plagscorethresh:
            print(f"   {Colors.BG_RED} SIGNIFICANT PLAGIARISM DETECTED {Colors.RESET}")
        else:
            print(f"   {Colors.BG_GREEN} ORIGINAL CONTENT (Likely) {Colors.RESET}")
            
    # Citation Verdict
    if is_paper:
        if citscore > citscorethresh:
            print(f"   {Colors.BG_RED} CITATION INTEGRITY ISSUES {Colors.RESET}")
        else:
            print(f"   {Colors.BG_GREEN} CITATIONS LOOK GOOD {Colors.RESET}")

    print(f"{Colors.HEADER}" + "="*50 + f"{Colors.RESET}\n")

def main():
    emb_model, index, corpus, corpus_set = initialize_models()
    
    while True:
        filepath = input(f"{Colors.BOLD}\nEnter file path (or 'exit'): {Colors.RESET}").strip()
        if filepath.lower() == 'exit': break
        
        if filepath.startswith('"') and filepath.endswith('"'):
            filepath = filepath[1:-1]
        
        try:
            # 1. Extract Data
            paragraph_data = extract_structured_data(filepath)
            full_text = "\n".join([p['text'] for p in paragraph_data])
            total_paragraphs = len(paragraph_data)
            
            # --- STEP 1: AI CHECK ---
            print(f"\n{Colors.HEADER}--- STEP 1: AI GENERATION CHECK ---{Colors.RESET}")
            ai_report = analyse(paragraph_data)
            
            if total_paragraphs > 0:
                avg_ai_prob = sum(p['ai_probability'] for p in ai_report) / total_paragraphs
            else:
                avg_ai_prob = 0.0

            # --- STEP 2: PLAGIARISM SCAN ---
            plagiarism_matches = []
            plagchecked = False
            is_paper, score, reasons = isresearchpaper.is_research_paper(filepath)
            
            should_scan_plagiarism = False
            
            if not is_paper:
                should_scan_plagiarism = True
            else:
                boolweb = input(f"{Colors.BLUE}This is a research paper. Check internet for plagiarism? (y/n): {Colors.RESET}")
                if boolweb.lower() in ['y', 'yes', '1']:
                    should_scan_plagiarism = True
            
            if should_scan_plagiarism:
                if index and corpus:
                    print(f"\n{Colors.HEADER}--- STEP 2: PLAGIARISM SCAN ---{Colors.RESET}")
                    # Passing filepath to scan_document which handles the sliding window
                    plagiarism_matches = general_detector.scan_document(filepath, index, emb_model, corpus, corpus_set)
                    plagchecked = True
                else:
                    print(f"{Colors.YELLOW}Database not loaded. Skipping plagiarism scan.{Colors.RESET}")

            plag_score = 0.0
            if total_paragraphs > 0 and plagiarism_matches:
                unique_flagged = set(m.get('paragraph_index') for m in plagiarism_matches if m.get('paragraph_index') is not None)
                plag_score = (len(unique_flagged) / total_paragraphs) * 100

            # --- STEP 3: RESEARCH PAPER ANALYSIS ---
            citation_issues = 0
            citation_count = 0
            cit_score = 0.0
            
            if is_paper:
                print(f"\n{Colors.HEADER}--- STEP 3: RESEARCH PAPER AUDIT ---{Colors.RESET}")
                citations = research.extract_bibliography(full_text)
                citation_count = len(citations)
                
                if citations:
                    print(f"{Colors.CYAN}✔ Detected {len(citations)} citations.{Colors.RESET}")
                    
                    # Optional Deep Scan
                    deep_scan_choice = input("Run Deep Citation Verification? (y/n): ").lower()
                    enable_second_layer = (deep_scan_choice == 'y')

                    citation_issues, checked_count = research.audit_paper_content(
                        paragraph_data, 
                        citations, 
                        model=emb_model, 
                        deep_scan=enable_second_layer 
                    )
                    
                    if checked_count > 0:
                        cit_score = (citation_issues / checked_count) * 100
                else:
                    print(f"{Colors.YELLOW}⚠ No bibliography found in research paper.{Colors.RESET}")
            
            # --- NEW STEP: VISUALIZE ---
            # This is the new function that prints the highlighted text
            visualize_document(paragraph_data, ai_report, plagiarism_matches)

            # --- FINAL DASHBOARD ---
            print_final_dashboard(avg_ai_prob, plag_score, cit_score, is_paper, plagchecked, citation_count, citation_issues)

        except Exception as e:
            print(f"{Colors.RED}Error processing file: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()