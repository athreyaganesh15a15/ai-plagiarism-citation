import sys
import os
import torch
import datetime
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
    HIGHLIGHT_PLAG = '\033[43m\033[30m'
    HIGHLIGHT_AI = '\033[31m'

MODEL_NAME = "all-MiniLM-L6-v2"

def initialize_models():
    print(f"{Colors.HEADER}=== SYSTEM INITIALIZATION ==={Colors.RESET}")
    print(f"{Colors.BLUE}[1/2] Loading Embedding Model...{Colors.RESET}")
    emb_model = SentenceTransformer(MODEL_NAME)
    
    print(f"{Colors.BLUE}[2/2] Loading Plagiarism Database...{Colors.RESET}")
    index, corpus, corpus_set = general_detector.load_system(emb_model)
    
    return emb_model, index, corpus, corpus_set

def generate_html_report(filename, paragraph_data, ai_report, plagiarism_matches, citation_issues_list, scores):
    """
    Generates a visual HTML report.
    - AI Score Tooltip is applied to ALL text.
    - Plagiarism is a clickable link to the source.
    - Plagiarism hover shows Source + AI Score.
    """
    
    # 1. Prepare Data Maps
    ai_map = {item['paragraph_index']: item['ai_probability'] for item in ai_report}
    
    plag_map = {}
    for item in plagiarism_matches:
        idx = item.get('paragraph_index')
        if idx is not None:
            if idx not in plag_map: plag_map[idx] = []
            plag_map[idx].append(item)

    cit_map = {}
    for issue in citation_issues_list:
        idx = issue["paragraph_index"]
        if idx not in cit_map:
            cit_map[idx] = []
        cit_map[idx].append(issue)

    # 2. Build HTML Content
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Report - {os.path.basename(filename)}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; color: #2c3e50; }}
            .dashboard {{ display: flex; justify-content: space-between; margin-bottom: 30px; background: #ecf0f1; padding: 20px; border-radius: 8px; }}
            .metric {{ text-align: center; }}
            .metric-val {{ font-size: 24px; font-weight: bold; display: block; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
            
            /* Highlighting Classes */
            .ai-generated {{ color: #d63031; font-weight: 500; }} /* Red Text */
            
            /* Base wrapper for normal text hover */
            .text-wrapper {{ cursor: help; }}
            
            /* Plagiarism Link Styling */
            a.plagiarism-link {{ 
                background-color: #ffeaa7; 
                color: black; 
                border-bottom: 2px solid #fdcb6e; 
                text-decoration: none;
                cursor: pointer;
                transition: background-color 0.2s;
            }}
            
            a.plagiarism-link:hover {{
                background-color: #fdcb6e;
                text-decoration: underline;
            }}

            /* Citation Issues Styling */
            .citation-issue {{
                background-color: #ffcccc;
                border-left: 4px solid #e47c3c;
                padding-left: 4px;
            }}
            
            .cit-link{{
                color: #c0392b;
                cursor: pointer;
                font-weight: bold;
            }}
            
            .legend {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; background: #fff; }}
            .dot {{ height: 10px; width: 10px; display: inline-block; border-radius: 50%; margin-right: 5px; }}
            
            .document-content {{ line-height: 1.6; font-size: 16px; white-space: pre-wrap; }}
            .paragraph {{ margin-bottom: 15px; position: relative; padding: 5px; }}
            .paragraph:hover {{ background-color: #fafafa; }}
            .p-meta {{ font-size: 10px; color: #bdc3c7; position: absolute; left: -30px; top: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 Document Analysis Report</h1>
            <p><strong>File:</strong> {filename}<br><strong>Date:</strong> {timestamp}</p>
            
            <div class="dashboard">
                <div class="metric">
                    <span class="metric-val" style="color: {'#d63031' if scores['ai'] > 50 else '#27ae60'}">{scores['ai']:.1f}%</span>
                    <span class="metric-label">AI Score</span>
                </div>
                <div class="metric">
                    <span class="metric-val" style="color: {'#d63031' if scores['plag'] > 20 else '#27ae60'}">{scores['plag']:.1f}%</span>
                    <span class="metric-label">Plagiarism</span>
                </div>
                <div class="metric">
                    <span class="metric-val" style="color: {'#d63031' if scores['cit'] > 20 else '#27ae60'}">{scores['cit']:.1f}%</span>
                    <span class="metric-label">Citation Issues</span>
                </div>
            </div>

            <div class="legend">
                <strong>Legend:</strong>&nbsp;&nbsp; 
                <span class="dot" style="background: #d63031;"></span> <span style="color: #d63031">Red Text</span> = AI Generated &nbsp;|&nbsp; 
                <span class="dot" style="background: #ffeaa7; border: 1px solid #fdcb6e;"></span> <span style="background: #ffeaa7">Yellow Highlight</span> = Plagiarized (Click to view source)
            </div>

            <div class="document-content">
    """

    # 3. Iterate and Construct Text
    for i, p in enumerate(paragraph_data):

        text = p['text']
        ai_prob = ai_map.get(i, 0)
        
        # 1. Determine Visual Style for AI
        para_class = "paragraph"

        if i in cit_map:
            para_class += " citation-issue"

        if ai_prob > 50:
            para_class += " ai-generated"
            
        # 2. Build Base AI Tooltip (for the whole paragraph container)
        ai_tooltip_text = f"🤖 AI Probability: {ai_prob:.1f}%"
        
        # 3. Handle Plagiarism Insertion
        display_text = text
        if i in plag_map:
            # Sort matches by length (longest first) to avoid replacing inside already replaced tags
            matches = sorted(plag_map[i], key=lambda x: len(x['user_chunk']), reverse=True)
            
            for m in matches:
                chunk = m['user_chunk']
                source = m.get('source', '#').replace("'", "&#39;")
                
                # COMBINED TOOLTIP: Source + AI Score
                # We use &#10; for a line break in the tooltip
                combined_tooltip = f"Source: {source}&#10;{ai_tooltip_text}"
                
                # Create the clickable Hyperlink
                replacement = (
                    f'<a href="{source}" target="_blank" class="plagiarism-link" title="{combined_tooltip}">'
                    f'{chunk}'
                    f'</a>'
                )
                display_text = display_text.replace(chunk, replacement)

        # 4. Wrap the text. 
        # The <span> carries the default AI tooltip. 
        # The <a> tags inside display_text have their own title attribute, which takes precedence on hover.
        extra_links = ""
        if i in cit_map:
            for c in cit_map[i]:
                extra_links += (
                    f'<div><span class="cit-link" '
                    f'onclick="alert(`Source: {c["source_title"]}\\nSimilarity: {c.get("similarity", 0):.2f}`)">'
                    f'[Citation Issue]</span></div>'
                )

        final_html_chunk = (
            f'<span class="text-wrapper" title="{ai_tooltip_text}">{display_text}</span>'
            f'{extra_links}'
        )

        html += f'<div class="{para_class}"><span class="p-meta">#{i+1}</span>{final_html_chunk}</div>'

    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # 4. Save File
    output_filename = f"D:\\Project\\aistudytools\\AI and Plagiarism Detect\\reports\\report_{os.path.basename(filename)}.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"\n{Colors.GREEN}✔ HTML Report saved to: {Colors.BOLD}{os.path.abspath(output_filename)}{Colors.RESET}")

def visualize_document_console(paragraph_data, ai_report, plagiarism_matches):
    """
    Reconstructs the document text and applies color highlighting in console.
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*50)
    print(f"      VISUAL DOCUMENT ANALYSIS (CONSOLE)      ")
    print("="*50 + f"{Colors.RESET}\n")
    print(f"KEY: {Colors.HIGHLIGHT_AI}Red Text = AI Generated{Colors.RESET} | {Colors.HIGHLIGHT_PLAG}Yellow Highlight = Plagiarized{Colors.RESET}\n")

    ai_map = {item['paragraph_index']: item['ai_probability'] for item in ai_report}
    plag_map = {}
    for item in plagiarism_matches:
        idx = item.get('paragraph_index')
        if idx is not None:
            if idx not in plag_map: plag_map[idx] = []
            plag_map[idx].append(item['user_chunk'])

    for i, p in enumerate(paragraph_data):
        text = p['text']
        ai_prob = ai_map.get(i, 0)
        
        para_color = Colors.RESET
        if ai_prob > 50:
            para_color = Colors.HIGHLIGHT_AI
        
        display_text = f"{para_color}{text}{Colors.RESET}"
        
        if i in plag_map:
            for chunk in plag_map[i]:
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
            avg_ai_prob = sum(p['ai_probability'] for p in ai_report) / total_paragraphs if total_paragraphs > 0 else 0.0

            # --- STEP 2: PLAGIARISM SCAN ---
            plagiarism_matches = []
            plagchecked = False
            is_paper, _, _ = isresearchpaper.is_research_paper(filepath)
            
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
                    deep_scan_choice = input("Run Deep Citation Verification? (y/n): ").lower()
                    enable_second_layer = (deep_scan_choice == 'y')

                    citation_issues, checked_count, citation_issues_list = research.audit_paper_content(
                        paragraph_data, 
                        citations, 
                        model=emb_model, 
                        deep_scan=enable_second_layer 
                    )
                    
                    if checked_count > 0:
                        cit_score = (citation_issues / checked_count) * 100
                else:
                    print(f"{Colors.YELLOW}⚠ No bibliography found in research paper.{Colors.RESET}")
            
            # --- CONSOLE VISUALIZATION ---
            visualize_document_console(paragraph_data, ai_report, plagiarism_matches)

            # --- FINAL DASHBOARD ---
            print_final_dashboard(avg_ai_prob, plag_score, cit_score, is_paper, plagchecked, citation_count, citation_issues)

            # --- HTML EXPORT ---
            export_choice = input(f"{Colors.CYAN}Export report to HTML? (y/n): {Colors.RESET}")
            if export_choice.lower() in ['y', 'yes']:
                scores = {'ai': avg_ai_prob, 'plag': plag_score, 'cit': cit_score}
                generate_html_report(filepath, paragraph_data, ai_report, plagiarism_matches, citation_issues_list, scores)

        except Exception as e:
            print(f"{Colors.RED}Error processing file: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()