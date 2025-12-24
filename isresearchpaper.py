import re
import os
from pdfminer.high_level import extract_text

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def is_research_paper(filepath):
    """
    Analyzes a PDF to determine if it is likely a research paper.
    Returns: (is_paper (bool), score (0-100), reasons (list))
    """
    if not filepath.lower().endswith('.pdf'):
        return False, 0, ["Not a PDF file"]

    try:
        # Extract first 3 pages (usually contains Abstract, Intro) and last 2 pages (References)
        # extracting full text might be slow for huge theses
        text = extract_text(filepath)
        
        if not text:
            return False, 0, ["Could not extract text"]

        score = 0
        reasons = []
        
        # --- Feature 1: Keyword Check (Abstract/Intro) ---
        # Look for standard academic headers in the first ~2000 characters
        first_chunk = text[:3000].lower()
        
        if "abstract" in first_chunk:
            score += 25
            reasons.append("Found 'Abstract'")
        
        if "introduction" in first_chunk:
            score += 15
            reasons.append("Found 'Introduction'")
            
        if "keywords" in first_chunk:
            score += 10
            reasons.append("Found 'Keywords'")

        # --- Feature 2: Metadata Identifiers ---
        if "doi:" in first_chunk or "https://doi.org" in first_chunk:
            score += 20
            reasons.append("Found DOI Link")
            
        if "journal" in first_chunk or "conference" in first_chunk or "proceedings" in first_chunk:
            score += 10
            reasons.append("Found Journal/Conference mention")

        # --- Feature 3: References Section ---
        # Look for the references header in the last 20% of the text
        last_chunk_start = int(len(text) * 0.7)
        last_chunk = text[last_chunk_start:].lower()
        
        if "references" in last_chunk or "bibliography" in last_chunk or "works cited" in last_chunk:
            score += 30
            reasons.append("Found 'References' section")
        
        # --- Feature 4: Citation Pattern Analysis ---
        # Look for [1], [2] or (Author, Year) style
        # We check the whole text for a density of these patterns
        
        # Regex for [1], [12], [1, 2]
        numbered_citations = len(re.findall(r'\[\d+(?:,\s*\d+)*\]', text))
        
        # Regex for (Smith, 2020) or (Smith et al., 2020) - simplified
        # Looks for capitalized word, comma, 4 digits inside parens
        auth_date_citations = len(re.findall(r'\([A-Z][a-z]+(?: et al\.)?,\s*(?:19|20)\d{2}\)', text))
        
        if numbered_citations > 10:
            score += 20
            reasons.append(f"Frequent numbered citations ({numbered_citations} found)")
        elif auth_date_citations > 10:
            score += 20
            reasons.append(f"Frequent author-date citations ({auth_date_citations} found)")

        # --- Feature 5: Layout / Two-Column Detection (Heuristic) ---
        # This is hard with just raw text, but sometimes newlines imply column breaks.
        # We'll skip complex layout analysis for speed and rely on content.

        # --- Verdict ---
        # Cap score at 100
        final_score = min(100, score)
        
        is_paper = final_score >= 50
        
        return is_paper, final_score, reasons

    except Exception as e:
        return False, 0, [f"Error reading file: {str(e)}"]

if __name__ == "__main__":
    print(f"{Colors.CYAN}--- Research Paper Classifier ---{Colors.RESET}")
    while True:
        path = input(f"\n{Colors.BLUE}Enter PDF path (or 'exit'): {Colors.RESET}")
        if path.lower() == 'exit': break
        
        if not os.path.exists(path):
            print(f"{Colors.RED}File not found.{Colors.RESET}")
            continue
            
        is_paper, score, reasons = is_research_paper(path)
        
        if is_paper:
            print(f"\n{Colors.GREEN}✔ IDENTIFIED AS RESEARCH PAPER (Confidence: {score}%){Colors.RESET}")
            print(f"  Evidence: {', '.join(reasons)}")
        else:
            print(f"\n{Colors.YELLOW}⚠ Unlikely to be a research paper (Confidence: {score}%){Colors.RESET}")
            if reasons:
                print(f"  Findings: {', '.join(reasons)}")
            else:
                print(f"  Reason: Lacks standard academic structure (Abstract, References, DOI, etc.)")