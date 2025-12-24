import os
from main import extract_structured_data, analyse, generate_html_report
import plagiarism as general_detector
import isresearchpaper
import research
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
index, corpus, corpus_set = general_detector.load_system(MODEL)

def run_full_analysis(filepath=None, pasted_text=None, allow_web=True, deep_citation=False):

    # 1. Build paragraph_data
    if filepath:
        paragraph_data = extract_structured_data(filepath)
        filename = os.path.basename(filepath)
    else:
        filename = "pasted_text.txt"
        paragraphs = pasted_text.split("\n\n")
        paragraph_data = [{"text": p.strip(), "formatted_text": p.strip()} for p in paragraphs]

    full_text = "\n".join([p["text"] for p in paragraph_data])

    # 2. AI DETECTION
    ai_report = analyse(paragraph_data)
    avg_ai = sum(p['ai_probability'] for p in ai_report) / len(paragraph_data)

    # 3. IS IT A RESEARCH PAPER?
    is_paper = False
    citation_issues = []
    citation_issue_list = []
    cit_score = 0

    if filepath and filepath.endswith(".pdf"):
        is_paper, _, _ = isresearchpaper.is_research_paper(filepath)

    # 4. PLAGIARISM
    plagiarism_matches = []

    if filepath:
        if allow_web:
            # Run full plagiarism scan (local + web)
            plagiarism_matches = general_detector.scan_document(
                filepath,
                index,
                MODEL,
                corpus,
                corpus_set
            )
        else:
            print("Skipping web plagiarism scan (User disabled web scanning).")
    else:
        print("Skipping plagiarism scan (No file uploaded — pasted text cannot be checked using web scan).")


    plag_score = 0
    if plagiarism_matches:
        unique_p = set(m["paragraph_index"] for m in plagiarism_matches)
        plag_score = len(unique_p) / len(paragraph_data) * 100

    # 5. CITATION AUDIT
    if is_paper:
        citations = research.extract_bibliography(full_text)
        if citations:
            issues, count, citation_issue_list = research.audit_paper_content(
                paragraph_data, citations, model=MODEL, deep_scan = deep_citation
            )
            if count > 0:
                cit_score = issues / count * 100

    scores = {"ai": avg_ai, "plag": plag_score, "cit": cit_score}

    # 6. GENERATE HTML REPORT
    report_name = f"report_{filename}.html"
    generate_html_report(
        filename,
        paragraph_data,
        ai_report,
        plagiarism_matches,
        citation_issue_list,
        scores
    )

    return report_name
