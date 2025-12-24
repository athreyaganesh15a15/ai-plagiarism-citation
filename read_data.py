from pdfminer.high_level import extract_text as pdf_extract
import docx
import os
# Install: pip install pdfplumber
import pdfplumber

def extract_structured_data(path):
    """
    Reads a file and returns a list of dictionaries.
    Each dictionary represents a paragraph:
    {
        "text": "Clean text for plagiarism check",
        "formatted_text": "Text with **bold**, *italics*, and [HEADERS]"
    }
    """
    structured_data = []

    # --- PDF Handling ---
    if path.lower().endswith(".pdf"):
        # extracting formatting from PDF is very complex. 
        # We return plain text for both fields to prevent errors.
        raw_text = pdf_extract(path)
        paragraphs = raw_text.split("\n\n")
        for p in paragraphs:
            clean = p.strip().replace("\n", " ")
            if clean:
                structured_data.append({
                    "text": clean,
                    "formatted_text": clean # Placeholder for PDF
                })

    # --- TXT Handling ---
    elif path.lower().endswith(".txt"):
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        paragraphs = raw_text.split("\n\n")
        for p in paragraphs:
            clean = p.strip().replace("\n", " ")
            if clean:
                structured_data.append({
                    "text": clean,
                    "formatted_text": clean
                })

    # --- DOCX Handling (The Important Part) ---
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        
        for para in doc.paragraphs:
            clean_text = para.text.strip()
            if not clean_text:
                continue # Skip empty lines

            # Build the formatted string by checking 'runs'
            formatted_parts = []
            
            # Check for Heading Styles
            prefix = ""
            if 'Heading' in para.style.name:
                prefix = f"[{para.style.name.upper()}] "

            for run in para.runs:
                text_chunk = run.text
                
                # Apply markers for styling
                if run.bold and text_chunk.strip():
                    text_chunk = f"**{text_chunk}**"
                if run.italic and text_chunk.strip():
                    text_chunk = f"*{text_chunk}*"
                
                formatted_parts.append(text_chunk)
            
            final_formatted = prefix + "".join(formatted_parts)

            structured_data.append({
                "text": clean_text,
                "formatted_text": final_formatted
            })
    
    else:
        raise ValueError("Unsupported file type")
    
    return structured_data

# Note: We removed 'chunk_by_paragraphs' because extract_structured_data 
# now handles the chunking automatically during the read process.