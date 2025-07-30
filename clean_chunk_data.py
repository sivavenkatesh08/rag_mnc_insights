import os
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 📁 Path to transcripts folder
BASE_DIR = Path(r"C:Navigate Labs\rag_mnc_insights\data\Transcripts")

# 🔨 Define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
)

def clean_text(text: str) -> str:
    """Remove empty lines and unnecessary whitespace."""
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)

def extract_metadata(file_path: Path):
    """Extract metadata such as year, quarter, and company from filename."""
    file_name = file_path.name
    pattern = r"(\d{4})-(\w+)-\d{2}-(\w+)\.txt"
    match = re.match(pattern, file_name)
    if match:
        year = int(match.group(1))
        month_str = match.group(2)[:3]
        company = match.group(3).upper()
        # Estimate quarter
        month_to_q = {
            "Jan": "Q1", "Feb": "Q1", "Mar": "Q1",
            "Apr": "Q2", "May": "Q2", "Jun": "Q2",
            "Jul": "Q3", "Aug": "Q3", "Sep": "Q3",
            "Oct": "Q4", "Nov": "Q4", "Dec": "Q4",
        }
        quarter = month_to_q.get(month_str, "Unknown")
        return {
            "company": company,
            "year": year,
            "quarter": quarter,
            "date": f"{year}-{month_str}",
            "filename": file_name
        }
    return {
        "company": file_path.parent.name,
        "filename": file_name
    }

def load_all_transcripts(base_dir: Path):
    all_chunks = []

    for company_dir in base_dir.iterdir():
        if company_dir.is_dir():
            company_name = company_dir.name

            for file_path in company_dir.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()

                cleaned_text = clean_text(raw_text)

                # 🧠 Extract metadata from filename
                filename_parts = file_path.stem.split("-")  # e.g., ['2020', 'Jan', '29', 'MSFT']
                year = filename_parts[0]
                month = filename_parts[1]

                # Convert month to quarter
                qmap = {
                    'Jan': 'Q1', 'Feb': 'Q1', 'Mar': 'Q1',
                    'Apr': 'Q2', 'May': 'Q2', 'Jun': 'Q2',
                    'Jul': 'Q3', 'Aug': 'Q3', 'Sep': 'Q3',
                    'Oct': 'Q4', 'Nov': 'Q4', 'Dec': 'Q4'
                }
                quarter = qmap.get(month, 'Unknown')

                chunks = text_splitter.create_documents(
                    [cleaned_text],
                    metadatas=[{
                        "company": company_name,
                        "filename": file_path.name,
                        "year": year,
                        "quarter": quarter
                    }]
                )
                all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = load_all_transcripts(BASE_DIR)
    print(f"✅ Total Chunks Created: {len(chunks)}")
    print("🧾 Sample chunk metadata:\n", chunks[0].metadata)
