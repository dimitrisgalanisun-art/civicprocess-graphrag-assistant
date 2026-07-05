import json
import re
from pathlib import Path

from src.config import RAW_MARKDOWN_FILE, QA_JSONL_FILE, validate_private_paths


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def read_markdown(path: Path) -> str:
    """Read the private markdown source file."""
    return path.read_text(encoding="utf-8")


def split_markdown_sections(markdown: str) -> list[dict]:
    """Split markdown into heading-based sections."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(heading_pattern.finditer(markdown))
    sections = []

    if not matches:
        return [
            {
                "section_title": "Document",
                "heading_level": 1,
                "content": markdown.strip(),
            }
        ]

    for index, match in enumerate(matches):
        heading_level = len(match.group(1))
        section_title = match.group(2).strip()

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)

        content = markdown[start:end].strip()

        if content:
            sections.append(
                {
                    "section_title": section_title,
                    "heading_level": heading_level,
                    "content": content,
                }
            )

    return sections


def clean_markdown_text(text: str) -> str:
    """
    Clean raw markdown into readable plain text.

    The goal is not to destroy technical identifiers.
    We keep table names, field names, and process names,
    but remove markdown noise.
    """
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Skip code fence markers but not code content itself.
        if line.startswith("```"):
            continue

        # Remove markdown heading marks.
        line = re.sub(r"^#{1,6}\s*", "", line)

        # Remove markdown bullets and numbering.
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)

        # Remove bold/italic markdown markers.
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("*", "")

        # Remove backticks but preserve the identifier inside them.
        line = re.sub(r"`([^`]+)`", r"\1", line)

        # Skip markdown table separator rows.
        if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", line):
            continue

        # Flatten markdown table rows into readable text.
        if "|" in line and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            line = " · ".join(cells)

        # Normalize whitespace.
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Reduce excessive newlines.
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split cleaned text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Try not to cut the final sentence too badly.
        if end < len(text):
            last_period = max(
                chunk.rfind("."),
                chunk.rfind(";"),
                chunk.rfind("·"),
                chunk.rfind("\n"),
            )

            if last_period > int(chunk_size * 0.6):
                chunk = chunk[: last_period + 1].strip()
                end = start + last_period + 1

        if chunk:
            chunks.append(chunk)

        next_start = max(end - overlap, start + 1)

        if next_start <= start:
            break

        start = next_start

    return chunks


def extract_technical_terms(text: str) -> list[str]:
    """
    Extract useful technical terms.

    These terms are used for both question generation and graph keywords.
    """
    terms = []

    # Terms inside backticks.
    terms.extend(re.findall(r"`([^`]{2,80})`", text))

    # Snake_case database identifiers.
    terms.extend(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]+_[a-zA-Z0-9_]+\b", text))

    # Common table-like identifiers without backticks.
    terms.extend(re.findall(r"\b[a-zA-Z]{3,}_[a-zA-Z0-9_]{3,}\b", text))

    # Clean and deduplicate.
    cleaned_terms = []

    for term in terms:
        term = term.strip().strip(".,;:()[]{}")
        if not term:
            continue

        if len(term) < 3:
            continue

        if term not in cleaned_terms:
            cleaned_terms.append(term)

    return cleaned_terms[:20]


def contains_any(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in text."""
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def generate_content_questions(section_title: str, chunk: str, raw_chunk: str) -> list[str]:
    """
    Generate content-based retrieval questions.

    Important:
    The first question must describe the actual chunk content,
    not the broad section or module name.
    """
    questions = []
    combined = chunk
    lowered = combined.lower()

    technical_terms = extract_technical_terms(combined)

    def add(question: str) -> None:
        question = re.sub(r"\s+", " ", question).strip()
        if question and question not in questions:
            questions.append(question)

    # 1. Highly specific process questions first.

        if contains_any(
        combined,
        [
            "πίνακας κατάταξης",
            "κατάταξη",
            "σειρά κατάταξης",
            "αποτέλεσμα κατάταξης",
            "ranking",
            "rank",
        ],
    ):
        add("Τι είναι ο πίνακας κατάταξης;")
        add("Πώς δημιουργείται ο πίνακας κατάταξης;")
        add("Πώς συνδέεται μια αίτηση με τον πίνακα κατάταξης;")

    if contains_any(
        combined,
        [
            "αξιολόγηση αίτησης",
            "αξιολόγηση",
            "evaluation",
            "κριτήρια επιλεξιμότητας",
            "κριτήριο βαθμολόγησης",
        ],
    ):
        add("Πώς γίνεται η αξιολόγηση αίτησης;")
        add("Ποια κριτήρια χρησιμοποιούνται στην αξιολόγηση;")

    if contains_any(
        combined,
        [
            "δικαιολογητικό",
            "δικαιολογητικά",
            "έγγραφο",
            "έγγραφα",
            "document",
            "έγκυρα",
            "μη έγκυρα",
        ],
    ):
        add("Πώς ελέγχονται τα δικαιολογητικά μιας αίτησης;")

    if contains_any(
        combined,
        [
            "υποβολή αίτησης",
            "υποβάλλουν",
            "υποβλημένη",
            "πρόχειρη",
            "αίτηση",
            "application",
        ],
    ):
        add("Πώς υποβάλλεται μια αίτηση;")
        add("Πότε θεωρείται οριστικοποιημένη μια αίτηση;")

    if contains_any(
        combined,
        [
            "υποβολή αίτησης",
            "υποβάλλουν",
            "υποβλημένη",
            "πρόχειρη",
            "αίτηση",
            "application",
        ],
    ):
        add("Πώς υποβάλλεται μια αίτηση δημότη;")
        add("Τι είναι η αίτηση δημότη;")
        add("Ποιες καταστάσεις μπορεί να έχει μια αίτηση;")
        add("Πότε κλειδώνει μια αίτηση;")
        add("How does a citizen application work?")

    if contains_any(
        combined,
        [
            "πρόγραμμα",
            "πρόσκληση",
            "κύκλος αιτήσεων",
            "κατάταξη",
            "οριστικοποίηση προγράμματος",
        ],
    ):
        add("Τι είναι το πρόγραμμα αιτήσεων;")
        add("Πώς ορίζεται ένα πρόγραμμα αιτήσεων;")
        add("Ποιες παράμετροι έχει μια πρόσκληση αιτήσεων;")
        add("Πότε απαιτείται πίνακας κατάταξης σε ένα πρόγραμμα;")
        add("How is an application program defined?")

    if contains_any(combined, ["κατάσταση", "status", "οριστική", "πρόχειρη", "προτεινόμενη"]):
        add("Ποιες καταστάσεις μπορεί να έχει μια αίτηση;")
        add("Πώς αλλάζει η κατάσταση μιας αίτησης;")
        add("Τι σημαίνει πρόχειρη, προτεινόμενη ή οριστική κατάσταση;")
        add("What statuses can an application have?")

    # 2. Table-specific questions, but only for real technical identifiers.
    # Avoid broad Greek labels like "Αιτήσεις Δημοτών".
    for term in technical_terms:
        if "_" not in term:
            continue

        add(f"Τι είναι το {term};")
        add(f"Πώς χρησιμοποιείται το {term};")
        add(f"Με ποιες οντότητες σχετίζεται το {term};")
        add(f"What is {term}?")

        if len(questions) >= 12:
            break

    # 3. Fallback only if nothing specific matched.
    if not questions:
        first_sentence = get_first_sentence(chunk)

        if first_sentence:
            add(f"Τι περιγράφει αυτό το τμήμα για {first_sentence[:100]};")
        else:
            add("Τι περιγράφει αυτό το τμήμα της τεκμηρίωσης;")

    return questions[:12]


def get_first_sentence(text: str) -> str:
    """Return a short first sentence from a text chunk."""
    parts = re.split(r"(?<=[.!;])\s+", text.strip())

    for part in parts:
        part = part.strip()

        if len(part) > 20:
            return part

    return text.strip()[:150]


def build_embedding_text(questions: list[str], answer: str, technical_terms: list[str]) -> str:
    """
    Build the text that will be embedded.

    Keep it focused. Too many generic questions make unrelated chunks look similar.
    """
    focused_questions = questions[:5]

    focused_terms = [
        term for term in technical_terms
        if "_" in term
    ][:12]

    questions_text = "\n".join(focused_questions)
    terms_text = ", ".join(focused_terms)

    return f"""
Search questions:
{questions_text}

Technical identifiers:
{terms_text}

Content:
{answer}
""".strip()


def build_records(markdown: str) -> list[dict]:
    """Build improved JSONL records from the markdown document."""
    sections = split_markdown_sections(markdown)
    records = []
    record_counter = 1

    for section in sections:
        section_title = section["section_title"]
        heading_level = section["heading_level"]

        raw_content = section["content"]
        cleaned_content = clean_markdown_text(raw_content)

        if not cleaned_content:
            continue

        chunks = chunk_text(cleaned_content)

        for chunk_index, chunk in enumerate(chunks, start=1):
            technical_terms = extract_technical_terms(chunk)
            questions = generate_content_questions(
                section_title=section_title,
                chunk=chunk,
                raw_chunk=chunk,
            )

            record = {
                "id": f"municipality_record_{record_counter:04d}",
                "question": questions[0],
                "questions": questions,
                "answer": chunk,
                "source": RAW_MARKDOWN_FILE.name,
                "section_title": section_title,
                "heading_level": heading_level,
                "chunk_index": chunk_index,
                "technical_terms": technical_terms,
                "text_for_embedding": build_embedding_text(
                    questions=questions,
                    answer=chunk,
                    technical_terms=technical_terms,
                ),
            }

            records.append(record)
            record_counter += 1

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def preview_records(records: list[dict], limit: int = 3) -> None:
    """Print a small preview."""
    print("\nPreview records")
    print("=" * 80)

    for record in records[:limit]:
        print(f"ID: {record['id']}")
        print(f"Question: {record['question']}")
        print("Alternative questions:")
        for question in record["questions"][:5]:
            print(f"  - {question}")
        print(f"Section: {record['section_title']}")
        print(f"Answer preview: {record['answer'][:300]}...")
        print("-" * 80)


def main() -> None:
    """Build the private JSONL dataset."""
    validate_private_paths()

    markdown = read_markdown(RAW_MARKDOWN_FILE)
    records = build_records(markdown)

    write_jsonl(records, QA_JSONL_FILE)

    print(f"Source markdown: {RAW_MARKDOWN_FILE}")
    print(f"Output JSONL: {QA_JSONL_FILE}")
    print(f"Records written: {len(records)}")

    preview_records(records)


if __name__ == "__main__":
    main()