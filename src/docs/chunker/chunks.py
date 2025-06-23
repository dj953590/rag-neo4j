import json
import re
from typing import Tuple, List

from beartype.typing import List, Dict

from src.utils.utils import encode_string_by_tiktoken, decode_tokens


def count_tokens(content):
    # Implement your token counting logic here
    tokens = encode_string_by_tiktoken(content, model_name="gpt-4o")
    return len(tokens), tokens


def merge_bounding_boxes(start_box, end_box):
    return [
        start_box[0],
        start_box[1],
        end_box[2],
        end_box[3]
    ]

def extract_page_chunks_md(md_pages):
    """
    Splits markdown from PyMuPDF4LLM into token-limited chunks with positional metadata.

    Returns:
        chunks: List of markdown strings
        chunk_data: List of dicts with keys:
            - 'chunk_text': markdown string
            - 'page': page number
            - 'positions': list of bboxes for words in this chunk
    """

    chunk_data = []
    full_text: str = ""

    for page in md_pages:
        page_number = page["metadata"]["page"]
        words = page.get("words", [])
        page_chunk = page["text"]
        full_text += page_chunk
        #for word in the words get the word and the bounding boxes which is first 4 in the list
        # words are in the format [x0, y0, x1, y1, word]
        page_word_positions = {}
        for word in words:
            # word[0] is x0, word[1] is y0, word[2] is x1, word[3] is y1, word[4] is the actual word
            if len(word) >= 5:
                word_bbox = [word[0], word[1], word[2], word[3]]
                # store the word from fifth index
                page_word_positions[word[4]] = word_bbox
        # Flush the last chunk of the page
        chunk_data.append({
                "chunk_text": page_chunk.strip(),
                "page": page_number,
                "positions": page_word_positions.copy()
            })

    full_text_stripped = full_text.strip()
    return chunk_data, full_text_stripped

def extract_chunks_md(md_pages, max_tokens=1024,):
    """
    Splits markdown from PyMuPDF4LLM into token-limited chunks with positional metadata.

    Returns:
        chunks: List of markdown strings
        chunk_data: List of dicts with keys:
            - 'chunk_text': markdown string
            - 'page': page number
            - 'positions': list of bboxes for words in this chunk
    """

    chunk_data = []
    full_text: str = ""

    for page in md_pages:
        page_number = page["metadata"]["page"]
        words = page.get("words", [])
        text = page["text"]
        full_text += text

        lines = text.split("\n")
        current_chunk = ""
        current_tokens = 0
        current_positions = []

        def flush_chunk():
            nonlocal current_chunk, current_tokens, current_positions
            if not current_chunk.strip():
                return

            # Tokenize and lowercase for matching
            chunk_words = re.findall(r'\b\w+\b', current_chunk.lower())
            bbox = None

            if chunk_words:
                first_word = chunk_words[0]
                last_word = chunk_words[-1]

                matched = [(i, w) for i, w in enumerate(words) if w[4].lower() == first_word]
                first_match = matched[0][1] if matched else None

                matched = [(i, w) for i, w in enumerate(words) if w[4].lower() == last_word]
                last_match = matched[-1][1] if matched else None

                if first_match and last_match:
                    bbox = [first_match[0], first_match[1], last_match[2], last_match[3]]

            if current_chunk.strip():
                chunk_data.append({
                    "chunk_text": current_chunk.strip(),
                    "page": page_number,
                    "positions": current_positions.copy()
                })
            current_chunk = ""
            current_tokens = 0
            current_positions = []

        for line in lines:
            token_count, words_list = count_tokens(line + "\n")
            if current_tokens + token_count > max_tokens:
                flush_chunk()

            current_chunk += line + "\n"
            current_tokens += token_count

        flush_chunk()  # Final flush at page end

    full_text_stripped = full_text.strip()
    return chunk_data, full_text_stripped


def extract_chunks_md_old(data: List, max_tokens: int = 500) -> tuple[list[str], str]:
    def chunk_text(text: str, chunk_token_size: int) -> List[str]:
        token_count, words_list = count_tokens(text)
        page_chunks = []
        current_chunk = []
        current_token_count = 0
        if token_count > chunk_token_size:
            words = decode_tokens(words_list)
            word_token_count = 1
            for word in words:
                if current_token_count + word_token_count > chunk_token_size:
                    # Ensure the chunk ends at a Markdown indicator
                    while current_chunk and not re.match(r'^[#*\-]', current_chunk[-1]):
                        word = current_chunk.pop()
                        current_token_count -= 1
                    if current_chunk:
                        page_chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                current_chunk.append(word)
                current_token_count += word_token_count
        else:
            page_chunks.append(text)

        if current_chunk:
            page_chunks.append(''.join(current_chunk))

        return page_chunks

    chunks = []
    full_text: str = ""
    for page in data:
        page_text = page['text']
        full_text += page_text
        page_chunks = chunk_text(page_text, max_tokens)
        chunks.extend(page_chunks)
    return chunks, full_text.strip()


def extract_chunks(data: list, max_tokens: int, min_percentage: int, overlap_tokens: int):
    min_tokens = max_tokens * min_percentage / 100
    chunks = []
    current_chunk = []
    current_token_count = 0
    current_bounding_box = None
    full_text: str = ""

    for page in data:
        for element in page['content']:
            full_text += element['text'] + " "
            token_count = count_tokens(element['text'])
            if current_token_count + token_count > max_tokens:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "bounding_box": current_bounding_box,
                        "token_count": current_token_count
                    })
                    # Start the new chunk from the overlap point
                    overlap_start = current_token_count - overlap_tokens
                    new_chunk = []
                    new_token_count = 0
                    for e in current_chunk:
                        new_token_count += count_tokens(e['text'])
                        if new_token_count > overlap_start:
                            new_chunk.append(e)

                    current_chunk = new_chunk
                    current_token_count = sum(count_tokens(e['text']) for e in current_chunk)
                    current_bounding_box = merge_bounding_boxes(current_chunk[0]['bounding_box'], current_chunk[-1][
                        'bounding_box']) if current_chunk else None
            current_chunk.append(element)
            current_token_count += token_count
            if current_bounding_box is None:
                current_bounding_box = element['bounding_box']
            else:
                current_bounding_box = merge_bounding_boxes(current_bounding_box, element['bounding_box'])

    if current_chunk:
        chunks.append({
            "content": current_chunk,
            "bounding_box": current_bounding_box,
            "token_count": current_token_count
        })

    # Combine small chunks recursively
    combined_chunks = []
    i = 0
    while i < len(chunks):
        combined_chunk = chunks[i]
        while i + 1 < len(chunks) and combined_chunk['token_count'] < min_tokens:
            next_chunk = chunks[i + 1]
            combined_chunk['content'].extend(next_chunk['content'])
            combined_chunk['bounding_box'] = merge_bounding_boxes(combined_chunk['bounding_box'],
                                                                  next_chunk['bounding_box'])
            combined_chunk['token_count'] += next_chunk['token_count']
            i += 1
        combined_chunks.append(combined_chunk)
        i += 1

    return combined_chunks, full_text.strip()


if __name__ == "__main__":
    filepath = "../../engine/examples/caterpillar/citibank-caterpillar.json"
    # Example usage
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    max_tokens = 500  # Specify your token limit here
    min_percentage = 10  # Specify the minimum percentage of max_tokens
    overlap_tokens = 50  # Specify the number of overlap tokens
    combined_chunks, full_text = extract_chunks(data, max_tokens, min_percentage, overlap_tokens)

    for i, chunk in enumerate(combined_chunks):
        print(f"Chunk {i + 1}:")
        combined_text = " ".join(element['text'] for element in chunk['content'])
        print(combined_text)
        print(f"Bounding Box: {chunk['bounding_box']}\n")
        print(f"Token Count: {chunk['token_count']}\n")

    print("Full Text:")
    print(full_text)
