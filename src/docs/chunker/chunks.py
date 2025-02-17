import json
import re

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


def extract_chunks_md(data: List, max_tokens: int = 500) -> List[str]:
    def chunk_text(text: str, chunk_token_size: int) -> List[str]:
        token_count, words_list = count_tokens(text)
        chunks = []
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
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                current_chunk.append(word)
                current_token_count += word_token_count
        else:
            chunks.append(text)

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    chunks = []
    for page in data:
        page_text = page['text']
        page_chunks = chunk_text(page_text, max_tokens)
        chunks.extend(page_chunks)
    return chunks


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
