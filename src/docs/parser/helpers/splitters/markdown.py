import re
from typing import List, Dict, Any, Tuple

from src.utils.utils import encode_string_by_tiktoken, decode_tokens


class CustomChunk:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        content_preview = str(self.page_content).replace('\n', ' ')
        if len(content_preview) > 100:
            content_preview = content_preview[:100] + '...'
        return f"CustomChunk(page_content='{content_preview}', metadata={self.metadata})"


class CustomMarkdownSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def _tokens(self, content:str):
        # Implement your token counting logic here
        tokens = decode_tokens(encode_string_by_tiktoken(content, model_name="gpt-4o"))
        return tokens

    def _split_by_markdown_headers(self, markdown_text: str) -> List[Dict[str, Any]]:
        sections = []
        lines = markdown_text.split('\n')
        current_section_lines = []
        current_headers = {}

        header_pattern = re.compile(r"^(#+)\s*(.*)$")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                if current_section_lines:
                    sections.append({
                        'page_content': "\n".join(current_section_lines).strip(),
                        'metadata': current_headers.copy()
                    })
                    current_section_lines = []

                header_level = len(match.group(1))
                header_text = match.group(2).strip()

                for i in range(header_level + 1, 7):
                    current_headers.pop(f"Header {i}", None)
                current_headers[f"Header {header_level}"] = header_text

            current_section_lines.append(line)

        if current_section_lines:
            sections.append({
                'page_content': "\n".join(current_section_lines).strip(),
                'metadata': current_headers.copy()
            })

        return sections

    def _split_text_with_overlap(self, text: str) -> List[str]:
        chunks = []
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunks.append(text[start_idx:end_idx])
            if end_idx == len(text):
                break
            start_idx += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_character_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return self._split_text_with_overlap(text)

        current_separator = separators[0]
        remaining_separators = separators[1:]

        if not current_separator:
            return self._split_text_with_overlap(text)

        parts = text.split(current_separator)
        segments = []
        current_segment_parts = []
        current_segment_length = 0

        for i, part in enumerate(parts):
            part_to_add = (current_separator if i > 0 else "") + part
            if current_segment_length + len(part_to_add) > self.chunk_size:
                if current_segment_parts:
                    segments.append("".join(current_segment_parts).strip())
                current_segment_parts = [part_to_add]
                current_segment_length = len(part_to_add)
            else:
                current_segment_parts.append(part_to_add)
                current_segment_length += len(part_to_add)

        if current_segment_parts:
            segments.append("".join(current_segment_parts).strip())

        final_chunks = []
        for segment in segments:
            final_chunks.extend(self._recursive_character_split(segment, remaining_separators))

        return final_chunks

    def _map_words_to_chunk(self, words: List[Tuple[Any, ...]], chunk: str, word_index: int) -> Tuple[List, int]:
        bboxes = []
        # use word index to collect the words from the start index in the words list
        w_index = word_index
        page_words = [(w[4].lower(), w) for i, w in enumerate(words) if i >= word_index]
        chunk_words = re.findall(r'\b\w+\b', chunk.lower())
        if not chunk_words:
            return [], w_index

        first, last = chunk_words[0], chunk_words[-1]
        start, end = -1, -1
        for i, (word, _) in enumerate(page_words):
            if word == first:
                start = i
                break
        if start == -1:
            return [], w_index

        for i in range(start, len(page_words)):
            if page_words[i][0] == last:
                if start <= i <= start + len(chunk_words) * 2:
                    end = i
                    break
        if end == -1:
            return [], w_index

        for i in range(start, end + 1):
            bboxes.append(page_words[i][1][0:4])

        w_index = end + 1
        return bboxes, w_index

    def split_documents(self, pages_output: List[Dict[str, Any]]) -> List[CustomChunk]:
        chunks = []
        for page in pages_output:
            page_text = page['text']
            page_words = page['words']
            metadata = page['metadata']
            page_num = metadata.get('page', 0)

            sections = self._split_by_markdown_headers(page_text)
            for section in sections:
                section_text = section['page_content']
                section_meta = section['metadata']
                sub_chunks = self._recursive_character_split(section_text, self.separators)
                start_word_index = 0
                for sub in sub_chunks:
                    meta = {**metadata, **section_meta}
                    meta['word_bboxes'], start_word_index = self._map_words_to_chunk(page_words, sub, start_word_index)
                    chunks.append(CustomChunk(page_content=sub, metadata=meta))
        return chunks
