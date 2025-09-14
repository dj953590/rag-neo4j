import asyncio
import csv
import html
import io
import json
import numpy as np
import os
import re
import tiktoken
from dataclasses import dataclass
from functools import wraps
from hashlib import md5, sha256
from typing import Any, List, Union
from src.utils.log import logger

ENCODER = None
PROJECT_BASE = os.getenv("RAG_PROJECT_BASE") or os.getenv("RAG_DEPLOY_BASE")
RAG_BASE = os.getenv("RAG_BASE")


def get_project_base_directory(*args):
    """
    Return the project base directory.
        param args: The path to join with the project base directory.
        return: PROJECT_BASE
    """
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def get_rag_directory(*args):
    """
    Return the RAG base directory.
        param args: The path to join with the RAG base directory.
        return: RAG_BASE
    """
    global RAG_BASE
    if RAG_BASE is None:
        RAG_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
    if args:
        return os.path.join(RAG_BASE, *args)
    return RAG_BASE


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class EmbeddingFunc:
    """
    A function to compute embeddings for a given input.

    Args:
        embedding_dim (int): The dimensionality of the embeddings.
        max_token_size (int): The maximum token size for the embedding.
        func (callable): The function to compute embeddings.
        concurrent_limit (int): The maximum number of concurrent calls.
    """
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 4

    def __post_init__(self):
        """
        Initialize the semaphore for limiting concurrent calls.
        If concurrent_limit is 0, unlimited concurrent calls are allowed.
        """
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Call the underlying function with the provided arguments and await its result.
        Use a semaphore to limit concurrent calls if concurrent_limit is not 0.
        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            np.ndarray: The computed embeddings.
        """
        async with self._semaphore:
            return await self.func(*args, **kwargs)


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + sha256(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """
    Add restriction of maximum async calling times for a async func

    Args:
        max_size (int): Maximum async calling times.
        waitting_time (float): Time to wait until next call.The limit_async_func_call function takes two parameters:
        max_size (the maximum number of concurrent calls allowed) and waitting_time (the time to wait before retrying

        if the maximum size is reached). Inside the limit_async_func_call function, a nested function final_decro is
        defined. This function will be the actual decorator. Inside final_decro, a class variable __current_size is
        initialized to 0. This variable keeps track of the current number of concurrent calls.
        4.The wait_func function is defined within final_decro. This function is the decorated function that will be
        called instead of the original function.Inside wait_func, a while loop checks if the current size is greater
        than or equal to the maximum size. If it is, the function waits for the specified waitting_time using
        await asyncio.sleep(waitting_time). Once the while loop finishes, the current size is incremented by 1,
        indicating that a new concurrent call has started. The decorated function is then called with the provided
        arguments and the result is awaited. After the decorated function finishes, the
        current size is decremented by 1, indicating that a concurrent call has finished.
        Finally, the wait_func function is returned as the decorated function.

    """

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """
    Wrap a function with attributes and create an EmbeddingFunc instance.
    Args:
        **kwargs:

    Returns:
        EmbeddingFunc: An EmbeddingFunc instance.
    """
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def decode_tokens(tokens: list[int], model_name: str = "gpt-4o") -> list[str]:
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    return [ENCODER.decode_single_token_bytes(token).decode('utf-8') for token in tokens]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

def truncate_list_ids_by_token_size(list_data: list, ids_data:list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i], ids_data[:i]
    return list_data, ids_data

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray, bits=8) -> tuple:
    """Quantize embedding to specified bits"""
    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    # Quantize to 0-255 range
    scale = (2 ** bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
        quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    scale = (max_val - min_val) / (2 ** bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return maybe_json_str
    except Exception:
        pass
        # try:
        #     content = (
        #         content.replace(kw_prompt[:-1], "")
        #         .replace("user", "")
        #         .replace("model", "")
        #         .strip()
        #     )
        #     maybe_json_str = "{" + content.split("{")[1].split("}")[0] + "}"
        #     json.loads(maybe_json_str)

        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)
    return combined_sources_result

# python
def process_combine_context_json(hl: str, ll: str) -> str:
    """
    Parse two CSV strings, combine rows (deduplicated), map rows to header attributes,
    and return a JSON string of the resulting list of objects.
    """
    list_hl = csv_string_to_list(hl.strip()) if hl and hl.strip() else []
    list_ll = csv_string_to_list(ll.strip()) if ll and ll.strip() else []

    header = None
    if list_hl and list_hl[0]:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if header is None and list_ll and list_ll[0]:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return json.dumps([], ensure_ascii=False, indent=2)

    combined_rows = []
    seen = set()
    for row in list_hl + list_ll:
        if not row:
            continue
        row_tuple = tuple(row)
        if row_tuple in seen:
            continue
        seen.add(row_tuple)

        # Normalize row length to header length:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            # join overflow columns into the last header field
            row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]

        combined_rows.append({header[i]: row[i] for i in range(len(header))})

    return json.dumps(combined_rows, ensure_ascii=False, indent=2)

def process_combine_chunks_ids(hl, ll):
    combined_chunks = []
    seen = set()
    for item in hl + ll:
        if item and item not in seen:
            combined_chunks.append(item)
            seen.add(item)

    return combined_chunks

def _safe_load_json(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return value

def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]

def escape_cypher_node(label: str) -> str:
    # Escape backslashes and single quotes for Cypher label usage
    label = label.strip('"')
    return label.strip().replace("\\", "\\\\").replace("'", "\\'")

def escape_cypher_properties(properties: dict) -> dict:
    # Recursively escape single quotes in string property values
    def escape_value(val):
        if isinstance(val, str):
            val = val.strip('"')
            return val.strip().replace("\\", "\\\\").replace("'", "\\'")
        elif isinstance(val, dict):
            return {k: escape_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [escape_value(v) for v in val]
        return val
    return {k: escape_value(v) for k, v in properties.items()}