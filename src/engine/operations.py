import asyncio
import json
import math
import re

from sqlalchemy import text
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from typing import Union
from collections import Counter, defaultdict
import warnings


from src.storage.db.sql.templates.classifier import SQL_TEMPLATE_CLASSIFIER
from src.storage.db.sqlbase import SQLBase
from src.utils.log import logger
from src.utils.utils import (
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    process_combine_contexts,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    compute_args_hash, truncate_list_ids_by_token_size,
    process_combine_chunks_ids,
)
from src.storage.db.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam, ClassifyParam,
)
from src.prompts.prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
        content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    """
        Split the provided content into chunks of tokens, with overlapping tokens.
    Args:
        content: The content to be chunked.
        overlap_token_size: The size of the overlap between chunks.
        max_token_size: The maximum size of each chunk.
        tiktoken_model: The TikTok model to use for encoding the content.
    Returns:
        A list of dictionaries, each containing a chunk of tokens and its properties.
    """
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start: start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
        entity_or_relation_name: str,
        description: str,
        global_config: dict,
) -> str:
    """
        Generate a summary of the provided entity or relation using the specified language and LLM model.
    Args:
        entity_or_relation_name: The name of the entity or relation.
        description: The description of the entity or relation.
        global_config: A dictionary containing global configuration settings.
    Returns:
        A summary of the provided entity or relation.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    """
    Extracts and processes a single entity record from the provided record attributes and chunk key.
    Args:
        record_attributes: A list of attributes for the entity record.
        chunk_key: The key of the chunk where the entity record is located.

    Returns:
        A dictionary containing the extracted entity details, if the record is valid, or None otherwise.
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    """
    Extracts and processes a single relationship record from the provided record attributes and chunk key.
    Args:
        record_attributes: A list of attributes for the relationship record.
        chunk_key: The key of the chunk where the relationship record is located.
    Returns:
        A dictionary containing the extracted relationship details, if the record is valid, or None otherwise.
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        global_config: dict,
):
    """
    Merges the provided nodes data into an existing entity or creates a new one, and then updates the knowledge graph
    with the merged entity details.
    Args:
        entity_name: The name of the entity to be merged or created.
        nodes_data: A list of dictionaries containing the details of the nodes to be merged or created.
        knowledge_graph_inst: A BaseGraphStorage object representing the knowledge graph.
        kg_db: A BaseGraphStorage object representing the knowledge graph database.
        global_config: A dictionary containing global configuration parameters.
    Returns:
        A boolean indicating whether the entity was successfully merged or created.
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    # Add data to graph database for storage
    doc_id: str = global_config["doc_id"]

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        doc_id=doc_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )

    await kg_db.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        global_config: dict,
):
    """
    Merges the provided edges data into an existing edge or creates a new one, and then updates the knowledge graph
    with the merged edge details.
    Args:
        src_id: The source node ID of the edge.
        tgt_id: The target node ID of the edge.
        edges_data: A list of dictionaries containing the details of the edges to be merged or created.
        knowledge_graph_inst: A BaseGraphStorage object representing the knowledge graph.
        global_config: A dictionary containing global configuration parameters.
    Returns:
        A boolean indicating whether the edge was successfully merged or created.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    # Add data to graph database for storage
    doc_id: str = global_config["doc_id"]
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            doc_id=doc_id,
        ),
    )

    await kg_db.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            doc_id=doc_id,
        ),
    )
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
    )
    return edge_data


async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        kg_db: BaseGraphStorage,
        global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    Extracts entities and relationships from the provided chunks and updates the knowledge graph.
    Args:
        chunks: A dictionary containing TextChunkSchema objects representing the input chunks.
        knowledge_graph_inst: A BaseGraphStorage object representing the knowledge graph.
        entity_vdb: A BaseVectorStorage object representing the entity vector database.
        relationships_vdb: A BaseVectorStorage object representing the relationships vector database.
        kg_db: A BaseGraphStorage object representing the knowledge graph database.
        global_config: A dictionary containing global configuration parameters.
    Returns:
        A BaseGraphStorage object representing the updated knowledge graph if extraction was successful,
        or None if there was an error.

    """
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    batch_size = global_config["entity_extract_batch_size"]
    chunks_max_join = global_config["entity_extract_chunk_max_join"]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=PROMPTS["DEFAULT_ENTITY_TYPES"],
        language=language,
    )
    #entity_types=",".join(entity_types),
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=PROMPTS["DEFAULT_ENTITY_TYPES"],
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        Processes a single chunk content and extracts entities and relationships.

        Args:
            chunk_key_dp: A tuple containing the chunk key and the corresponding TextChunkSchema object.

        Returns:
            A tuple containing the number of entities and relationships successfully extracted.
        """

        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        """
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)
        """

        logger.info(f"Extracting entities from chunk: {chunk_key}")
        logger.debug(f"Prompt for Extracting entities from chunk: {hint_prompt}")

        final_result = await use_llm_func(hint_prompt)

        """
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        """
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
            ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        #log all the entities and relations

        logger.info(f"Extracted entities: {maybe_nodes}")
        logger.info(f"Extracted relations: {maybe_edges}")

        return dict(maybe_nodes), dict(maybe_edges)

    async def process_batch(batch):
        batch_set_results = await asyncio.gather(*[_process_single_content(chunk) for chunk in batch])
        return batch_set_results

    async def join_chunks_with_tags(ordered_chunks, max_join):
        """
        Joins up to max_join chunks, tagging each with its chunk_key at the top and bottom.
        Returns a list of (joined_chunk_key, chunk_dict) tuples, matching the ordered_chunks format.
        """
        joined = []
        for i in range(0, len(ordered_chunks), max_join):
            batch = ordered_chunks[i:i + max_join]
            joined_text = []
            joined_keys = []
            for chunk_key, chunk in batch:
                content = chunk["content"]
                tagged = f"[ID: {chunk_key}]\n{content}\n[ID: {chunk_key}]"
                joined_text.append(tagged)
                joined_keys.append(chunk_key)
            joined_content = "\n\n".join(joined_text)
            # Use a combined key or the first key as the new chunk_key
            joined_chunk_key = "|".join(joined_keys)
            joined.append((joined_chunk_key, {"content": joined_content}))
        return joined

    results = []
    """
    # single processing a chunk
    for chunk_key_dp in tqdm(ordered_chunks, total=len(ordered_chunks), desc="Extracting entities from chunks",
                             unit="chunk"):
        result = await _process_single_content(chunk_key_dp)
        results.append(result)
    """
    # join the chunks using the limit of entity_extract_chunk_max_join and tag with the chunk_key at the top of content and bottom with ID:

    # for batch Split ordered_chunks into batches of size 4
    ordered_chunks = await join_chunks_with_tags(ordered_chunks, chunks_max_join)

    logger.info(f"Total chunks after joining: {len(ordered_chunks)}")

    batches = [ordered_chunks[i:i + batch_size] for i in range(0, len(ordered_chunks), batch_size)]
    for batch in tqdm(batches, total=len(batches), desc="Processing batches", unit="batch"):
        batch_results = await process_batch(batch)
        results.extend(batch_results)


    """
     #  no limit full parallel processing
     for result in tqdm_async(
            asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
            total=len(ordered_chunks),
            desc="Extracting entities from chunks",
            unit="chunk",
    ):
        results.append(await result)
    """

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    logger.info(f"Inserting entities into storage...{len(maybe_nodes)}")

    BATCH_SIZE = global_config.get("GRAHDB_BATCH_SIZE", 200)  # Example batch size, adjust as needed

    # Process entities in batches
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    maybe_nodes_items = list(maybe_nodes.items())
    num_entity_batches = math.ceil(len(maybe_nodes_items) / BATCH_SIZE)

    for batch_idx in range(num_entity_batches):
        batch = maybe_nodes_items[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        for result in tqdm_async(
                asyncio.as_completed(
                    [
                        _merge_nodes_then_upsert(k, v, knowledge_graph_inst, kg_db, global_config)
                        for k, v in batch
                    ]
                ),
                total=len(batch),
                desc=f"Inserting entities (Batch {batch_idx + 1}/{num_entity_batches})",
                unit="entity",
        ):
            all_entities_data.append(await result)

    # Process relationships in batches
    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    maybe_edges_items = list(maybe_edges.items())
    num_relationship_batches = math.ceil(len(maybe_edges_items) / BATCH_SIZE)

    for batch_idx in range(num_relationship_batches):
        batch = maybe_edges_items[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        for result in tqdm_async(
                asyncio.as_completed(
                    [
                        _merge_edges_then_upsert(
                            k[0], k[1], v, knowledge_graph_inst, kg_db, global_config
                        )
                        for k, v in batch
                    ]
                ),
                total=len(batch),
                desc=f"Inserting relationships (Batch {batch_idx + 1}/{num_relationship_batches})",
                unit="relationship",
        ):
            all_relationships_data.append(await result)


    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    doc_id: str = global_config["doc_id"]
    doc_name: str = global_config["doc_name"]
    entity_sequence = 0
    if entity_vdb is not None:
        data_for_vdb = {}
        for dp in all_entities_data:
            data_for_vdb[compute_mdhash_id(dp["entity_name"], prefix="ent-")] = {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
                "doc_id": doc_id,
                "doc_name": doc_name,
                "s_id": dp["source_id"],
                "chunk_sequence": entity_sequence,
            }
            entity_sequence += 1
        await entity_vdb.upsert(data_for_vdb)
    relationship_sequence = 0
    if relationships_vdb is not None:
        data_for_vdb = {}
        for dp in all_relationships_data:
            data_for_vdb[
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-")
            ] = {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],
                "doc_id": doc_id,
                "doc_name": doc_name,
                "s_id": dp["source_id"],
                "chunk_sequence": relationship_sequence,
            }
            relationship_sequence += 1
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def kg_query(
        query,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
        global_config: dict,
) -> tuple[str, list[str], list[str]]:
    """
    Performs a graph query on the knowledge graph.

    Args:
        query: The graph query string.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
        entities_vdb: The entities vector database instance.
        relationships_vdb: The relationships vector database instance.
        text_chunks_db: The text chunks database instance.
        query_param: The query parameters.
        global_config: The global configuration.
    Returns:
        The result of the graph query.
    # Handle cache

    """
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # Set mode
    if query_param.mode not in ["hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]

    # LLM generate keywords
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    logger.info(f"Query Keywords prompt: {kw_prompt}")

    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info(f"Query Keywords prompt result: {result}")

    try:
        # json_text = locate_json_string_body_from_string(result) # handled in use_model_func
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]

    # Handle parsing error
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return PROMPTS["fail_response"]

    # Handle keywords missing
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        hl_keywords = ", ".join(hl_keywords)

    # Build context
    keywords = [ll_keywords, hl_keywords]
    key_words = ll_keywords + "," + hl_keywords
    chunks_ids = []
    context, chunks_ids  = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        kg_db,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    logger.info(f"Query: {query}")
    logger.info(f"System prompt: {sys_prompt}")
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response, chunks_ids, key_words


async def _build_query_context(
        query: list,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
        global_config: dict,
):
    """
    Builds the context for the graph query.

    Args:
        query: The list of keywords.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
        entities_vdb: The entities vector database instance.
        relationships_vdb: The relationships vector database instance.
        text_chunks_db: The text chunks database instance.
        query_param: The query parameters.
    Returns:
        The context for the graph query.
    # ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""
    # hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""
    """

    ll_kewwords, hl_keywrds = query[0], query[1]
    ll_chunks_ids, hl_chunks_ids = [], []
    if ll_kewwords == "":
        ll_entities_context, ll_relations_context, ll_text_units_context = (
            "",
            "",
            "",
        )
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
    else:
        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
            ll_chunks_ids,
        ) = await _get_node_data(
            ll_kewwords,
            knowledge_graph_inst,
            kg_db,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )
    if hl_keywrds == "":
        hl_entities_context, hl_relations_context, hl_text_units_context = (
            "",
            "",
            "",
        )
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
    else:
        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
            hl_chunks_ids,
        ) = await _get_edge_data(
            hl_keywrds,
            knowledge_graph_inst,
            kg_db,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    entities_context, relations_context, text_units_context, chunks_ids = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
            [hl_chunks_ids, ll_chunks_ids],
    )
    completed_context = f"""
            -----Entities-----
            ```csv
            {entities_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Sources-----
            ```csv
            {text_units_context}
            ```
            """
    return completed_context, chunks_ids


async def _get_node_data(
        query,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
        global_config: dict,
):
    """
    Gets the node data for the local mode query.

    Args:
        query: The query keywords.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
        entities_vdb: The entities vector database instance.
        text_chunks_db: The text chunks database instance.
        query_param: The query parameters.
    Returns:
        The entities context, relationships context, and text units context for the local mode query.

    # get similar entities
    """
    chunks_ids = []
    results = await entities_vdb.query(query, param=query_param)
    if not len(results):
        return "", "", "",""
    # get entity information
    node_datas = await asyncio.gather(
        *[
            kg_db.get_node(r["entity_name"], param=query_param)
            for r in results
            if "entity_name" in r
        ]
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[
            kg_db.node_degree(r["entity_name"], param=query_param)
            for r in results
            if "entity_name" in r
        ]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None and "entity_name" in k
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entity text chunk
    use_text_units, chunks_ids = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst, kg_db
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst, kg_db
    )
    logger.info(
        f"Local query uses {len(node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    entities_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context, chunks_ids


async def _find_most_related_text_unit_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
):
    """
    Finds the most related text unit for each entity in the given node_datas.

    Args:
        node_datas: The entities' data.
        query_param: The query parameters.
        text_chunks_db: The text chunks database instance.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
    Returns:
        A list of text units for each entity.
    """
    # Splits the source_id of each entity in node_datas by GRAPH_FIELD_SEP and stores the results in text_units.
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    # Asynchronously gathers the edges for each entity in node_datas using knowledge_graph_inst.get_node_edges.
    edges = await asyncio.gather(
        *[kg_db.get_node_edges(dp["entity_name"], param=query_param) for dp in node_datas]
    )
    # Initializes an empty set all_one_hop_nodes and updates it with the second node of each edge in edges.
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    # Converts all_one_hop_nodes to a list and asynchronously gathers the node data for each node in all_one_hop_nodes.
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[kg_db.get_node(e, param=query_param) for e in all_one_hop_nodes]
    )

    # Creates a dictionary all_one_hop_text_units_lookup that maps each node to its source_id split by GRAPH_FIELD_SEP,
    # ensuring the node data is not None and contains source_id.
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    # Initializes an empty dictionary all_text_units_lookup and populates it with text unit data, order, and relation
    # counts. Increments relation_counts if the text unit is related to a one-hop node.
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                            e[1] in all_one_hop_text_units_lookup
                            and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return [], []
    # Sorts all_text_units by order and relation_counts, truncates the list by token size, and extracts the content.
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    # Extracts the data from all_text_units and returns the list of text units.
    all_text_data = [t["data"] for t in all_text_units]
    all_text_ids = [t["id"] for t in all_text_units]
    return all_text_data, all_text_ids


async def _find_most_related_edges_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
):
    """
    Finds the most related edges for each entity in the given node_datas.

    Args:
        node_datas: The entities data.
        query_param: The query parameters.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
    Returns:
        A list of edges for each entity.
    """
    all_related_edges = await asyncio.gather(
        *[kg_db.get_node_edges(dp["entity_name"], param=query_param) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        if this_edges is None:
            continue
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[kg_db.get_edge(e[0], e[1],param=query_param) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[kg_db.edge_degree(e[0], e[1], param=query_param) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
        keywords,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
):
    """
    Retrieves edge data based on the given keywords.

    Args:
        keywords: The keywords to search for.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
        relationships_vdb: The relationships vector storage instance.
        text_chunks_db: The text chunks database instance.
        query_param: The query parameters.
    Returns:
        A tuple of edge data, entities, and text units.
    """
    chunks_ids =[]
    results = await relationships_vdb.query(keywords, param=query_param)

    if not len(results):
        return [], [], [], []

    edge_datas = await asyncio.gather(
        *[
            kg_db.get_edge(r["src_id"], r["tgt_id"], param=query_param)
            for r in results
            if "src_id" in r and "tgt_id" in r
        ]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[
            kg_db.edge_degree(r["src_id"], r["tgt_id"], param=query_param)
            for r in results
            if "src_id" in r and "tgt_id" in r
        ]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None and "src_id" in k and "tgt_id" in k
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst, kg_db
    )
    use_text_units, chunks_ids = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context, chunks_ids


async def _find_most_related_entities_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
        kg_db: BaseGraphStorage,
):
    """
        Finds the most related entities for each edge in the given edge_datas.
    Args:
        edge_datas: The edges data.
        query_param: The query parameters.
        knowledge_graph_inst: The knowledge graph instance.
        kg_db: The knowledge graph database instance.
    Returns:
        A list of entities for each edge.
    """
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas_list = await asyncio.gather(
        *[kg_db.get_node(entity_name, param=query_param) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[kg_db.node_degree(entity_name, param=query_param) for entity_name in entity_names]
    )

    node_datas = []
    for k, n, d in zip(entity_names, node_datas_list, node_degrees):
        if k is None or n is None or d is None:
            logger.info(f"k: {k}, n: {n}, d: {d}")
        else:
            node_datas.append({**n, "entity_name": k, "rank": d})

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
        edge_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    """
    Finds the related text units for each edge in the given edge_datas.
    Args:
        edge_datas: The edges data.
        query_param: The query parameters.
        text_chunks_db: The text chunks database instance.
    Returns:
        A list of text units for each edge.
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return [], []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]
    all_text_units_ids = [t["id"] for t in truncated_text_units]

    return all_text_units, all_text_units_ids


def combine_contexts(entities, relationships, sources, chunks_ids):
    """
    Combine and deduplicate entities, relationships, and sources.
    Args:
        entities: The entities.
        relationships: The relationships.
        sources: The sources.
    Returns:
        The combined entities, relationships, and sources.
    # Function to extract entities, relationships, and sources from context strings
    """
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    combined_chunks_ids =     process_combine_chunks_ids(chunks_ids[0], chunks_ids[1])

    return combined_entities, combined_relationships, combined_sources, combined_chunks_ids


async def naive_query(
        query,
        chunks_vdb: BaseVectorStorage,
        query_param: QueryParam,
        global_config: dict,
):
    """
    Naive query handler.
    Args:
        query: The query.
        chunks_vdb: The chunks vector database instance.
        query_param: The query parameters.
        global_config: The global configuration.
    Returns:
        The response.
    # Handle cache
    """
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    chunks = await chunks_vdb.query(query, param=query_param)
    if not len(chunks):
        return PROMPTS["fail_response"]

    chunks_ids = [r["chunk_id"] for r in chunks]
    #chunks = [r["content"] for r in results if "content" in r]
    #chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks, chunks_ids = truncate_list_ids_by_token_size(
        valid_chunks,
        chunks_ids,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )

    logger.info(f"System prompt: {sys_prompt}")

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response, chunks_ids

async def document_classification(chunks_vdb:BaseVectorStorage,
                                 classify_param: ClassifyParam,
                                 global_config: dict,):
    """
        Classifies a document based on its content.
    Args:
        chunks_vdb: The database containing text chunks.
        classify_param: The classification parameters.
        global_config: The global configuration.
    Returns:
        A dictionary containing the initial summary and refined category.
    """
    use_llm_func = global_config["llm_model_func"]
    # check if required data is provided
    if not classify_param.doc_id or not classify_param.pages:
        logger.error("Document ID and pages are required for classification.")
        return {"error": "Document ID and pages are required for classification."}

    params = {
        "doc_id": classify_param.doc_id,
        "pages": classify_param.pages,
        "namespace": chunks_vdb.namespace
    }
    chunks, summary_text = await extract_pages(chunks_vdb, params)
    init_prompt = await document_classification_prompt(summary_text)
    init_response = await use_llm_func(init_prompt)

    return init_response


async def  document_classification_prompt(initial_text: str) -> str:
    basic_classification_prompt = PROMPTS["basic_document_classification"]
    context_base = dict(
        categories=PROMPTS["DEFAULT_DOCUMENT_DEFINITION"],
        initial_text=initial_text,
    )
    classify_prompt = basic_classification_prompt.format(**context_base)
    return classify_prompt

async def extract_pages(chunks_vdb: BaseVectorStorage, params: dict) -> tuple[list[str], str]:

    classic_sql = SQL_TEMPLATE_CLASSIFIER["classic_sql"]
    # Await the result if pgdb.execute is async, otherwise remove await
    result = await chunks_vdb.read(classic_sql, params=params)
    # Assuming result is a list of dicts with 'chunk_id' and 'content'
    chunk_ids = [row[0] for row in result]
    combined_text = "\n".join(row[1] for row in result)
    return chunk_ids, combined_text