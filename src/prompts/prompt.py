from src.entities.legal.legal import load_legal_entities

from pathlib import Path

base_dir = Path(__file__).parent
agreement_entities_path = (base_dir / '..' / 'entities' / 'legal' / 'data' / 'Agreement.txt').resolve()
document_definition_path = (base_dir / '..' / 'entities' / 'legal' / 'data' / 'DocumentsDefinition.txt').resolve()
ten_k_entities_path = (base_dir / '..' / 'entities' / 'legal' / 'data' / '10KEntities.txt').resolve()

def default_entities_types(file_path):
    default_entities = load_legal_entities(file_path)
    return default_entities.dump()


DEFAULT_ENTITY_TYPES = default_entities_types(agreement_entities_path)
DEFAULT_DOCUMENT_DEFINITION = default_entities_types(document_definition_path)
DEFAULT_10K_ENTITIES = default_entities_types(ten_k_entities_path)

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = DEFAULT_ENTITY_TYPES
PROMPTS["DEFAULT_DOCUMENT_DEFINITION"] = DEFAULT_DOCUMENT_DEFINITION
PROMPTS["DEFAULT_10K_ENTITIES"] = DEFAULT_10K_ENTITIES

PROMPTS["entity_extraction"] = """You are credit analyst, expert in named entity relationship extractor for credit, legal and financial domains.
- **Your Goal** - 
Given a **Markdown** text tagged with **ID:** at the top and bottom of each document page and a predefined set of entity types and relationship types, perform a precise and comprehensive extraction. Use **{language}** as the output language.

---
### **Steps**
1.  **Process Contextual Information:**
    - First, process the "DEFINITIONS; CONSTRUCTION" section to build an internal context map. For each capitalized term defined in this section, map it to its full, real-world referent or a comprehensive description. For example, recognize that "Bank" refers to "CITIBANK, N.A." and "Borrower" refers to "PUERTO RICO ELECTRIC POWER AUTHORITY."

2.  **Identify and Classify Entities:**
    - Scan the entire text and identify only real-world entities that fall into the following valid, specific types {entity_types}
    - DO NOT extract any entity that is merely a defined term (e.g., "Advances", "Agreement", "Default Rate"). Map such terms back to their real-world referents identified in Step 1.
    - For each identified entity, extract:
        - **Entity Name**: The full, exact name as it appears in the text (e.g., "CITIBANK, N.A.").
        - **Entity Type**: A type from the predefined list of entity types.
        - **Entity Description**: A brief, factual summary of the entity's obligations or significance.
    - Format each entity as a structured tuple:
      `("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>){record_delimiter}`

3.  **Extract Relationships Among Identified Entities:**
    - Identify clear, factual relationships between the extracted entities.
    - For each relationship, extract:
        - **Source Entity**: The full name of the first entity involved.
        - **Target Entity**: The full name of the second entity.
        - **Relationship Description**: A concise, factual summary of the relationship.
        - **Relationship Keywords**: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details.
        - **Relationship Type**: The specific type from the predefined list. Use only the following predefined relationship types: `is_party_to`, `is_represented_by`, `provides_credit_to`, `is_governed_by`, `is_authorized_by`, `has_obligation_to`
        - **Relationship Strength**: A numeric score (1-10) indicating how strong the relationship is between the source entity and target entity.  
          
    - **Format each relationship** as:  
     `("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_strength>){record_delimiter}`
   - **DO NOT** include entities with **NO relationships** in the output.
   - **DO NOT** include relationships that do not have a **source entity** and a **target entity**.

4.  **Return Output:**
    - Return output in structured format in {language} as a single list of all the entities and relationships identified in steps 2 and 3.
    - Use **{record_delimiter}** as the delimiter between records.
    - End response with `{completion_delimiter}`.

######################
-Real Data-
######################
Text: {input_text}
######################
-Examples-
######################
{examples}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:
Entity_types: ["company", client, "author", "ticker", "industry", "event", "statistics"]
Text:
Tata Technologies (TATE.NS) Initiate at Sell: Leveraged to Auto Vertical;
Upsides Priced In CITI's TAKE Initiate at Sell; TP of Rs1000 – Tata
Technologies (TTL) is a ER&D service provider (particularly exposed to
automotive vertical) and will likely benefit from (a) resilience of tech
spends in the automotive vertical given its specialization and expertise
(b) focus on other key verticals like aerospace – recently, TTL has become
a strategic supplier for Airbus (c) Mining opportunities in existing set of
marquee clients – TML, JLR, etc. Tata Group parentage will further help.
While we are constructive on the business medium term, valuations
(~53x 1-yr forward Citi ests) appear to be pricing in the positives.
Initiate at Sell with TP of Rs1,000 (~10% downside).
Sell
Price (30 Jan 24 15:30) Rs1,120.85
Target price Rs1,000.00
Expected share price return -10.8%
Expected dividend yield 0.7%
Expected total return -10.1%
Market Cap Rs454,694M
US$5,470M
The legal entities employing the authors of this report are listed below (and their regulators are listed further herein). Rajiv Berlia; Surendra Goyal
################
Output:
("entity"{tuple_delimiter}Tata Technologies{tuple_delimiter}company{tuple_delimiter}Tata Technologies (TATE.NS) Initiate at Sell: Leveraged to Auto Vertical.){record_delimiter}
("entity"{tuple_delimiter}TATE.NS{tuple_delimiter}Ticker{tuple_delimiter}Tata Technologies (TATE.NS) Initiate at Sell.){record_delimiter}
("entity"{tuple_delimiter}automotive{tuple_delimiter}industry{tuple_delimiter}Technologies (TTL) is a ER&D service provider (particularly exposed to automotive vertical)){record_delimiter}
("entity"{tuple_delimiter}aerospace{tuple_delimiter}industry{tuple_delimiter}focus on other key verticals like aerospace – recently, TTL has become a strategic supplier for Airbus){record_delimiter}
("entity"{tuple_delimiter}Price{tuple_delimiter}statistics{tuple_delimiter}Initiate at Sell with TP of Rs1,000 (~10% downside).Sell Price (30 Jan 24 15:30) Rs1,120.85 Target price Rs1,000.00){record_delimiter}
("entity"{tuple_delimiter}TML{tuple_delimiter}client{tuple_delimiter}mining opportunities in existing set of marquee clients – TML, JLR, etc. Tata Group parentage will further help){record_delimiter}
("entity"{tuple_delimiter}Rajiv Berlia{tuple_delimiter}author{tuple_delimiter}The legal entities employing the authors of this report are listed below and their regulators are listed further herein {tuple_delimiter}Rajiv Berlia; Surendra Goyal){record_delimiter}
("relationship"{tuple_delimiter}TTL{tuple_delimiter}VinFast{tuple_delimiter}TTL was engaged by VinFast across full vehicle turnkey programs for their VF6 and VF7 models resulting in significant growth for Tata{tuple_delimiter}turnkey program{tuple_delimiter}is_party_to{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}TTL{tuple_delimiter}Sell{tuple_delimiter}We initiate TTL with a {tuple_delimiter}Sell rating and target price of Rs1000{tuple_delimiter}Rating, Target price{tuple_delimiter}is_represented_by{tuple_delimiter}6){record_delimiter}
#############################""",
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a expert credit underwriter responsible for generating a comprehensive summary of the entities and description provided.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. **Comprehensiveness:** The summary must integrate key information from all provided descriptions. Do not omit important facts.
2. **Context:** The summary must explicitly mention the name of the entity or relation for full context.
3. **Conflict:** In case of conflicting or inconsistent descriptions, determine if they originate from multiple, distinct entities or relationships that share the same name. If so, summarize each entity or relationship separately and then consolidate all summaries.
4. **Style:** The output must be written from an objective, third-person perspective.
5. **Length:** Maintain depth and completeness while ensuring the summary's length not exceed {summary_length} tokens.
6. **Language:** The entire output must be written in {language}.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["hybrid_rag_response"] = """You are a expert credit underwriter responding to user query about data in the tables provided.

# INSTRUCTIONS
Your goal is to provide a response to the user query using the relevant information in the input data:
- the "Entities" and "Relationships" tables contain high-level information. Use these tables to identify the most important entities and relationships to respond to the query.
- the "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

Follow these steps:
1. Read and understand the user query.
2. Look at the "Entities" and "Relationships" tables to get a general sense of the data and understand which information is the most relevant to answer the query.
3. Carefully analyze all the "Sources" to get more detailed information. Information could be scattered across several sources, use the identified relevant entities and relationships to guide yourself through the analysis of the sources.
4. Write the response to the user query - which must include the inline references - based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.


# INPUT DATA
{context_data}

Answer:
"""
PROMPTS["rag_response"] = """---Role---

You are a expert credit underwriter responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a expert credit underwriter tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, credit, legal or financial domain jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a expert credit underwriter responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents are in Markdown format---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown and include the inline references for chunk id's - based on the information you have gathered.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS[
    "basic_document_classification"
] = """
You are a financial document classification expert.

Analyze the following document excerpt and determine the most likely category and subcategory. This excerpt is from the first few pages of the document.

Categories to choose from:
{categories}

Excerpt:
{initial_text}

Respond in JSON:
{{
    "category": "<best category>",
    "subcategory": "<underlying type or subcategory>",
    "confidence": 0.0 to 1.0,
    "justification": "<why you chose this category>"
}}
"""

PROMPTS[
    "progressive_document_classification"
] = """
You are continuing the classification of a financial document using a progressive strategy.

You previously classified the document as:
{prev_summary}

Now analyze the next chunk (Chunk {chunk_num}) and determine whether this chunk:
1. Supports the previous classification
2. Introduces a new classification
3. Is irrelevant or inconclusive

Chunk {chunk_num}:
{chunk_text}

Respond in JSON:
{{
  "chunk_number": {chunk_num},
  "chunk_classification": "<category or 'inconclusive'>",
  "is_consistent_with_initial": true/false,
  "confidence": 0.0 to 1.0,
  "new_keywords": ["<if any>"],
  "justification": "<reasoning>",
  "refined_category": "<refined suggestion, or same>"
}}
"""
