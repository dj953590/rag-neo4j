GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["company", "client", "author", "ticker", "industry", "event", "statistics"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
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
("entity"{tuple_delimiter}"Tata Technologies"{tuple_delimiter}"company"{tuple_delimiter}"Tata Technologies (TATE.NS) Initiate at Sell: Leveraged to Auto Vertical."){record_delimiter}
("entity"{tuple_delimiter}"TATE.NS"{tuple_delimiter}"Ticker"{tuple_delimiter}"ata Technologies (TATE.NS) Initiate at Sell."){record_delimiter}
("entity"{tuple_delimiter}"automotive"{tuple_delimiter}"industry"{tuple_delimiter}"Technologies (TTL) is a ER&D service provider (particularly exposed to automotive vertical)."){record_delimiter}
("entity"{tuple_delimiter}"aerospace"{tuple_delimiter}"industry"{tuple_delimiter}"focus on other key verticals like aerospace – recently, TTL has become a strategic supplier for Airbus."){record_delimiter}
("entity"{tuple_delimiter}"Price"{tuple_delimiter}"statistics"{tuple_delimiter}"Initiate at Sell with TP of Rs1,000 (~10% downside).Sell Price (30 Jan 24 15:30) Rs1,120.85 Target price Rs1,000.00."){record_delimiter}
("entity"{tuple_delimiter}"TML"{tuple_delimiter}"client"{tuple_delimiter}"mining opportunities in existing set of marquee clients – TML, JLR, etc. Tata Group parentage will further help."){record_delimiter}
("entity"{tuple_delimiter}"Rajiv Berlia"{tuple_delimiter}"author"{tuple_delimiter}"The legal entities employing the authors of this report are listed below and their regulators are listed further herein {tuple_delimiter}Rajiv Berlia; Surendra Goyal."){record_delimiter}
("relationship"{tuple_delimiter}"TTL"{tuple_delimiter}"VinFast"{tuple_delimiter}"TTL was engaged by {tuple_delimiter}VinFast across full vehicle turnkey programs for their VF6 and VF7 models resulting in significant growth for Tata"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"TTL"{tuple_delimiter}"Sell"{tuple_delimiter}"We initiate TTL with a {tuple_delimiter}Sell rating and {tuple_delimiter}target price of Rs1000"{tuple_delimiter}6){record_delimiter}
("content_keywords"{tuple_delimiter}"Growth Drivers, Client Expansion, Company Financials, Revenues"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

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

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


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

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

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

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
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
