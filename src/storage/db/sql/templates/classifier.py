SQL_TEMPLATE_CLASSIFIER = {
    "classic_sql": """SELECT chunk_id, content
                    FROM rag.documents
                    WHERE doc_id = :doc_id
                    AND chunk_sequence <= :pages
                    AND mdata->>'ns' = :namespace""",
}