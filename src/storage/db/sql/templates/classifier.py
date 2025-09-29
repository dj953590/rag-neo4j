SQL_TEMPLATE_CLASSIFIER = {
    "classic_sql": """SELECT chunk_id, content
                    FROM rag.documents
                    WHERE doc_id = :doc_id
                    AND chunk_sequence <= :pages
                    AND mdata->>'ns' = :namespace""",
    "document_result": """SELECT d.chunk_id, d.content, d.chunk_sequence
                    FROM rag.documents d
                    WHERE d.doc_id = :doc_id
                    AND d.mdata->>'ns' = :namespace""",
}
