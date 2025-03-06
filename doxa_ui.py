import streamlit as st
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Force full‑width layout
st.set_page_config(page_title="DoXa-E", layout="wide")

# Modern CSS styling
st.markdown("""
    <style>
        /* Hide default elements */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* Modern color scheme */
        :root {
            --primary: #2563eb;
            --secondary: #3b82f6;
            --background: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
        }

        /* Base layout fixes */
        .appview-container .main .block-container {
            padding: 1rem !important;
            max-width: 100% !important;
            height: 100vh;
        }

        /* Main columns container */
        [data-testid="stHorizontalBlock"] {
            gap: 1.5rem;
            align-items: stretch;
            height: calc(100vh - 2rem);
        }

        /* Column styling */
        [data-testid="stHorizontalBlock"] > div {
            min-width: 400px;
            height: 100%;
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
            overflow-y: auto;
        }

        /* Title styling */
        .title {
            font-family: 'Inter', sans-serif;
            color: var(--primary);
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            letter-spacing: -0.02em;
        }

        /* Section headers */
        .section-header {
            color: var(--primary) !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }

        /* Response history header */
        .response-history-header {
            font-size: 1.1rem !important;
            color: var(--primary) !important;
            margin-bottom: 1rem !important;
        }

        /* Form elements */
        .stTextInput input, .stTextArea textarea {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            font-size: 0.95rem !important;
        }

        /* Buttons */
        .stButton button {
            width: 100% !important;
            background: var(--primary) !important;
            color: white !important;
            border: none !important;
            padding: 14px 24px !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s !important;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.25);
        }

        /* Response cards */
        .response-card {
            padding: 1.5rem;
            background: var(--card-bg);
            border-radius: 12px;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            transition: transform 0.2s;
        }

        .response-card:hover {
            transform: translateY(-2px);
        }
    </style>
""", unsafe_allow_html=True)


def send_request(endpoint, **kwargs):
    try:
        response = requests.post(endpoint, **kwargs, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)


def main():
    # Main title
    st.markdown('<div class="title">📄 Document Extraction Engine (DoXa-E)</div>', unsafe_allow_html=True)

    # Initialize session state
    if "processing_id" not in st.session_state:
        st.session_state.processing_id = None
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None

    # Create main columns
    col_left, col_right = st.columns([1, 2], gap="large")

    # Left Panel - Document Processing
    with col_left:
        st.markdown('<div class="section-header">🗂️ Document Processing</div>', unsafe_allow_html=True)

        # File Upload Section
        uploaded_file = st.file_uploader(
            "📤 Upload Document",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )

        gfrn_id = st.text_input(
            "🔖 GFRN ID",
            placeholder="Enter unique document ID",
            help="Unique identifier for the document"
        )

        if st.button("Upload Document", key="upload_btn"):
            if uploaded_file and gfrn_id.strip():
                file_data = uploaded_file.getvalue()
                data = {"parent_id": gfrn_id.strip(), "state": "UPLOADED", "name": uploaded_file.name}
                with st.spinner("⏳ Uploading document..."):
                    result, error = send_request("http://localhost:8000/upload", files=files, data=data)
                if result:
                    st.session_state.processing_id = result.get("processing_id")
                    st.success("✅ Document uploaded successfully!")
                else:
                    st.error(f"❌ Upload failed: {error}")
            else:
                st.warning("⚠️ Please provide both document and GFRN ID")

        # Processing Status
        if st.session_state.processing_id:
            st.markdown("---")
            st.markdown('<div class="section-header">📈 Processing Status</div>', unsafe_allow_html=True)
            st_autorefresh(interval=5000, key="status_refresh")

            try:
                status_response = requests.get(
                    f"http://localhost:8000/status?processing_id={st.session_state.processing_id}",
                    timeout=5
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    st.session_state.processing_status = status_data.get("status")
                    status_color = "#10B981" if st.session_state.processing_status == "READY" else "#3B82F6"
                    st.markdown(f"""
                        <div style="padding: 1rem; background: {status_color}10; border-radius: 8px; border-left: 4px solid {status_color};">
                            <div style="font-size: 0.95rem; color: {status_color}; margin-bottom: 0.5rem;">
                                {st.session_state.processing_status}
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B;">
                                Processing ID: {st.session_state.processing_id}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Failed to fetch processing status")
            except Exception as e:
                st.error(f"Status check failed: {str(e)}")

    # Right Panel - Query Execution
    with col_right:
        st.markdown('<div class="section-header">🔍 Query Execution</div>', unsafe_allow_html=True)

        try:
            agreements_response = requests.get("http://localhost:8000/agreements", timeout=5)
            ready_docs = [
                doc["doc_id"] for doc in agreements_response.json()
                if doc.get("status", "").upper() == "READY"
            ] if agreements_response.status_code == 200 else []
        except Exception as e:
            ready_docs = []
            st.error(f"Failed to load documents: {str(e)}")

        if ready_docs:
            selected_doc = st.selectbox("📄 Select Ready Document", ready_docs)
            query_text = st.text_area(
                "💬 Enter your query:",
                height=150,
                placeholder="Type your question about the document...",
                key="query_input"
            )

            if st.button("🚀 Send Query", key="send_query"):
                if query_text.strip():
                    with st.spinner("🔍 Analyzing document..."):
                        payload = {"doc_id": selected_doc, "query": query_text}
                        result, error = send_request("http://localhost:8000/query", json=payload)

                    if result:
                        timestamp = datetime.now().strftime("%H:%M · %m/%d/%Y")
                        st.session_state.responses.insert(0, {
                            "timestamp": timestamp,
                            "module": selected_doc,
                            "query": query_text,
                            "response": result.get("response", "")
                        })
                        st.success("✅ Query executed successfully!")
                    else:
                        st.error(f"❌ Query failed: {error}")
                else:
                    st.warning("⚠️ Please enter a query before submitting")

            # Response History
            st.markdown("---")
            st.markdown('<div class="response-history-header">📜 Response History</div>', unsafe_allow_html=True)

            if st.session_state.responses:
                for resp in st.session_state.responses:
                    st.markdown(f'''
                        <div class="response-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <div style="font-size: 0.9rem; color: #64748B;">{resp['timestamp']}</div>
                                <div style="font-weight: 500; color: var(--primary);">{resp['module']}</div>
                            </div>
                            <div style="margin-bottom: 1rem; color: var(--text);">
                                <strong>❓ Query:</strong> {resp['query']}
                            </div>
                            <div style="background: #F8FAFC; padding: 1rem; border-radius: 8px;">
                                <strong>💡 Response:</strong> {resp['response']}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("📭 No responses yet. Submit a query to see results.")
        else:
            st.info("📥 No processed documents available. Please upload a document first.")


if __name__ == '__main__':
    main()
