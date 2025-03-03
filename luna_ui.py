import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="Helios Luna AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🚀"
)

# CSS styling for side-by-side layout
st.markdown("""
    <style>
        /* Hide Streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --background: #f8f9fa;
            --text: #2c3e50;
            --border: #e0e0e0;
        }

        /* Base container styling */
        .appview-container .main .block-container {
            padding: 0;
            margin: 0;
            max-width: 100%;
            height: 100vh;
        }

        /* Column containers */
        .stHorizontalBlock {
            gap: 1rem;
            height: 100vh;
            align-items: stretch;
        }

        /* Left panel styling */
        [data-testid="stHorizontalBlock"] > div:first-child {
            flex: 1;
            min-width: 400px;
            max-width: 33.33%;
            height: calc(100vh - 1rem);
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
            overflow-y: auto;
            margin: 0.5rem 0 0.5rem 0.5rem;
        }

        /* Right panel styling */
        [data-testid="stHorizontalBlock"] > div:last-child {
            flex: 2;
            min-width: 600px;
            height: calc(100vh - 1rem);
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
            overflow-y: auto;
            margin: 0.5rem 0.5rem 0.5rem 0;
        }

        /* Title styling */
        .title {
            font-family: 'Inter', sans-serif;
            color: var(--primary);
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }

        /* Input styling */
        .stTextArea textarea {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-family: 'Inter', sans-serif;
        }

        /* Button styling */
        .stButton button {
            width: 100%;
            background: var(--secondary) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 500;
            transition: all 0.2s;
            margin-top: 1rem;
        }

        .stButton button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(52,152,219,0.2);
        }

        /* Response cards */
        .response-card {
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            animation: fadeIn 0.3s ease-in;
        }

        .response-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #666;
        }

        .module-tag {
            background: var(--secondary);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Scrollable response area */
        .response-container {
            height: calc(100vh - 200px);
            overflow-y: auto;
            padding-right: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


def send_request(endpoint, query):
    with st.spinner('Processing your request...'):
        try:
            response = requests.post(endpoint, json={"query": query}, timeout=30)
            response.raise_for_status()
            return response.text, None
        except Exception as e:
            return None, str(e)


def main():
    if "responses" not in st.session_state:
        st.session_state.responses = []

    # Create side-by-side columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="title">Helios Luna AI Assistant</div>', unsafe_allow_html=True)
        query = st.text_area("Enter your query:",
                             height=150,
                             placeholder="Type your question here...",
                             key="query_input")

        modules = {
            "SQL Optimizer": "http://localhost:8000/luna/sql_optimizer",
            "SQL Metadata": "http://localhost:8000/luna/sql_metadata",
            "SQL Builder": "http://localhost:8000/luna/sql_builder",
            "NL WOPR": "http://localhost:8000/luna/nl_wopr"
        }

        module_selection = st.selectbox("Select Module", modules.keys())
        endpoint = modules[module_selection]

        if st.button("Send Request"):
            if query.strip():
                response, error = send_request(endpoint, query)
                if response:
                    timestamp = datetime.now().strftime("%H:%M · %m/%d/%Y")
                    st.session_state.responses.insert(0, {
                        "module": module_selection,
                        "query": query,
                        "response": response,
                        "timestamp": timestamp
                    })
                else:
                    st.error(f"Error: {error}")
                st.session_state.query_input = ""
            else:
                st.warning("Please enter a query before submitting")

    with col2:
        st.markdown('<div class="title">Response History</div>', unsafe_allow_html=True)
        st.markdown('<div class="response-container">', unsafe_allow_html=True)

        if not st.session_state.responses:
            st.markdown('<div class="response-card">No responses yet. Submit a query to begin.</div>',
                        unsafe_allow_html=True)
        else:
            for response in st.session_state.responses:
                st.markdown(f'''
                    <div class="response-card">
                        <div class="response-header">
                            <span>{response["timestamp"]}</span>
                            <span class="module-tag">{response["module"]}</span>
                        </div>
                        <div><strong>Query:</strong> {response["query"]}</div>
                        <hr style="margin: 0.5rem 0; border-color: var(--border);">
                        <div>{response["response"]}</div>
                    </div>
                ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close response-container


if __name__ == '__main__':
    main()
