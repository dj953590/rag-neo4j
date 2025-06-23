import streamlit as st
import base64
import random
import time
import fitz
from datetime import datetime
from pathlib import Path

# Define the repository's root directory
repo_root = Path(__file__).parent

# Create the custom sun yellow icon with document
def display_pdf(file_path):
    """Display PDF content in the right panel."""
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        pdf_pages = []

        # Extract each page as an image
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            pdf_pages.append(img_data)

        # Render pages in Streamlit
        for page_image in pdf_pages:
            st.image(page_image, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading PDF: {e}")

SUN_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <!-- Sun circle -->
  <circle cx="50" cy="50" r="40" fill="#FFD700" stroke="#FFA500" stroke-width="2"/>
  
  <!-- Sun rays -->
  <line x1="15" y1="50" x2="35" y2="50" stroke="#FFA500" stroke-width="3"/>
  <line x1="65" y1="50" x2="85" y2="50" stroke="#FFA500" stroke-width="3"/>
  <line x1="50" y1="15" x2="50" y2="35" stroke="#FFA500" stroke-width="3"/>
  <line x1="50" y1="65" x2="50" y2="85" stroke="#FFA500" stroke-width="3"/>
  <line x1="30" y1="30" x2="45" y2="45" stroke="#FFA500" stroke-width="3"/>
  <line x1="70" y1="30" x2="55" y2="45" stroke="#FFA500" stroke-width="3"/>
  <line x1="30" y1="70" x2="45" y2="55" stroke="#FFA500" stroke-width="3"/>
  <line x1="70" y1="70" x2="55" y2="55" stroke="#FFA500" stroke-width="3"/>
  
  <!-- Document -->
  <rect x="30" y="40" width="40" height="30" rx="3" fill="white" stroke="#333" stroke-width="1.5"/>
  <rect x="35" y="45" width="30" height="5" fill="#f0f0f0"/>
  <rect x="35" y="55" width="30" height="5" fill="#f0f0f0"/>
  <rect x="35" y="65" width="20" height="5" fill="#f0f0f0"/>
</svg>
"""

# Convert SVG to base64 for favicon
icon_b64 = base64.b64encode(SUN_ICON.encode('utf-8')).decode('utf-8')

# Configure the page
st.set_page_config(
    page_title="Helios CREDA",
    page_icon=f"data:image/svg+xml;base64,{icon_b64}",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown(f"""
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #FFD700;  /* Sun yellow */
            --accent: #FFA500;    /* Orange */
            --background: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --highlight: #fff9c4;
        }}
        
        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Main container styling */
        .appview-container .main .block-container {{
            max-width: 100%;
            padding: 0;
            margin: 0;
        }}
        
        /* Dual panel layout */
        .main-grid {{
            display: grid;
            grid-template-columns: 35% 65%;
            height: 100vh;
            gap: 0;
        }}
        
        /* Left panel styling */
        .left-panel {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-right: 1px solid var(--border);
            height: 100vh;
            overflow-y: auto;
        }}
        
        /* Right panel styling */
        .right-panel {{
            background: #f9f9f9;
            padding: 1.5rem;
            height: 100vh;
            overflow-y: auto;
        }}
        
        /* Title styling */
        .main-title {{
            color: var(--primary);
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border);
        }}
        
        /* Document selector styling */
        .document-selector {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }}
        
        /* Input section styling */
        .input-section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }}
        
        /* Response section styling */
        .response-section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }}
        
        /* Feedback section styling */
        .feedback-section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
        }}
        
        /* Response card styling */
        .response-card {{
            padding: 1.5rem;
            background: #f1f5f9;
            border-radius: 8px;
            border-left: 4px solid var(--secondary);
            margin-bottom: 1rem;
        }}
        
        .response-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: #64748b;
        }}
        
        .document-tag {{
            background: var(--secondary);
            color: var(--primary);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        /* Document context styling */
        .document-context {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 1rem;
            font-family: 'Courier New', monospace;
            line-height: 1.6;
        }}
        
        .context-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .highlighted {{
            background-color: var(--highlight);
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        /* Button styling */
        .stButton>button {{
            background: var(--primary) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s !important;
            width: 100%;
        }}
        
        .stButton>button:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(44,62,80,0.2);
        }}
        
        /* Text area styling */
        .stTextArea>div>div>textarea {{
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        /* Feedback radio buttons */
        .stRadio>div>label {{
            margin-right: 1rem;
            cursor: pointer;
        }}
        
        /* Progress bar styling */
        .stProgress>div>div>div>div {{
            background: linear-gradient(90deg, var(--secondary), var(--accent)) !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Preconfigured list of credit documents with sample content
CREDIT_DOCUMENTS = {
    "Credit Agreement - Amazon (2023)": {
        "file_path": str(repo_root / 'src/engine/examples/amazon/citibank-amazon.pdf')
    },
    "Credit Agreement - Caterpillar (2024)": {
        "file_path": str(repo_root / 'src/engine/examples/caterpillar/citibank-caterpillar.pdf')
    },
    "Credit Agreement - Dunkin": {
        "file_path": str(repo_root / 'src/engine/examples/dunkin/citibank-dunkin.pdf')
    }
}

# Simulated responses based on document and query
def generate_response(document, query):
    """Generate a simulated response based on the document and query"""

    # Simulate processing time
    with st.spinner("Analyzing document..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        progress_bar.empty()

    # Document-specific responses
    doc_data = CREDIT_DOCUMENTS[document]

    # Try to find matching context
    for keyword in doc_data["context"]:
        if keyword in query.lower():
            return {
                "response": doc_data["context"][keyword],
                "context": doc_data["context"][keyword],
                "section": keyword
            }

    # Generic responses
    responses = [
        f"The {document} specifies that the borrower must maintain certain financial ratios throughout the term of the agreement.",
        f"According to section 4.3 of the {document}, prepayments are allowed without penalty after the first anniversary of the agreement.",
        f"The {document} includes standard representations and warranties regarding the borrower's financial condition and legal authority.",
        f"Assignment provisions in the {document} require lender consent for any transfer of the borrower's obligations.",
        f"The governing law clause in the {document} specifies that disputes will be resolved under New York law.",
    ]

    return {
        "response": random.choice(responses),
        "context": random.choice(list(doc_data["context"].values())),
        "section": "General Provisions"
    }

def highlight_context(document_content, context):
    """Highlight the context in the document content"""
    if context and context in document_content:
        highlighted_content = document_content.replace(
            context,
            f'<span class="highlighted">{context}</span>'
        )
        return highlighted_content
    return document_content

def main():
    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = list(CREDIT_DOCUMENTS.keys())[0]
    if "response" not in st.session_state:
        st.session_state.response = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "feedback_value" not in st.session_state:
        st.session_state.feedback_value = None
    if "feedback_comment" not in st.session_state:
        st.session_state.feedback_comment = ""

    # Create dual panel layout
    col1, col2 = st.columns([1, 1.8])

    # Left panel - Controls and Query
    with col1:
        st.markdown('<div>', unsafe_allow_html=True)

        # Main title
        st.markdown('<h1 class="main-title">Helios CREDA</h1>', unsafe_allow_html=True)
        st.caption("Credit Document Expert Assistant")

        # Document selection
        st.markdown('<div>', unsafe_allow_html=True)
        selected_doc = st.selectbox("Choose a document to analyze", list(CREDIT_DOCUMENTS.keys()))
        st.session_state.selected_doc = selected_doc
        st.markdown('</div>', unsafe_allow_html=True)

        # Query input
        st.markdown('<div>', unsafe_allow_html=True)
        query = st.text_area(
            "Enter your natural language question:",
            height=300,
            placeholder="E.g., 'What is the interest rate?' or 'Explain the covenant requirements'",
            key="query_input"
        )

        if st.button("Submit Question", key="submit_btn", use_container_width=True):
            if query.strip():
                st.session_state.query = query
                response = generate_response(selected_doc, query)
                st.session_state.response = response
                st.session_state.feedback_given = False
                st.session_state.feedback_value = None
                st.session_state.feedback_comment = ""
            else:
                st.warning("Please enter a question before submitting")
        st.markdown('</div>', unsafe_allow_html=True)

        # Display response if available
        if st.session_state.response:
            st.markdown('<div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="response-card">
                    <div class="response-header">
                        <div>Query: <strong>{st.session_state.query}</strong></div>
                        <div class="document-tag">{st.session_state.selected_doc}</div>
                    </div>
                    <div>{st.session_state.response['response']}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Feedback section
            st.markdown('<div>', unsafe_allow_html=True)
            if not st.session_state.feedback_given:
                feedback = st.radio(
                    "Select your feedback:",
                    ["Yes", "No"],
                    index=None,
                    key="feedback_radio",
                    horizontal=True
                )

                if feedback == "No":
                    st.session_state.feedback_comment = st.text_area(
                        "Please help us improve. What was missing or incorrect?",
                        height=100,
                        key="feedback_comment"
                    )

                if feedback:
                    if st.button("Submit Feedback", key="feedback_btn", use_container_width=True):
                        st.session_state.feedback_given = True
                        st.session_state.feedback_value = feedback
                        st.success("Thank you for your feedback! We'll use it to improve our responses.")
            else:
                st.success("Thank you for your feedback! We'll use it to improve our responses.")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close left-panel

    # Right panel - Document display with context
    with col2:
        st.markdown('<div>', unsafe_allow_html=True)

        doc_data = CREDIT_DOCUMENTS[st.session_state.selected_doc]

        # Document header
        st.markdown(f"""
            <div class="context-header">
                <h2>📄 {st.session_state.selected_doc}</h2>
                <div class="document-tag">Document Context</div>
            </div>
        """, unsafe_allow_html=True)

        # Display context if available
        if st.session_state.response and st.session_state.response.get('context'):
            st.subheader(f"🔍 Context from document: {st.session_state.response['section']}")

        # Display PDF content dynamically
        display_pdf(doc_data["file_path"])

        st.markdown('</div>', unsafe_allow_html=True)  # Close right-panel

if __name__ == "__main__":
    main()