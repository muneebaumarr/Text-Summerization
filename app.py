import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#FOr Downlloading the sumerize file 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import ListFlowable, ListItem
from io import BytesIO

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Professional AI Summarizer", layout="wide")

st.title("📄 Professional AI PDF Summarizer")
st.markdown("Upload a PDF and generate structured professional summaries.")

# Initialize LLM
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY")
)

# Sidebar Controls
st.sidebar.header("Summarization Settings")

length = st.sidebar.selectbox(
    "Summary Length",
    ["short", "medium", "long"]
)

format_type = st.sidebar.selectbox(
    "Output Format",
    ["paragraph", "bullet"]
)

style = st.sidebar.selectbox(
    "Writing Style",
    ["technical", "executive", "academic", "simple"]
)

#download the summerize file 
def generate_pdf(summary_text: str):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    # If bullet format → convert to bullet list
    if summary_text.strip().startswith("-") or "•" in summary_text:
        lines = summary_text.split("\n")
        bullet_items = []

        for line in lines:
            line = line.strip()
            if line:
                bullet_items.append(
                    ListItem(Paragraph(line, normal_style))
                )

        elements.append(ListFlowable(bullet_items, bulletType='bullet'))

    else:
        paragraphs = summary_text.split("\n\n")
        for para in paragraphs:
            elements.append(Paragraph(para, normal_style))
            elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)

    return buffer

def build_instructions(length: str, format_type: str, style: str):
    length_instruction = {
        "short": "Provide a concise summary in under 150 words.",
        "medium": "Provide a balanced summary covering all important points.",
        "long": "Provide a detailed summary preserving depth and nuance."
    }.get(length.lower(), "Provide a balanced summary.")

    format_instruction = {
        "paragraph": "Write in clear, structured paragraphs.",
        "bullet": "Present the summary in well-organized bullet points."
    }.get(format_type.lower(), "Write in structured paragraphs.")

    style_instruction = {
        "technical": "Use precise technical language and preserve terminology.",
        "executive": "Focus on key insights and implications in executive tone.",
        "academic": "Use formal academic style with clarity and objectivity.",
        "simple": "Explain concepts in simple, easy-to-understand language."
    }.get(style.lower(), "Use professional tone.")

    return length_instruction, format_instruction, style_instruction


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and st.button("Generate Summary"):

    with st.spinner("Processing document..."):

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        # Build instructions
        length_instruction, format_instruction, style_instruction = build_instructions(
            length, format_type, style
        )

        # Prompts
        map_prompt = ChatPromptTemplate.from_template("""
You are a professional summarizer.

{length_instruction}
{format_instruction}
{style_instruction}

Preserve key definitions and cause-effect relationships.

TEXT:
{text}

Section Summary:
""")

        combine_prompt = ChatPromptTemplate.from_template("""
You are compiling a final professional report.

{length_instruction}
{format_instruction}
{style_instruction}

Ensure logical flow and remove repetition.

COMBINED SECTION SUMMARIES:
{text}

Final Professional Summary:
""")

        # Chains
        map_chain = map_prompt | llm | StrOutputParser()
        combine_chain = combine_prompt | llm | StrOutputParser()

        # Map Phase
        partial_summaries = []
        for doc in docs:
            summary = map_chain.invoke({
                "text": doc.page_content,
                "length_instruction": length_instruction,
                "format_instruction": format_instruction,
                "style_instruction": style_instruction
            })
            partial_summaries.append(summary)

        # Hierarchical Reduce — batch summaries to stay under token limits
        BATCH_SIZE = 5
        while len(partial_summaries) > 1:
            batches = [
                partial_summaries[i : i + BATCH_SIZE]
                for i in range(0, len(partial_summaries), BATCH_SIZE)
            ]
            partial_summaries = []
            for batch in batches:
                merged = combine_chain.invoke({
                    "text": "\n\n".join(batch),
                    "length_instruction": length_instruction,
                    "format_instruction": format_instruction,
                    "style_instruction": style_instruction
                })
                partial_summaries.append(merged)

        final_summary = partial_summaries[0]

        os.remove(temp_path)

    st.success("Summary Generated Successfully!")
    st.markdown("## 📝 Final Summary")
    st.write(final_summary)
    pdf_file = generate_pdf(final_summary)

    st.download_button(
        label="📥 Download Summary as PDF",
        data=pdf_file,
        file_name="professional_summary.pdf",
        mime="application/pdf"
    )