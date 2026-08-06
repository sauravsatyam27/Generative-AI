# ==========================================
# Import Required Libraries
# ==========================================

# Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment variables
from dotenv import load_dotenv

# Streamlit UI
import streamlit as st

# LangChain PromptTemplate
from langchain_core.prompts import PromptTemplate


# ==========================================
# Load .env
# ==========================================

load_dotenv()


# ==========================================
# Create Gemini Model
# ==========================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ==========================================
# Streamlit Heading
# ==========================================

st.header("Research Tool")


# ==========================================
# Research Paper
# ==========================================

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)


# ==========================================
# Explanation Style
# ==========================================

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)


# ==========================================
# Explanation Length
# ==========================================

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)


# ==========================================
# Create Prompt Template
# ==========================================

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}"
with the following specifications:

Explanation Style:
{style_input}

Explanation Length:
{length_input}

1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple and intuitive
  explanations where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

If certain information is not available, respond with:
"Insufficient information available"
instead of guessing.

Ensure the summary is clear, accurate, and aligned with
the provided style and length.
""",

    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)


# ==========================================
# Summarize Button
# ==========================================

if st.button("Summarize"):

    # --------------------------------------
    # Fill PromptTemplate placeholders
    # --------------------------------------

    prompt = template.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        }
    )


    # --------------------------------------
    # Send Prompt to Gemini
    # --------------------------------------

    result = model.invoke(prompt)


    # --------------------------------------
    # Display Gemini Response
    # --------------------------------------

    st.write(result.content)