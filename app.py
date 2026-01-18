import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY MANAGEMENT ---
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except:
    # Fallback for Local Testing
    OPENROUTER_API_KEY = "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"
    TAVILY_API_KEY = "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"

# --- DATA MODELS ---
class VerificationResult(BaseModel):
    claim: str = Field(description="The main claim extracted from the section")
    verification_status: str = Field(description="One of: 'Verified', 'Inaccurate', 'False'")
    correction: str = Field(description="Detailed correction or confirmation based on search results")
    source_url: str = Field(description="The URL of the best source found")

class ClaimList(BaseModel):
    claims: List[str] = Field(description="A list of main claims found in the text")

# --- FUNCTIONS ---

def extract_text_from_pdf(file):
    file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return "\n".join([doc.page_content for doc in docs])

def extract_claims(text, llm):
    parser = JsonOutputParser(pydantic_object=ClaimList)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior editor. 
        Analyze the text structure. If it has numbered sections, extract exactly **ONE** main composite claim per section that summarizes the key statistics/facts.
        Return a JSON list."""),
        ("human", "Text:\n{text}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    try:
        return chain.invoke({"text": text, "format_instructions": parser.get_format_instructions()})['claims']
    except:
        return []

def verify_claim(claim, llm, search_tool):
    search_query = f"{claim} verification fact check details"
    
    try:
        search_results = search_tool.invoke({"query": search_query})
        context = "\n".join([f"Source ({res['url']}): {res['content']}" for res in search_results[:4]])
    except Exception as e:
        return VerificationResult(claim=claim, verification_status="Error", correction=str(e), source_url="N/A")

    parser = JsonOutputParser(pydantic_object=VerificationResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional fact-checker. 
        Analyze the CLAIM against the SEARCH CONTEXT from multiple sources.
        
        GUIDELINES:
        1. **Multiple Sources:** Synthesize information from the provided search results. If sources conflict, prioritize the most recent and authoritative one.
        2. **Historical Context:** If the claim is about a past event (e.g., 2019), use older sources if they verify the event occurred.
        3. **Future/Current:** If the claim is about the present/future (e.g., 2026), compare it against the latest available data.
        
        STATUS RULES:
        - 'Verified': The claim is supported by the evidence.
        - 'Inaccurate': The claim is partially true but gets numbers, dates, or specific details wrong.
        - 'False': The claim is completely unsupported, contradicted by evidence, or is a prediction presented as fact that hasn't happened.
        """),
        ("human", "CLAIM: {claim}\n\nSEARCH CONTEXT: {context}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    try:
        return chain.invoke({"claim": claim, "context": context, "format_instructions": parser.get_format_instructions()})
    except:
        return VerificationResult(claim=claim, verification_status="Error", correction="Processing failed", source_url="N/A")

# --- UI LAYOUT ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/verified-account.png", width=64)
    st.title("FactCheck AI")
    
    st.markdown("### ❓ What does this tool do?")
    st.markdown("""
    This automated agent acts as a gatekeeper between drafts and publication.
    
    **1. Extract:** Scans PDF documents to identify specific statistical claims, dates, and financial figures.
    
    **2. Verify:** Cross-references each claim against **multiple live web sources** to ensure accuracy across timelines.
    
    **3. Report:** Flags claims as:
    * ✅ **Verified** (Matches reality)
    * ⚠️ **Inaccurate** (Details wrong)
    * ❌ **False** (Contradicts reality)
    """)
    
    st.divider()
    st.caption("Powered by LangChain, OpenRouter & Tavily")

st.title("🛡️ Automated Fact-Checking Agent")

# Simplified Input Area (Just PDF Upload)
uploaded_file = st.file_uploader("📂 Upload PDF Report", type="pdf")

if uploaded_file:
    # Check for keys before processing
    if "PASTE_" in OPENROUTER_API_KEY:
        st.error("⚠️ Keys missing! Please add them to `app.py` or Streamlit Secrets.")
        st.stop()

    text_content = extract_text_from_pdf(uploaded_file)
    
    if st.button("Start Verification", type="primary"):
        llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )
        
        search = TavilySearchResults(
            max_results=5, 
            tavily_api_key=TAVILY_API_KEY, 
            search_depth="advanced" 
        )

        with st.status("🕵️ Analyzing document...", expanded=True) as status:
            claims = extract_claims(text_content, llm)
            st.write(f"Found {len(claims)} composite claims.")
            
            results = []
            progress = st.progress(0)
            
            for i, claim in enumerate(claims):
                st.write(f"Checking: Section {i+1}...")
                res = verify_claim(claim, llm, search)
                results.append(res)
                progress.progress((i + 1) / len(claims))
            
            status.update(label="Analysis Completed!", state="complete", expanded=False)

        # Dashboard Summary
        verified = sum(1 for r in results if r['verification_status'] == "Verified")
        score = int((verified / len(results)) * 100) if results else 0
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Truth Score", f"{score}%")
        c2.metric("Claims Checked", len(results))
        c3.metric("Verified", verified)
        st.progress(score/100)

        # Detailed Report
        st.subheader("📝 Verification Details")
        for item in results:
            color = "green" if item['verification_status'] == "Verified" else "red"
            icon = "✅" if item['verification_status'] == "Verified" else ("⚠️" if item['verification_status'] == "Inaccurate" else "❌")
            
            with st.expander(f"{icon} {item['verification_status']}: {item['claim'][:60]}...", expanded=True):
                st.markdown(f"**Claim:** {item['claim']}")
                st.markdown(f"**Analysis:** :{color}[{item['correction']}]")
                if item['source_url'] != "N/A":
                    st.link_button("Read Source", item['source_url'])