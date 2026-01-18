import streamlit as st
import tempfile
import os
import re
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY MANAGEMENT ---
# Using standard retrieval; fallback is strictly for demo purposes if secrets aren't set
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")
except:
    OPENROUTER_API_KEY = ""
    TAVILY_API_KEY = ""

# --- DATA MODELS ---
class VerificationResult(BaseModel):
    claim: str = Field(description="The claim being checked")
    verification_status: str = Field(description="One of: 'Verified', 'Inaccurate', 'False'")
    correction: str = Field(description="A concise explanation citing the correct numbers/facts found.")
    source_url: str = Field(description="The single most authoritative URL found.")

class SearchQuery(BaseModel):
    query: str = Field(description="An optimized search query to check the claim")

class ClaimList(BaseModel):
    claims: List[str] = Field(description="A list of specific, checkable facts found in the text")

#backend

def extract_text_from_pdf(file):
    """Standard PDF extraction."""
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
    """
    Extracts strictly factual claims (numbers, dates, entities).
    Ignores fluff to save tokens and focus on hard facts.
    """
    parser = JsonOutputParser(pydantic_object=ClaimList)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert Fact Extraction Engine.
Your goal is to identify the **Hard Facts** in the document that need verification.

RULES:
1. Extract 5 to 10 of the most critical statistical, financial, or date-based claims.
2. Ignore generic marketing fluff (e.g., "We are the best").
3. Claims must be standalone sentences containing specific numbers, entities, or dates.
4. If a sentence has multiple facts, split them or keep the most verifiable one.

Return valid JSON.
"""),
        ("human", "Text Context:\n{text}\n\n{format_instructions}")
    ])

    chain = prompt | llm | parser
    try:
        return chain.invoke({
            "text": text[:15000], # Limit context to prevent overflow
            "format_instructions": parser.get_format_instructions()
        })["claims"]
    except Exception as e:
        st.error(f"Extraction Error: {e}")
        return []

def generate_search_query(claim, llm):
    """
    Converts a claim into a Google-optimized search query.
    Example: "Apple sold 50m units" -> "Apple iPhone sales figures Q4 2024 official data"
    """
    parser = JsonOutputParser(pydantic_object=SearchQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Search Engine Optimization expert. Convert the user's claim into a single, highly effective search query to find official data/statistics. Return JSON."),
        ("human", "Claim: {claim}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    try:
        return chain.invoke({"claim": claim, "format_instructions": parser.get_format_instructions()})["query"]
    except:
        return claim # Fallback to raw claim

def verify_claim(claim, llm, search_tool):
    """
    The Core Brain:
    1. Generates a specific search query.
    2. Searches Tavily.
    3. Uses LLM to reason (Chain-of-Thought) about the evidence vs. the claim.
    """
    
    # 1. Optimize Search
    search_query = generate_search_query(claim, llm)
    
    # 2. Perform Search
    try:
        raw_results = search_tool.invoke({"query": search_query})
        
        # Deduplicate and clean results
        seen_urls = set()
        clean_context = []
        for res in raw_results:
            if res['url'] not in seen_urls and len(res['content']) > 50:
                clean_context.append(f"Source ({res['url']}):\n{res['content']}")
                seen_urls.add(res['url'])
                if len(clean_context) >= 4: break # Top 4 sources max
        
        context_str = "\n\n".join(clean_context)
        
        if not clean_context:
            return VerificationResult(
                claim=claim, verification_status="False",
                correction="No public evidence found to support this specific claim.", source_url="N/A"
            )
            
    except Exception as e:
        return VerificationResult(
            claim=claim, verification_status="False",
            correction=f"Search tool error: {str(e)}", source_url="N/A"
        )

    # 3. LLM Adjudication (The Judge)
    parser = JsonOutputParser(pydantic_object=VerificationResult)
    today = datetime.now().strftime("%B %d, %Y")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are a Lead Fact-Checker for a major news wire. Your job is to strictly evaluate claims against search evidence.
Today's Date: {today}

**VERDICT RULES:**
1. **Verified**: The evidence *explicitly* confirms the numbers/facts in the claim. (Allow minor rounding differences, e.g., 10.1% vs 10%).
2. **Inaccurate**: The claim is *based* on truth but has specific errors (e.g., outdated statistics, wrong year, slightly exaggerated numbers). 
   *Example: Claim says "GDP is 5%", Evidence says "GDP is 3%".*
3. **False**: The claim is completely fabricated, debunked by evidence, or there is absolutely no proof in the search results.

**CRITICAL INSTRUCTIONS:**
- If the claim cites "current" data but the evidence shows it is from 3 years ago, mark as **Inaccurate** (Outdated).
- If the numbers are significantly different (e.g., Claim: $1B, Reality: $500M), mark as **False**.
- You must cite the *correct* real-time number in the 'correction' field.
- Return valid JSON only.
"""),
        ("human", """
CLAIM TO CHECK: "{claim}"

SEARCH EVIDENCE:
{context}

{format_instructions}
""")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "claim": claim,
            "context": context_str,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Fallback if LLM forgets URL
        if not result['source_url'] or result['source_url'] == "N/A":
            result['source_url'] = clean_context[0].split("\n")[0].replace("Source (", "").replace("):", "")
            
        return result
    except Exception:
        return VerificationResult(
            claim=claim, verification_status="False",
            correction="Could not verify due to processing error.", source_url="N/A"
        )


# --- UI LAYOUT (UNCHANGED FRONTEND) ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/verified-account.png", width=64)
    st.title("FactCheck AI")

    st.markdown("### ❓ What does this tool do?")
    st.markdown("""
This automated agent acts as a gatekeeper between drafts and publication.

**1. Extract:** Scans PDF documents to identify specific statistical claims, dates, and financial figures.

**2. Verify:** Cross-references each claim against multiple live web sources to ensure accuracy across timelines.

**3. Report:** Flags claims as:
* ✅ **Verified** (Matches reality)
* ⚠️ **Inaccurate** (Details wrong)
* ❌ **False** (Contradicts reality)
""")

    st.divider()
    st.caption("Powered by LangChain, OpenRouter & Tavily")

st.title("🛡️ Automated Fact-Checking Agent")

uploaded_file = st.file_uploader("📂 Upload PDF Report", type="pdf")

if uploaded_file:
    # Ensure keys are present
    if not OPENROUTER_API_KEY or not TAVILY_API_KEY:
        st.error("⚠️ Keys missing! Please add `OPENROUTER_API_KEY` and `TAVILY_API_KEY` to Streamlit Secrets.")
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
            st.write(f"Found {len(claims)} core factual claims.")

            results = []
            progress = st.progress(0)

            for i, claim in enumerate(claims):
                st.write(f"Checking: Claim {i+1}...")
                res = verify_claim(claim, llm, search)
                results.append(res)
                progress.progress((i + 1) / len(claims))

            status.update(label="Analysis Completed!", state="complete", expanded=False)

        # Ensure we have results before calculating metrics
        if results:
            verified = sum(1 for r in results if r["verification_status"] == "Verified")
            score = int((verified / len(results)) * 100)
        else:
            verified = 0
            score = 0

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Truth Score", f"{score}%")
        c2.metric("Claims Checked", len(results))
        c3.metric("Verified", verified)
        st.progress(score / 100)

        st.subheader("📝 Verification Details")
        for item in results:
            if item["verification_status"] == "Verified":
                color = "green"
                icon = "✅"
            elif item["verification_status"] == "Inaccurate":
                color = "orange"
                icon = "⚠️"
            else:
                color = "red"
                icon = "❌"

            with st.expander(f"{icon} {item['verification_status']}: {item['claim'][:60]}...", expanded=True):
                st.markdown(f"**Claim:** {item['claim']}")
                st.markdown(f"**Analysis:** :{color}[{item['correction']}]")
                if item["source_url"] and item["source_url"] != "N/A":
                    st.link_button("Read Source", item["source_url"])
