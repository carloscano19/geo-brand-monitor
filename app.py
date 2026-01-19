import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import re
import json
import base64
from urllib.parse import urlparse
from collections import Counter
import requests
from bs4 import BeautifulSoup

# LLM API clients (will be imported conditionally)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GEO/AIO Brand Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - CLEAN LIGHT THEME
# ============================================================================
st.markdown("""
<style>
    /* Main background - Pure white */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar - Very light gray */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Typography - Dark for readability */
    .stMarkdown, .stText, p, span, label {
        color: #111827 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 600;
    }

    /* Professional blue gradient buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        color: #111827;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #111827;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Dataframe */
    .dataframe {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }

    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': '',
        'anthropic': '',
        'perplexity': '',
        'gemini': '',
        'serpapi': ''
    }

if 'brand_name' not in st.session_state:
    st.session_state.brand_name = ''

if 'brand_domain' not in st.session_state:
    st.session_state.brand_domain = ''

if 'competitor_name' not in st.session_state:
    st.session_state.competitor_name = ''

# ============================================================================
# SIDEBAR - SETTINGS
# ============================================================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("🔑 API Keys")
    st.session_state.api_keys['openai'] = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_keys['openai'],
        type="password",
        help="Your OpenAI API key for GPT-4 models"
    )

    st.session_state.api_keys['anthropic'] = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_keys['anthropic'],
        type="password",
        help="Your Anthropic API key for Claude models"
    )

    st.session_state.api_keys['perplexity'] = st.text_input(
        "Perplexity API Key",
        value=st.session_state.api_keys['perplexity'],
        type="password",
        help="Your Perplexity API key"
    )

    st.session_state.api_keys['gemini'] = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_keys['gemini'],
        type="password",
        help="Your Google Gemini API key"
    )

    st.session_state.api_keys['serpapi'] = st.text_input(
        "SerpApi Key",
        value=st.session_state.api_keys['serpapi'],
        type="password",
        help="Your SerpApi key for Google AI Overviews monitoring"
    )

    st.markdown("---")

    st.subheader("🎯 Brand Information")
    st.session_state.brand_name = st.text_input(
        "Brand Name",
        value=st.session_state.brand_name,
        placeholder="e.g., FanTokenPro",
        help="The name of your brand to monitor"
    )

    st.session_state.brand_domain = st.text_input(
        "Brand Domain",
        value=st.session_state.brand_domain,
        placeholder="e.g., fantokenpro.com",
        help="Your brand's domain to detect citations"
    )

    st.session_state.competitor_name = st.text_input(
        "Competitor Name",
        value=st.session_state.competitor_name,
        placeholder="e.g., Socios.com",
        help="Name of your main competitor to track mentions"
    )

    st.markdown("---")

    st.subheader("💾 Database (GitHub)")
    if 'github_token' not in st.session_state:
        st.session_state.github_token = ""
    if 'github_repo' not in st.session_state:
        st.session_state.github_repo = "carloscano19/geo-brand-monitor"

    st.session_state.github_token = st.text_input(
        "GitHub Personal Token",
        value=st.session_state.github_token,
        type="password",
        help="Your GitHub Personal Access Token for saving audit history"
    )

    st.session_state.github_repo = st.text_input(
        "Repo Name",
        value=st.session_state.github_repo,
        placeholder="username/repo-name",
        help="GitHub repository in format: username/repo-name"
    )

    st.markdown("---")
    st.caption("🌍 GEO/AIO Brand Monitor v2.0 - Ultimate Edition")

# ============================================================================
# COUNTRY LIST (Shared between Tab 1 and Tab 2)
# ============================================================================
COUNTRY_LIST = [
    # Global
    "Global",
    # North America
    "USA",
    "Canada",
    "Mexico",
    # South America / LATAM
    "Argentina",
    "Brazil",
    "Chile",
    "Colombia",
    "Peru",
    "Venezuela",
    "Ecuador",
    "Bolivia",
    "Uruguay",
    "Paraguay",
    # Europe
    "United Kingdom",
    "Spain",
    "France",
    "Germany",
    "Italy",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Poland",
    "Czech Republic",
    "Austria",
    "Switzerland",
    "Ireland",
    "Greece",
    "Romania",
    # APAC
    "Australia",
    "Japan",
    "South Korea",
    "Singapore",
    "India",
    "China",
    "Thailand",
    "Indonesia",
    "Malaysia",
    "Philippines",
    "Vietnam",
    "New Zealand",
    # Middle East & Others
    "Turkey",
    "UAE",
    "Saudi Arabia",
    "Israel",
    "South Africa"
]

# ============================================================================
# HELPER FUNCTIONS FOR LLM CALLS
# ============================================================================
def generate_prompts_with_llm(topic, country, language, api_keys):
    """
    Generate prompts using OpenAI or Anthropic API.
    Returns a list of 20 queries or None if failed.
    """
    system_prompt = f"""You are an expert in GEO (Generative Engine Optimization).
For the given topic '{topic}', generate 20 distinct, high-intent search queries that a user in '{country}' speaking '{language}' would ask an AI Search Engine (like Perplexity or SearchGPT).
Include a mix of:
- Informational queries (What is...)
- Commercial investigation (Best X for Y...)
- Transactional (Price of...)
- Comparative (X vs Y...)
Do not number them, just provide the raw queries one per line."""

    # Try OpenAI first
    if api_keys['openai'] and openai:
        try:
            client = openai.OpenAI(api_key=api_keys['openai'])
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate queries for: {topic}"}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            queries = response.choices[0].message.content.strip().split('\n')
            # Clean up queries (remove numbering if present, empty lines)
            queries = [q.strip() for q in queries if q.strip()]
            queries = [q.lstrip('0123456789.-)> ') for q in queries]
            return queries[:20]  # Ensure we get exactly 20
        except openai.RateLimitError:
            st.error("⚠️ OpenAI Quota Exceeded: You've reached your API usage limit or have billing issues. Please check your OpenAI account billing and quota at https://platform.openai.com/account/billing")
            return None
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return None

    # Try Anthropic if OpenAI not available
    elif api_keys['anthropic'] and anthropic:
        try:
            client = anthropic.Anthropic(api_key=api_keys['anthropic'])
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                temperature=0.8,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Generate queries for: {topic}"}
                ]
            )
            queries = message.content[0].text.strip().split('\n')
            # Clean up queries
            queries = [q.strip() for q in queries if q.strip()]
            queries = [q.lstrip('0123456789.-)> ') for q in queries]
            return queries[:20]
        except Exception as e:
            st.error(f"Anthropic API Error: {str(e)}")
            return None

    return None


def extract_urls_from_text(text):
    """Extract URLs from text using regex."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


def count_brand_mentions(text, brand_name):
    """Count how many times the brand name appears in the text (case-insensitive)."""
    if not brand_name:
        return 0
    return text.lower().count(brand_name.lower())


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain if domain else None
    except:
        return None


def parse_analysis_block(response_text):
    """
    Parse the analysis block from LLM response.
    Returns dict with: answer, sentiment, competitor_mentioned, advice
    """
    # Split by separator
    if "|||ANALYSIS|||" in response_text:
        parts = response_text.split("|||ANALYSIS|||")
        answer = parts[0].strip()
        analysis_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse analysis block
        sentiment = "Neutral"
        competitor_mentioned = "No"
        advice = "No advice provided"

        if analysis_text:
            lines = analysis_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Sentiment:"):
                    sentiment = line.replace("Sentiment:", "").strip()
                elif line.startswith("Competitor_Mentioned:"):
                    competitor_mentioned = line.replace("Competitor_Mentioned:", "").strip()
                elif line.startswith("Advice:"):
                    advice = line.replace("Advice:", "").strip()

        return {
            'answer': answer,
            'sentiment': sentiment,
            'competitor_mentioned': competitor_mentioned,
            'advice': advice
        }
    else:
        # Fallback if separator not found
        return {
            'answer': response_text,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }


def clean_and_parse_json(response_text):
    """
    Robust JSON parser that handles different LLM response formats.
    Finds the first '{' and last '}' to extract valid JSON, ignoring preamble text.
    """
    # Remove markdown code blocks if present
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()

    # Find the first { and last }
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = response_text[first_brace:last_brace + 1]
        return json.loads(json_str)
    else:
        # If no braces found, try parsing as is
        return json.loads(response_text)


def call_perplexity_api(prompt, api_key, country):
    """
    Call Perplexity API and return response with citations.
    Returns dict with: answer, citations, brand_mentioned, sentiment, competitor_mentioned, advice
    """
    if not api_key:
        return {
            'error': 'No API Key',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )

        # Get brand and competitor names
        brand_name = st.session_state.brand_name
        competitor_name = st.session_state.competitor_name if st.session_state.competitor_name else "N/A"

        # Triple Engine System Prompt
        system_instruction = f"""Answer the user's question naturally. After your answer, append a strict analysis block using this format:

|||ANALYSIS|||
Sentiment: [Positive/Neutral/Negative]
Competitor_Mentioned: [Yes/No]
Advice: [One short actionable GEO tip to improve visibility for {brand_name}]

Context: The user's brand is "{brand_name}" and their competitor is "{competitor_name}". Analyze if the competitor was mentioned in your answer."""

        # Add country context to prompt
        contextualized_prompt = f"[Context: User in {country}] {system_instruction}\n\nUser question: {prompt}"

        response = client.chat.completions.create(
            model="sonar-reasoning",
            messages=[
                {"role": "user", "content": contextualized_prompt}
            ]
        )

        full_response = response.choices[0].message.content

        # Parse the analysis block
        parsed = parse_analysis_block(full_response)
        answer_text = parsed['answer']

        # Extract citations
        citations = extract_urls_from_text(full_response)

        # Check for brand mentions
        brand_mentioned = count_brand_mentions(answer_text, brand_name) > 0

        return {
            'answer': answer_text,
            'citations': citations,
            'brand_mentioned': brand_mentioned,
            'sentiment': parsed['sentiment'],
            'competitor_mentioned': parsed['competitor_mentioned'],
            'advice': parsed['advice'],
            'error': None
        }

    except openai.RateLimitError:
        return {
            'error': 'Quota Exceeded: Please check your Perplexity API billing and quota',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }
    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }


def call_gpt4o_api(prompt, api_key, country):
    """Call GPT-4o API and return response."""
    if not api_key:
        return {
            'error': 'No API Key',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }

    try:
        client = openai.OpenAI(api_key=api_key)

        # Get brand and competitor names
        brand_name = st.session_state.brand_name
        competitor_name = st.session_state.competitor_name if st.session_state.competitor_name else "N/A"

        # Triple Engine System Prompt
        system_instruction = f"""Answer the user's question naturally. After your answer, append a strict analysis block using this format:

|||ANALYSIS|||
Sentiment: [Positive/Neutral/Negative]
Competitor_Mentioned: [Yes/No]
Advice: [One short actionable GEO tip to improve visibility for {brand_name}]

Context: The user's brand is "{brand_name}" and their competitor is "{competitor_name}". Analyze if the competitor was mentioned in your answer."""

        contextualized_prompt = f"[Context: User in {country}] {system_instruction}\n\nUser question: {prompt}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": contextualized_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        full_response = response.choices[0].message.content

        # Parse the analysis block
        parsed = parse_analysis_block(full_response)
        answer_text = parsed['answer']

        # Extract any URLs from the answer
        citations = extract_urls_from_text(full_response)

        # Check for brand mentions
        brand_mentioned = count_brand_mentions(answer_text, brand_name) > 0

        return {
            'answer': answer_text,
            'citations': citations,
            'brand_mentioned': brand_mentioned,
            'sentiment': parsed['sentiment'],
            'competitor_mentioned': parsed['competitor_mentioned'],
            'advice': parsed['advice'],
            'error': None
        }

    except openai.RateLimitError:
        return {
            'error': 'Quota Exceeded: Please check your OpenAI API billing and quota at https://platform.openai.com/account/billing',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }
    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }


def call_claude_api(prompt, api_key, country):
    """Call Claude 3.5 API and return response."""
    if not api_key:
        return {
            'error': 'No API Key',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Get brand and competitor names
        brand_name = st.session_state.brand_name
        competitor_name = st.session_state.competitor_name if st.session_state.competitor_name else "N/A"

        # Triple Engine System Prompt
        system_instruction = f"""Answer the user's question naturally. After your answer, append a strict analysis block using this format:

|||ANALYSIS|||
Sentiment: [Positive/Neutral/Negative]
Competitor_Mentioned: [Yes/No]
Advice: [One short actionable GEO tip to improve visibility for {brand_name}]

Context: The user's brand is "{brand_name}" and their competitor is "{competitor_name}". Analyze if the competitor was mentioned in your answer."""

        contextualized_prompt = f"[Context: User in {country}] {system_instruction}\n\nUser question: {prompt}"

        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": contextualized_prompt}
            ]
        )

        full_response = message.content[0].text

        # Parse the analysis block
        parsed = parse_analysis_block(full_response)
        answer_text = parsed['answer']

        # Extract any URLs from the answer
        citations = extract_urls_from_text(full_response)

        # Check for brand mentions
        brand_mentioned = count_brand_mentions(answer_text, brand_name) > 0

        return {
            'answer': answer_text,
            'citations': citations,
            'brand_mentioned': brand_mentioned,
            'sentiment': parsed['sentiment'],
            'competitor_mentioned': parsed['competitor_mentioned'],
            'advice': parsed['advice'],
            'error': None
        }

    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }


def call_gemini_api(prompt, api_key, country):
    """Call Gemini 1.5 API and return response."""
    if not api_key:
        return {
            'error': 'No API Key',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }

    if not genai:
        return {
            'error': 'Gemini library not installed',
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Get brand and competitor names
        brand_name = st.session_state.brand_name
        competitor_name = st.session_state.competitor_name if st.session_state.competitor_name else "N/A"

        # Triple Engine System Prompt
        system_instruction = f"""Answer the user's question naturally. After your answer, append a strict analysis block using this format:

|||ANALYSIS|||
Sentiment: [Positive/Neutral/Negative]
Competitor_Mentioned: [Yes/No]
Advice: [One short actionable GEO tip to improve visibility for {brand_name}]

Context: The user's brand is "{brand_name}" and their competitor is "{competitor_name}". Analyze if the competitor was mentioned in your answer."""

        contextualized_prompt = f"[Context: User in {country}] {system_instruction}\n\nUser question: {prompt}"

        response = model.generate_content(contextualized_prompt)
        full_response = response.text

        # Parse the analysis block
        parsed = parse_analysis_block(full_response)
        answer_text = parsed['answer']

        # Extract any URLs from the answer
        citations = extract_urls_from_text(full_response)

        # Check for brand mentions
        brand_mentioned = count_brand_mentions(answer_text, brand_name) > 0

        return {
            'answer': answer_text,
            'citations': citations,
            'brand_mentioned': brand_mentioned,
            'sentiment': parsed['sentiment'],
            'competitor_mentioned': parsed['competitor_mentioned'],
            'advice': parsed['advice'],
            'error': None
        }

    except Exception as e:
        return {
            'error': str(e),
            'answer': '',
            'citations': [],
            'brand_mentioned': False,
            'sentiment': 'Neutral',
            'competitor_mentioned': 'No',
            'advice': 'No advice provided'
        }


def run_audit(prompts, models, country, api_keys):
    """
    Run the actual audit by calling each model for each prompt.
    Returns a list of result dictionaries.
    """
    results = []

    for prompt in prompts:
        for model in models:
            # Determine which API to call
            result = None

            if model == "Perplexity":
                result = call_perplexity_api(prompt, api_keys['perplexity'], country)
            elif model == "GPT-4o":
                result = call_gpt4o_api(prompt, api_keys['openai'], country)
            elif model == "Claude 3.5":
                result = call_claude_api(prompt, api_keys['anthropic'], country)
            elif model == "Gemini 1.5":
                result = call_gemini_api(prompt, api_keys['gemini'], country)

            if result:
                # Format the result for the dataframe
                answer_snippet = result['answer'][:100] + '...' if len(result['answer']) > 100 else result['answer']

                if result['error']:
                    # Error occurred
                    results.append({
                        'Model': model,
                        'Country': country,
                        'Prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                        'Brand Mentioned?': 'Error',
                        'Source Count': 0,
                        'Sources List': result['error'],
                        'Answer Snippet': f"Error: {result['error']}"
                    })
                else:
                    # Success
                    sources_list = ', '.join(result['citations'][:3]) if result['citations'] else (
                        "N/A (Standard Model)" if model != "Perplexity" else "No citations found"
                    )

                    results.append({
                        'Model': model,
                        'Country': country,
                        'Prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                        'Brand Mentioned?': 'Yes' if result['brand_mentioned'] else 'No',
                        'Source Count': len(result['citations']),
                        'Sources List': sources_list,
                        'Answer Snippet': answer_snippet
                    })

    return results


# ============================================================================
# GITHUB STORAGE HELPER FUNCTIONS
# ============================================================================
def save_to_github(data_row, repo, token):
    """
    Save a new row to history.csv in the GitHub repository.

    Args:
        data_row (dict): Dictionary with keys: date, brand, keyword, model, score, missing_topics
        repo (str): GitHub repository in format 'username/repo'
        token (str): GitHub Personal Access Token

    Returns:
        tuple: (success: bool, message: str)
    """
    if not token or not repo:
        return False, "GitHub token and repo are required"

    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{repo}/contents/history.csv"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        # Try to fetch existing file
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            # File exists, get content and SHA
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')
            sha = file_data['sha']

            # Append new row
            new_row = f"{data_row['date']},{data_row['brand']},{data_row['keyword']},{data_row['model']},{data_row['score']},\"{data_row['missing_topics']}\"\n"
            updated_content = content + new_row

        elif response.status_code == 404:
            # File doesn't exist, create with header
            sha = None
            header = "date,brand,keyword,model,score,missing_topics\n"
            new_row = f"{data_row['date']},{data_row['brand']},{data_row['keyword']},{data_row['model']},{data_row['score']},\"{data_row['missing_topics']}\"\n"
            updated_content = header + new_row

        else:
            return False, f"Error fetching file: {response.status_code}"

        # Encode updated content
        encoded_content = base64.b64encode(updated_content.encode('utf-8')).decode('utf-8')

        # Prepare commit data
        commit_data = {
            "message": f"Add audit result for {data_row['keyword']}",
            "content": encoded_content
        }

        if sha:
            commit_data["sha"] = sha

        # Update/create file
        update_response = requests.put(api_url, headers=headers, json=commit_data)

        if update_response.status_code in [200, 201]:
            return True, "Successfully saved to GitHub"
        else:
            return False, f"Error updating file: {update_response.status_code} - {update_response.text}"

    except Exception as e:
        return False, f"Exception: {str(e)}"


def load_from_github(repo, token):
    """
    Load history.csv from GitHub repository.

    Args:
        repo (str): GitHub repository in format 'username/repo'
        token (str): GitHub Personal Access Token

    Returns:
        tuple: (success: bool, data: pd.DataFrame or error message)
    """
    if not token or not repo:
        return False, "GitHub token and repo are required"

    # GitHub API endpoint
    api_url = f"https://api.github.com/repos/{repo}/contents/history.csv"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            # File exists, decode content
            file_data = response.json()
            content = base64.b64decode(file_data['content']).decode('utf-8')

            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(content))

            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            return True, df

        elif response.status_code == 404:
            return False, "History file not found. Save some audits first!"
        else:
            return False, f"Error: {response.status_code}"

    except Exception as e:
        return False, f"Exception: {str(e)}"


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("🌍 GEO/AIO Brand Monitor - Ultimate Edition")
st.markdown("**Advanced LLM Brand Monitoring with Sentiment Analysis, Competitor Tracking & AI-Powered GEO Insights**")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏭 PROMPT GENERATOR", "🌍 MARKET MONITOR (AUDIT)", "🔍 GOOGLE AI MONITOR", "✨ GEO CONTENT AUDITOR", "📈 History"])

# ============================================================================
# TAB 1: PROMPT GENERATOR
# ============================================================================
with tab1:
    st.header("🏭 Prompt Generator")
    st.markdown("Generate search query variations from your topics to test brand visibility across LLM models.")

    col1, col2 = st.columns([2, 1])

    with col1:
        topics_input = st.text_area(
            "Enter Topics (one per line)",
            height=150,
            placeholder="blockchain gaming\nfan engagement\ncrypto rewards\ndigital collectibles",
            help="Enter each topic on a new line"
        )

        # Target Language and Country
        col_lang, col_country = st.columns(2)

        with col_lang:
            target_language = st.selectbox(
                "Target Language",
                options=[
                    "English",
                    "Spanish",
                    "Portuguese",
                    "French",
                    "German",
                    "Italian"
                ],
                help="Language for the generated queries"
            )

        with col_country:
            target_country = st.selectbox(
                "Target Country",
                options=COUNTRY_LIST,
                help="Geographic context for the queries"
            )

    with col2:
        st.info("""
        **How it works:**

        1. Enter your topics
        2. Select language & country
        3. Click generate
        4. Get 20 AI-generated queries
        5. Download as CSV

        **Tip:** Use specific topics related to your brand's niche.
        """)

    if st.button("✨ Generate Prompt Variations", key="generate_prompts"):
        if not topics_input.strip():
            st.error("⚠️ Please enter at least one topic.")
        else:
            # Check if API keys are available
            has_api_key = bool(st.session_state.api_keys['openai'] or st.session_state.api_keys['anthropic'])

            if not has_api_key:
                st.warning("🤖 **No API Key detected!** Please enter your OpenAI or Anthropic API Key in the sidebar to activate the AI brain and generate intelligent queries.")
                st.info("💡 **Tip:** Get your API keys from:\n- OpenAI: https://platform.openai.com/api-keys\n- Anthropic: https://console.anthropic.com/")
            else:
                topics = [t.strip() for t in topics_input.split('\n') if t.strip()]

                with st.spinner(f"🧠 AI is generating {len(topics) * 20} intelligent queries..."):
                    all_generated_prompts = []

                    # Generate queries for each topic
                    for topic in topics:
                        queries = generate_prompts_with_llm(
                            topic=topic,
                            country=target_country,
                            language=target_language,
                            api_keys=st.session_state.api_keys
                        )

                        if queries:
                            for i, query in enumerate(queries):
                                all_generated_prompts.append({
                                    'ID': len(all_generated_prompts) + 1,
                                    'Topic': topic,
                                    'Language': target_language,
                                    'Country': target_country,
                                    'Generated Query': query
                                })

                if all_generated_prompts:
                    df_prompts = pd.DataFrame(all_generated_prompts)

                    st.success(f"✅ Generated {len(all_generated_prompts)} AI-powered prompt variations!")

                    st.dataframe(df_prompts, use_container_width=True, height=400)

                    # Download button
                    csv = df_prompts.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prompts as CSV",
                        data=csv,
                        file_name=f"ai_prompts_{target_country}_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ Failed to generate prompts. Please check your API keys and try again.")

# ============================================================================
# TAB 2: MARKET MONITOR (AUDIT)
# ============================================================================
with tab2:
    st.header("🌍 Market Monitor (Audit)")
    st.markdown("Test your prompts across multiple LLM models and geographic regions to monitor brand visibility.")

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        prompts_input = st.text_area(
            "Enter Prompts to Test (one per line)",
            height=150,
            placeholder="What are the best fan engagement platforms?\nHow to monetize sports fandom?\nTop blockchain gaming solutions",
            help="Enter each prompt on a new line"
        )

    with col2:
        models_selected = st.multiselect(
            "Select Models",
            options=["GPT-4o", "Claude 3.5", "Gemini 1.5", "Perplexity"],
            default=["GPT-4o", "Claude 3.5"],
            help="Choose which LLM models to test"
        )

        country_selected = st.selectbox(
            "Simulation Country",
            options=COUNTRY_LIST,
            help="Geographic context for the audit"
        )

    st.markdown("---")

    if st.button("🚀 START AUDIT", key="start_audit"):
        if not prompts_input.strip():
            st.error("⚠️ Please enter at least one prompt to test.")
        elif not models_selected:
            st.error("⚠️ Please select at least one model.")
        elif not st.session_state.brand_name:
            st.warning("⚠️ Please enter your Brand Name in the sidebar to detect brand mentions.")
        else:
            prompts = [p.strip() for p in prompts_input.split('\n') if p.strip()]

            # Check if we have the necessary API keys
            missing_keys = []
            for model in models_selected:
                if model == "GPT-4o" and not st.session_state.api_keys['openai']:
                    missing_keys.append("OpenAI (for GPT-4o)")
                elif model == "Claude 3.5" and not st.session_state.api_keys['anthropic']:
                    missing_keys.append("Anthropic (for Claude 3.5)")
                elif model == "Gemini 1.5" and not st.session_state.api_keys['gemini']:
                    missing_keys.append("Google (for Gemini 1.5)")
                elif model == "Perplexity" and not st.session_state.api_keys['perplexity']:
                    missing_keys.append("Perplexity")

            if missing_keys:
                st.warning(f"⚠️ Missing API keys for: {', '.join(missing_keys)}. These models will show errors.")

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_iterations = len(prompts) * len(models_selected)
            current_iteration = 0

            results = []

            # Run actual audit
            for i, prompt in enumerate(prompts):
                for model in models_selected:
                    current_iteration += 1
                    progress = current_iteration / total_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing: {model} | Prompt {i + 1}/{len(prompts)}")

                    # Call the appropriate API
                    if model == "Perplexity":
                        result = call_perplexity_api(prompt, st.session_state.api_keys['perplexity'], country_selected)
                    elif model == "GPT-4o":
                        result = call_gpt4o_api(prompt, st.session_state.api_keys['openai'], country_selected)
                    elif model == "Claude 3.5":
                        result = call_claude_api(prompt, st.session_state.api_keys['anthropic'], country_selected)
                    elif model == "Gemini 1.5":
                        result = call_gemini_api(prompt, st.session_state.api_keys['gemini'], country_selected)

                    # Format result
                    if result:
                        answer_snippet = result['answer'][:100] + '...' if len(result['answer']) > 100 else result['answer']

                        if result['error']:
                            results.append({
                                'Model': model,
                                'Country': country_selected,
                                'Sentiment': 'Error',
                                'Brand Mentioned?': 'Error',
                                'Competitor Mentioned?': 'Error',
                                'GEO Advice': result.get('advice', 'N/A'),
                                'Prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                                'Full Prompt': prompt,
                                'Source Count': 0,
                                'Sources List': result['error'],
                                'Full Citations': [],
                                'Answer Snippet': f"Error: {result['error']}",
                                'Full Answer': f"Error: {result['error']}"
                            })
                        else:
                            sources_list = ', '.join(result['citations'][:3]) if result['citations'] else (
                                "N/A (Standard Model)" if model != "Perplexity" else "No citations found"
                            )

                            results.append({
                                'Model': model,
                                'Country': country_selected,
                                'Sentiment': result.get('sentiment', 'Neutral'),
                                'Brand Mentioned?': 'Yes' if result['brand_mentioned'] else 'No',
                                'Competitor Mentioned?': result.get('competitor_mentioned', 'No'),
                                'GEO Advice': result.get('advice', 'No advice provided'),
                                'Prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                                'Full Prompt': prompt,
                                'Source Count': len(result['citations']),
                                'Sources List': sources_list,
                                'Full Citations': result['citations'],
                                'Answer Snippet': answer_snippet,
                                'Full Answer': result['answer']
                            })

            progress_bar.progress(1.0)
            status_text.text("✅ Audit completed!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # Create results dataframe
            df_results = pd.DataFrame(results)

            # Calculate KPIs based on REAL data
            total_tests = len([r for r in results if r['Brand Mentioned?'] != 'Error'])
            brand_mentions = sum(1 for r in results if r['Brand Mentioned?'] == 'Yes')
            source_citations = sum(r['Source Count'] for r in results if r['Brand Mentioned?'] != 'Error')
            share_of_voice = (brand_mentions / total_tests * 100) if total_tests > 0 else 0
            citation_rate = (source_citations / total_tests * 100) if total_tests > 0 else 0

            # Display KPI Cards
            st.markdown("### 📊 Audit Summary")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Share of Voice</div>
                    <div class="kpi-value">{share_of_voice:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Brand Mentions</div>
                    <div class="kpi-value">{brand_mentions}/{total_tests}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Source Citations</div>
                    <div class="kpi-value">{source_citations}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Avg Citations</div>
                    <div class="kpi-value">{citation_rate:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ============================================================================
            # VISUAL ANALYTICS SECTION
            # ============================================================================
            st.markdown("### 📊 Visual Analytics")

            chart_col1, chart_col2, chart_col3 = st.columns(3)

            with chart_col1:
                # Chart 1: Share of Voice - Brand vs Competitor Comparison
                if results:
                    model_stats = []
                    for model in set(r['Model'] for r in results):
                        model_results = [r for r in results if r['Model'] == model and r['Brand Mentioned?'] != 'Error']
                        if model_results:
                            brand_mentions = sum(1 for r in model_results if r['Brand Mentioned?'] == 'Yes')
                            competitor_mentions = sum(1 for r in model_results if r.get('Competitor Mentioned?') == 'Yes')
                            total = len(model_results)
                            brand_pct = (brand_mentions / total * 100) if total > 0 else 0
                            competitor_pct = (competitor_mentions / total * 100) if total > 0 else 0
                            model_stats.append({
                                'Model': model,
                                'Brand': brand_pct,
                                'Competitor': competitor_pct
                            })

                    if model_stats:
                        models = [s['Model'] for s in model_stats]
                        brand_values = [s['Brand'] for s in model_stats]
                        competitor_values = [s['Competitor'] for s in model_stats]

                        fig1 = go.Figure(data=[
                            go.Bar(name='My Brand', x=models, y=brand_values, marker_color='#10b981',
                                   text=[f"{v:.1f}%" for v in brand_values], textposition='outside'),
                            go.Bar(name='Competitor', x=models, y=competitor_values, marker_color='#ef4444',
                                   text=[f"{v:.1f}%" for v in competitor_values], textposition='outside')
                        ])

                        fig1.update_layout(
                            title="Share of Voice: Brand vs Competitor",
                            xaxis_title="Model",
                            yaxis_title="Mention Rate (%)",
                            barmode='group',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#111827', size=11),
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No data yet")

            with chart_col2:
                # Chart 2: Sentiment Distribution (Pie Chart)
                if results:
                    sentiment_counts = {
                        'Positive': 0,
                        'Neutral': 0,
                        'Negative': 0
                    }

                    for r in results:
                        if r.get('Sentiment') and r['Sentiment'] != 'Error':
                            sentiment = r['Sentiment']
                            if sentiment in sentiment_counts:
                                sentiment_counts[sentiment] += 1

                    if sum(sentiment_counts.values()) > 0:
                        fig2 = go.Figure(data=[go.Pie(
                            labels=list(sentiment_counts.keys()),
                            values=list(sentiment_counts.values()),
                            marker=dict(colors=['#10b981', '#fbbf24', '#ef4444']),
                            hole=0.3,
                            textinfo='label+percent',
                            textposition='auto'
                        )])

                        fig2.update_layout(
                            title="Sentiment Distribution",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#111827', size=11),
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No sentiment data yet")
                else:
                    st.info("No data yet")

            with chart_col3:
                # Chart 3: Top Cited Sources
                if results:
                    # Extract all domains from citations
                    all_domains = []
                    for r in results:
                        if r['Brand Mentioned?'] != 'Error' and r.get('Full Citations'):
                            for citation in r['Full Citations']:
                                domain = extract_domain(citation)
                                if domain:
                                    all_domains.append(domain)

                    if all_domains:
                        # Count domain frequencies
                        domain_counts = Counter(all_domains)
                        top_sources = domain_counts.most_common(5)

                        source_data = pd.DataFrame(top_sources, columns=['Domain', 'Count'])

                        fig3 = px.bar(
                            source_data,
                            y='Domain',
                            x='Count',
                            orientation='h',
                            title='Top 5 Cited Sources',
                            color='Count',
                            color_continuous_scale=['#764ba2', '#667eea']
                        )
                        fig3.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(color='#111827', size=11),
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("No citations found")
                else:
                    st.info("No data yet")

            st.markdown("---")

            # ============================================================================
            # DETAILED RESULTS WITH EXPANDERS
            # ============================================================================
            st.markdown("### 📋 Detailed Results")

            if results:
                for idx, result in enumerate(results):
                    # Create summary for expander label
                    if result['Brand Mentioned?'] == 'Yes':
                        mention_emoji = "✅"
                        mention_color = "🟢"
                    elif result['Brand Mentioned?'] == 'No':
                        mention_emoji = "❌"
                        mention_color = "🔴"
                    else:
                        mention_emoji = "⚠️"
                        mention_color = "🟡"

                    summary = f"{mention_emoji} {result['Model']} | {result['Prompt']} | {mention_color}"

                    with st.expander(summary, expanded=False):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown(f"**🤖 Model:** {result['Model']}")
                            st.markdown(f"**🌍 Country:** {result['Country']}")
                            st.markdown(f"**🎯 Brand Mentioned:** {result['Brand Mentioned?']}")
                            st.markdown(f"**🏆 Competitor Mentioned:** {result.get('Competitor Mentioned?', 'N/A')}")

                        with col_b:
                            st.markdown(f"**💭 Sentiment:** {result.get('Sentiment', 'N/A')}")
                            st.markdown(f"**📊 Source Count:** {result['Source Count']}")
                            st.markdown(f"**🔗 Top Sources:** {result['Sources List']}")

                        st.markdown("---")

                        # GEO Advice in styled box
                        if result.get('GEO Advice') and result['GEO Advice'] != 'No advice provided':
                            st.warning(f"🤖 **GEO ADVICE:** {result['GEO Advice']}")

                        st.markdown(f"**❓ Full Prompt:**")
                        st.info(result['Full Prompt'])

                        st.markdown("**💬 Full Answer:**")
                        st.markdown(result['Full Answer'])
            else:
                st.info("No results to display. Run an audit to see detailed results.")

            st.markdown("---")

            # Download button
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"brand_audit_{country_selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ============================================================================
# TAB 3: GOOGLE AI MONITOR
# ============================================================================
with tab3:
    st.header("🔍 Google AI Overviews (SGE) & FAQs")
    st.markdown("Monitor Google's AI Overviews, People Also Ask, and organic rankings for your brand using SerpApi.")

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        queries_input = st.text_area(
            "Enter Search Queries (one per line)",
            height=150,
            placeholder="best fan engagement platforms\nblockchain gaming solutions\nfan token platforms",
            help="Enter each search query on a new line"
        )

    with col2:
        # Country Selector
        google_country = st.selectbox(
            "Target Country",
            options=COUNTRY_LIST,
            help="Geographic location for Google search results",
            key="google_ai_country"
        )

        st.info("""
        **How it works:**

        1. Enter your queries
        2. Select target country
        3. Click "Analyze"
        4. See AI Overviews & PAA
        5. Check brand visibility

        **Tip:** Monitor queries where your brand should appear.
        """)

    st.markdown("---")

    if st.button("🚀 Analyze Google Results", key="analyze_google"):
        if not queries_input.strip():
            st.error("⚠️ Please enter at least one search query.")
        elif not st.session_state.api_keys['serpapi']:
            st.warning("⚠️ Please enter your SerpApi Key in the sidebar.")
            st.info("💡 **Tip:** Get your SerpApi key from: https://serpapi.com/")
        else:
            queries = [q.strip() for q in queries_input.split('\n') if q.strip()]

            # Country code mapping for SerpApi
            country_code_map = {
                "Global": "us",
                "USA": "us",
                "Canada": "ca",
                "Mexico": "mx",
                "Argentina": "ar",
                "Brazil": "br",
                "Chile": "cl",
                "Colombia": "co",
                "Peru": "pe",
                "Venezuela": "ve",
                "Ecuador": "ec",
                "Bolivia": "bo",
                "Uruguay": "uy",
                "Paraguay": "py",
                "United Kingdom": "uk",
                "Spain": "es",
                "France": "fr",
                "Germany": "de",
                "Italy": "it",
                "Portugal": "pt",
                "Netherlands": "nl",
                "Belgium": "be",
                "Sweden": "se",
                "Norway": "no",
                "Denmark": "dk",
                "Finland": "fi",
                "Poland": "pl",
                "Czech Republic": "cz",
                "Austria": "at",
                "Switzerland": "ch",
                "Ireland": "ie",
                "Greece": "gr",
                "Romania": "ro",
                "Australia": "au",
                "Japan": "jp",
                "South Korea": "kr",
                "Singapore": "sg",
                "India": "in",
                "China": "cn",
                "Thailand": "th",
                "Indonesia": "id",
                "Malaysia": "my",
                "Philippines": "ph",
                "Vietnam": "vn",
                "New Zealand": "nz",
                "Turkey": "tr",
                "UAE": "ae",
                "Saudi Arabia": "sa",
                "Israel": "il",
                "South Africa": "za"
            }

            # Language code mapping
            language_code_map = {
                "Global": "en",
                "USA": "en",
                "Canada": "en",
                "Mexico": "es",
                "Argentina": "es",
                "Brazil": "pt",
                "Chile": "es",
                "Colombia": "es",
                "Peru": "es",
                "Venezuela": "es",
                "Ecuador": "es",
                "Bolivia": "es",
                "Uruguay": "es",
                "Paraguay": "es",
                "United Kingdom": "en",
                "Spain": "es",
                "France": "fr",
                "Germany": "de",
                "Italy": "it",
                "Portugal": "pt",
                "Netherlands": "nl",
                "Belgium": "nl",
                "Sweden": "sv",
                "Norway": "no",
                "Denmark": "da",
                "Finland": "fi",
                "Poland": "pl",
                "Czech Republic": "cs",
                "Austria": "de",
                "Switzerland": "de",
                "Ireland": "en",
                "Greece": "el",
                "Romania": "ro",
                "Australia": "en",
                "Japan": "ja",
                "South Korea": "ko",
                "Singapore": "en",
                "India": "en",
                "China": "zh",
                "Thailand": "th",
                "Indonesia": "id",
                "Malaysia": "ms",
                "Philippines": "en",
                "Vietnam": "vi",
                "New Zealand": "en",
                "Turkey": "tr",
                "UAE": "ar",
                "Saudi Arabia": "ar",
                "Israel": "he",
                "South Africa": "en"
            }

            gl_code = country_code_map.get(google_country, "us")
            hl_code = language_code_map.get(google_country, "en")

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            brand_domain = st.session_state.brand_domain

            for i, query in enumerate(queries):
                progress = (i + 1) / len(queries)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing query {i + 1}/{len(queries)}: {query[:50]}...")

                try:
                    if GoogleSearch:
                        # Call SerpApi
                        search = GoogleSearch({
                            "q": query,
                            "gl": gl_code,
                            "hl": hl_code,
                            "api_key": st.session_state.api_keys['serpapi']
                        })

                        search_results = search.get_dict()

                        # Parse AI Overview
                        ai_triggered = "No"
                        ai_text = "N/A"
                        ai_sources = "N/A"
                        paa_questions = "N/A"
                        brand_in_organic = "Not Found"
                        organic_position = "N/A"

                        # Check for AI Overview
                        if 'ai_overview' in search_results:
                            ai_triggered = "Yes"
                            ai_overview = search_results['ai_overview']

                            # Extract AI text
                            if isinstance(ai_overview, dict):
                                ai_text = ai_overview.get('text', ai_overview.get('snippet', 'N/A'))[:200]
                            elif isinstance(ai_overview, str):
                                ai_text = ai_overview[:200]

                            # Extract AI sources
                            ai_source_list = []
                            if isinstance(ai_overview, dict) and 'sources' in ai_overview:
                                for source in ai_overview['sources'][:3]:
                                    if isinstance(source, dict):
                                        link = source.get('link', source.get('url', ''))
                                        if link:
                                            ai_source_list.append(link)
                                    elif isinstance(source, str):
                                        ai_source_list.append(source)

                            if ai_source_list:
                                ai_sources = ', '.join(ai_source_list)
                            else:
                                ai_sources = "No sources found"

                        # Check for People Also Ask (PAA)
                        if 'related_questions' in search_results:
                            paa_list = []
                            for rq in search_results['related_questions'][:3]:
                                if isinstance(rq, dict):
                                    question = rq.get('question', rq.get('title', ''))
                                    if question:
                                        paa_list.append(question)
                                elif isinstance(rq, str):
                                    paa_list.append(rq)

                            if paa_list:
                                paa_questions = ' | '.join(paa_list)

                        # Check organic results for brand domain
                        if 'organic_results' in search_results and brand_domain:
                            for idx, result in enumerate(search_results['organic_results'][:10]):
                                if isinstance(result, dict):
                                    result_link = result.get('link', result.get('url', ''))
                                    result_domain = extract_domain(result_link)

                                    if result_domain and brand_domain.lower() in result_domain.lower():
                                        brand_in_organic = "Found"
                                        organic_position = f"Position {idx + 1}"
                                        break

                        results.append({
                            'Query': query,
                            'AI Overview Triggered?': ai_triggered,
                            'AI Text Snippet': ai_text,
                            'AI Sources': ai_sources,
                            'PAA Questions': paa_questions,
                            'Brand in Organic': brand_in_organic,
                            'Organic Position': organic_position
                        })

                    else:
                        results.append({
                            'Query': query,
                            'AI Overview Triggered?': 'Error',
                            'AI Text Snippet': 'SerpApi library not installed',
                            'AI Sources': 'N/A',
                            'PAA Questions': 'N/A',
                            'Brand in Organic': 'N/A',
                            'Organic Position': 'N/A'
                        })

                except Exception as e:
                    results.append({
                        'Query': query,
                        'AI Overview Triggered?': 'Error',
                        'AI Text Snippet': str(e),
                        'AI Sources': 'N/A',
                        'PAA Questions': 'N/A',
                        'Brand in Organic': 'N/A',
                        'Organic Position': 'N/A'
                    })

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            progress_bar.progress(1.0)
            status_text.text("✅ Analysis completed!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # Create results dataframe
            df_google_results = pd.DataFrame(results)

            # Calculate KPIs
            total_queries = len(results)
            ai_triggered_count = sum(1 for r in results if r['AI Overview Triggered?'] == 'Yes')
            ai_trigger_rate = (ai_triggered_count / total_queries * 100) if total_queries > 0 else 0
            brand_found_count = sum(1 for r in results if r['Brand in Organic'] == 'Found')
            brand_visibility_rate = (brand_found_count / total_queries * 100) if total_queries > 0 else 0

            # Display KPI Cards
            st.markdown("### 📊 Google AI Overview Summary")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Queries</div>
                    <div class="kpi-value">{total_queries}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">AI Overview Rate</div>
                    <div class="kpi-value">{ai_trigger_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">AI Triggered</div>
                    <div class="kpi-value">{ai_triggered_count}/{total_queries}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Brand Visibility</div>
                    <div class="kpi-value">{brand_visibility_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Display results with color coding
            st.markdown("### 📋 Detailed Results")

            # Apply color styling to the dataframe
            def highlight_ai_trigger(row):
                if row['AI Overview Triggered?'] == 'Yes':
                    return ['background-color: #d1fae5; color: #065f46'] * len(row)
                elif row['AI Overview Triggered?'] == 'No':
                    return ['background-color: #f3f4f6; color: #4b5563'] * len(row)
                else:
                    return ['background-color: #fee2e2; color: #991b1b'] * len(row)

            styled_df = df_google_results.style.apply(highlight_ai_trigger, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

            st.markdown("---")

            # Detailed expandable results
            st.markdown("### 🔍 Detailed Query Analysis")

            for idx, result in enumerate(results):
                if result['AI Overview Triggered?'] == 'Yes':
                    emoji = "✅"
                    color = "🟢"
                elif result['AI Overview Triggered?'] == 'No':
                    emoji = "⚪"
                    color = "⚫"
                else:
                    emoji = "⚠️"
                    color = "🔴"

                summary = f"{emoji} {result['Query']} | AI: {result['AI Overview Triggered?']} {color}"

                with st.expander(summary, expanded=False):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown(f"**🔍 Query:** {result['Query']}")
                        st.markdown(f"**🤖 AI Overview:** {result['AI Overview Triggered?']}")
                        st.markdown(f"**🏆 Brand Found:** {result['Brand in Organic']}")
                        st.markdown(f"**📊 Position:** {result['Organic Position']}")

                    with col_b:
                        st.markdown(f"**💬 AI Text:**")
                        st.info(result['AI Text Snippet'])

                    st.markdown("---")
                    st.markdown(f"**🔗 AI Sources:** {result['AI Sources']}")
                    st.markdown(f"**❓ PAA Questions:** {result['PAA Questions']}")

            st.markdown("---")

            # Download button
            csv = df_google_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Google AI Results as CSV",
                data=csv,
                file_name=f"google_ai_monitor_{google_country}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ============================================================================
# TAB 4: GEO CONTENT AUDITOR
# ============================================================================
with tab4:
    st.header("✨ GEO Content Auditor")
    st.markdown("**Content Gap Analysis** - Compare your URL content against the AI's Perfect Answer")
    st.markdown("---")

    # INSTRUCTION BOX AT THE TOP
    st.info("""
    ### 🧠 How this GEO Auditor Works
    1. **Scrape:** We extract the visible content from your provided URL using Python.
    2. **Generate Ideal:** We ask the selected AI Model to generate the 'Perfect Answer' for your target keyword. **Perplexity uses Live Web Data** for its ideal answer.
    3. **Gap Analysis:** The AI acts as an impartial judge, comparing your content vs. the ideal answer to identify missing topics, keywords, and semantic gaps.
    """)

    st.markdown("---")

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        target_keyword = st.text_input(
            "Target Keyword/Question",
            placeholder="best fan tokens",
            help="The keyword or question you want to rank for"
        )

        target_url = st.text_input(
            "Your URL",
            placeholder="https://fantoken.com/guide",
            help="The URL you want to audit"
        )

        audit_region = st.selectbox(
            "Target Region",
            options=COUNTRY_LIST,
            help="Geographic context for the ideal answer",
            key="content_audit_region"
        )

        # MODEL SELECTOR
        auditor_model = st.selectbox(
            "Select Auditor Model",
            options=["GPT-4o (OpenAI)", "Claude 3.5 Sonnet (Anthropic)", "Gemini 1.5 Pro (Google)", "Perplexity (Sonar/Online)"],
            help="Choose which AI model to use for content analysis"
        )

    with col2:
        st.info("""
        **How it works:**

        1. Enter your target keyword
        2. Provide your URL to audit
        3. Select target region
        4. Select AI model
        5. Click "Audit My Content"
        6. Get actionable gap analysis

        **Tip:** This reveals what's missing in your content to rank better.
        """)

    st.markdown("---")

    if st.button("🔍 Audit My Content", key="audit_content"):
        if not target_keyword.strip():
            st.error("⚠️ Please enter a target keyword or question.")
        elif not target_url.strip():
            st.error("⚠️ Please enter a URL to audit.")
        elif "GPT-4o" in auditor_model and not st.session_state.api_keys['openai']:
            st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to use GPT-4o for content analysis.")
            st.info("💡 **Tip:** Get your API key from: https://platform.openai.com/api-keys")
        elif "Claude" in auditor_model and not st.session_state.api_keys['anthropic']:
            st.warning("⚠️ Please enter your Anthropic API Key in the sidebar to use Claude 3.5 Sonnet for content analysis.")
            st.info("💡 **Tip:** Get your API key from: https://console.anthropic.com/")
        elif "Gemini" in auditor_model and not st.session_state.api_keys['gemini']:
            st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to use Gemini 1.5 Pro for content analysis.")
            st.info("💡 **Tip:** Get your API key from: https://aistudio.google.com/app/apikey")
        elif "Perplexity" in auditor_model and not st.session_state.api_keys['perplexity']:
            st.warning("⚠️ Please enter your Perplexity API Key in the sidebar to use Perplexity for content analysis.")
            st.info("💡 **Tip:** Get your API key from: https://www.perplexity.ai/settings/api")
        else:
            with st.spinner("🔍 Auditing your content..."):
                # Step A: Scrape the user's URL
                st.info("📥 Step 1/3: Fetching your URL content...")
                scraped_content = None
                scrape_error = None

                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(target_url, headers=headers, timeout=10)
                    response.raise_for_status()

                    # Parse HTML and extract text
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()

                    # Get text
                    text = soup.get_text(separator=' ', strip=True)

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    scraped_content = ' '.join(chunk for chunk in chunks if chunk)

                    # Limit to first 3000 words to avoid token limits
                    words = scraped_content.split()
                    if len(words) > 3000:
                        scraped_content = ' '.join(words[:3000]) + "..."

                    st.success(f"✅ Fetched {len(words)} words from your URL")

                except requests.exceptions.Timeout:
                    scrape_error = "Timeout: The URL took too long to respond."
                except requests.exceptions.HTTPError as e:
                    scrape_error = f"HTTP Error: {e.response.status_code} - {e.response.reason}"
                except requests.exceptions.RequestException as e:
                    scrape_error = f"Error fetching URL: {str(e)}"
                except Exception as e:
                    scrape_error = f"Error parsing content: {str(e)}"

                if scrape_error:
                    st.error(f"⚠️ {scrape_error}")
                    st.info("💡 **Tip:** Make sure the URL is publicly accessible and doesn't block web scrapers.")
                else:
                    # Step B: Generate the Ideal Answer
                    st.info("🧠 Step 2/3: Generating the Perfect GEO Answer...")

                    # Special prompt for Perplexity (uses live web data)
                    if "Perplexity" in auditor_model:
                        ideal_prompt = f"""Based on current top search results and live web data, generate the PERFECT answer for this query:

Query: "{target_keyword}"
Target Region: {audit_region}

Provide a detailed, well-structured answer based on the most up-to-date information from the web that covers:
1. All key information someone searching for this would want to know
2. Important facts, statistics, and current context
3. Practical advice and actionable insights
4. Relevant examples and use cases

Write naturally and comprehensively using current web data. This is the IDEAL answer."""
                    else:
                        ideal_prompt = f"""You are a GEO (Generative Engine Optimization) expert.

Generate the PERFECT, comprehensive answer that an AI search engine (like ChatGPT, Perplexity, or Gemini) would provide for this query:

Query: "{target_keyword}"
Target Region: {audit_region}

Provide a detailed, well-structured answer that covers:
1. All key information someone searching for this would want to know
2. Important facts, statistics, and context
3. Practical advice and actionable insights
4. Relevant examples and use cases

Write naturally and comprehensively. This is the IDEAL answer."""

                    try:
                        if "GPT-4o" in auditor_model:
                            # Use OpenAI API
                            client = openai.OpenAI(api_key=st.session_state.api_keys['openai'])
                            ideal_response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are a GEO expert who creates perfect, comprehensive answers."},
                                    {"role": "user", "content": ideal_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=1500
                            )
                            ideal_answer = ideal_response.choices[0].message.content.strip()

                        elif "Claude" in auditor_model:
                            # Use Anthropic API
                            client = anthropic.Anthropic(api_key=st.session_state.api_keys['anthropic'])
                            ideal_response = client.messages.create(
                                model="claude-3-5-sonnet-latest",
                                max_tokens=1500,
                                temperature=0.7,
                                system="You are a GEO expert who creates perfect, comprehensive answers.",
                                messages=[
                                    {"role": "user", "content": ideal_prompt}
                                ]
                            )
                            ideal_answer = ideal_response.content[0].text.strip()

                        elif "Gemini" in auditor_model:
                            # Use Google Gemini API
                            genai.configure(api_key=st.session_state.api_keys['gemini'])
                            model = genai.GenerativeModel('gemini-1.5-pro')
                            ideal_response = model.generate_content(ideal_prompt)
                            ideal_answer = ideal_response.text.strip()

                        elif "Perplexity" in auditor_model:
                            # Use Perplexity API (special: uses live web data)
                            client = openai.OpenAI(
                                api_key=st.session_state.api_keys['perplexity'],
                                base_url="https://api.perplexity.ai"
                            )
                            ideal_response = client.chat.completions.create(
                                model="sonar-pro",
                                messages=[
                                    {"role": "user", "content": ideal_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=1500
                            )
                            ideal_answer = ideal_response.choices[0].message.content.strip()

                        st.success("✅ Perfect Answer generated")

                    except Exception as e:
                        st.error(f"⚠️ Error generating ideal answer: {str(e)}")
                        ideal_answer = None

                    if ideal_answer:
                        # Step C: Compare and Generate Gap Analysis
                        st.info("📊 Step 3/3: Analyzing content gaps...")

                        comparison_prompt = f"""You are a GEO Expert. Compare the 'User Content' vs the 'Ideal AI Answer' for this query: "{target_keyword}".

USER CONTENT (from {target_url}):
{scraped_content}

---

IDEAL AI ANSWER:
{ideal_answer}

---

Analyze exactly what is MISSING in the User Content to rank for this query in AI search engines.

Return ONLY a valid JSON object (no markdown, no code blocks) with exactly these 3 fields:
{{
  "Score": <number from 0-100 representing content completeness>,
  "Missing_Topics": ["topic 1", "topic 2", "topic 3", ...],
  "Actionable_Plan": ["action 1", "action 2", "action 3"]
}}

Focus on concrete gaps. The Score should reflect how well the user content matches the ideal answer."""

                        try:
                            if "GPT-4o" in auditor_model:
                                # Use OpenAI API
                                client = openai.OpenAI(api_key=st.session_state.api_keys['openai'])
                                comparison_response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system", "content": "You are a GEO expert that returns valid JSON analysis."},
                                        {"role": "user", "content": comparison_prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=1000
                                )
                                analysis_text = comparison_response.choices[0].message.content.strip()

                            elif "Claude" in auditor_model:
                                # Use Anthropic API
                                client = anthropic.Anthropic(api_key=st.session_state.api_keys['anthropic'])
                                comparison_response = client.messages.create(
                                    model="claude-3-5-sonnet-latest",
                                    max_tokens=1000,
                                    temperature=0.3,
                                    system="You are a GEO expert that returns valid JSON analysis.",
                                    messages=[
                                        {"role": "user", "content": comparison_prompt}
                                    ]
                                )
                                analysis_text = comparison_response.content[0].text.strip()

                            elif "Gemini" in auditor_model:
                                # Use Google Gemini API
                                genai.configure(api_key=st.session_state.api_keys['gemini'])
                                model = genai.GenerativeModel('gemini-1.5-pro')
                                comparison_response = model.generate_content(comparison_prompt)
                                analysis_text = comparison_response.text.strip()

                            elif "Perplexity" in auditor_model:
                                # Use Perplexity API
                                client = openai.OpenAI(
                                    api_key=st.session_state.api_keys['perplexity'],
                                    base_url="https://api.perplexity.ai"
                                )
                                comparison_response = client.chat.completions.create(
                                    model="sonar-pro",
                                    messages=[
                                        {"role": "system", "content": "You are a GEO expert that returns valid JSON analysis."},
                                        {"role": "user", "content": comparison_prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=1000
                                )
                                analysis_text = comparison_response.choices[0].message.content.strip()

                            # Use robust JSON parser to handle different formats
                            analysis = clean_and_parse_json(analysis_text)

                            # Validate the structure
                            score = analysis.get('Score', 0)
                            missing_topics = analysis.get('Missing_Topics', [])
                            actionable_plan = analysis.get('Actionable_Plan', [])

                            st.success("✅ Analysis complete!")

                            # ============================================================================
                            # DISPLAY RESULTS
                            # ============================================================================
                            st.markdown("---")
                            st.markdown("### 📊 Content Audit Results")

                            # Big Score Display
                            score_col1, score_col2, score_col3 = st.columns([1, 2, 1])

                            with score_col2:
                                # Determine color based on score
                                if score >= 80:
                                    score_color = "#10b981"  # Green
                                    score_emoji = "🎉"
                                    score_label = "Excellent"
                                elif score >= 60:
                                    score_color = "#fbbf24"  # Yellow
                                    score_emoji = "👍"
                                    score_label = "Good"
                                elif score >= 40:
                                    score_color = "#f59e0b"  # Orange
                                    score_emoji = "⚠️"
                                    score_label = "Needs Work"
                                else:
                                    score_color = "#ef4444"  # Red
                                    score_emoji = "🚨"
                                    score_label = "Critical"

                                st.markdown(f"""
                                <div style="background: {score_color}; padding: 30px; border-radius: 15px; text-align: center; color: white;">
                                    <div style="font-size: 1.2rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">
                                        Content Completeness Score
                                    </div>
                                    <div style="font-size: 4rem; font-weight: 700; margin: 20px 0;">
                                        {score_emoji} {score}/100
                                    </div>
                                    <div style="font-size: 1.5rem; font-weight: 600; opacity: 0.9;">
                                        {score_label}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("---")

                            # Two Column Comparison
                            st.markdown("### 📋 Content Comparison")

                            comp_col1, comp_col2 = st.columns(2)

                            with comp_col1:
                                st.markdown("#### ✅ What You Have")
                                st.markdown(f"""
                                <div style="background: #d1fae5; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981; color: #065f46;">
                                    <strong>Your Content ({len(scraped_content.split())} words)</strong><br><br>
                                    {scraped_content[:500]}...
                                    <br><br>
                                    <em>This is what your page currently covers.</em>
                                </div>
                                """, unsafe_allow_html=True)

                            with comp_col2:
                                st.markdown("#### ❌ What You're Missing")

                                if missing_topics:
                                    missing_html = "<ul style='margin: 0; padding-left: 20px;'>"
                                    for topic in missing_topics:
                                        missing_html += f"<li><strong>{topic}</strong></li>"
                                    missing_html += "</ul>"

                                    st.markdown(f"""
                                    <div style="background: #fee2e2; padding: 20px; border-radius: 10px; border-left: 5px solid #ef4444; color: #991b1b;">
                                        <strong>Missing Topics & Entities:</strong><br><br>
                                        {missing_html}
                                        <br>
                                        <em>Add these topics to improve your GEO ranking.</em>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div style="background: #d1fae5; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981; color: #065f46;">
                                        <strong>Great news!</strong> Your content appears comprehensive for this query.
                                    </div>
                                    """, unsafe_allow_html=True)

                            st.markdown("---")

                            # Actionable Plan
                            if actionable_plan:
                                st.markdown("### 🎯 Actionable Optimization Plan")
                                plan_text = ""
                                for i, action in enumerate(actionable_plan, 1):
                                    plan_text += f"**{i}.** {action}\n\n"

                                st.info(plan_text)

                            # Show Ideal Answer in Expander
                            with st.expander("📖 View the Perfect GEO Answer (Reference)", expanded=False):
                                st.markdown(ideal_answer)

                            st.markdown("---")

                            # Download report
                            report_data = {
                                'Keyword': target_keyword,
                                'URL': target_url,
                                'Region': audit_region,
                                'Score': score,
                                'Missing_Topics': ', '.join(missing_topics) if missing_topics else 'None',
                                'Actionable_Plan': ' | '.join(actionable_plan) if actionable_plan else 'None',
                                'Audit_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                            df_report = pd.DataFrame([report_data])
                            csv_report = df_report.to_csv(index=False)

                            col_btn1, col_btn2 = st.columns(2)

                            with col_btn1:
                                st.download_button(
                                    label="📥 Download Audit Report as CSV",
                                    data=csv_report,
                                    file_name=f"content_audit_{target_keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                            with col_btn2:
                                if st.button("💾 Save to History", key="save_to_history"):
                                    if not st.session_state.github_token:
                                        st.warning("⚠️ Please enter your GitHub Personal Token in the sidebar.")
                                    elif not st.session_state.github_repo:
                                        st.warning("⚠️ Please enter your GitHub Repo Name in the sidebar.")
                                    else:
                                        # Prepare data row for GitHub
                                        data_row = {
                                            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'brand': st.session_state.brand_name if st.session_state.brand_name else 'N/A',
                                            'keyword': target_keyword,
                                            'model': auditor_model,
                                            'score': score,
                                            'missing_topics': ', '.join(missing_topics) if missing_topics else 'None'
                                        }

                                        with st.spinner("💾 Saving to GitHub..."):
                                            success, message = save_to_github(
                                                data_row,
                                                st.session_state.github_repo,
                                                st.session_state.github_token
                                            )

                                            if success:
                                                st.success(f"✅ {message}")
                                            else:
                                                st.error(f"❌ {message}")

                        except json.JSONDecodeError as e:
                            st.error(f"⚠️ Error parsing analysis JSON: {str(e)}")
                            st.info(f"Raw response: {analysis_text}")
                        except Exception as e:
                            st.error(f"⚠️ Error analyzing content: {str(e)}")

# ============================================================================
# TAB 5: HISTORY
# ============================================================================
with tab5:
    st.header("📈 History")
    st.markdown("**View your audit history and track content scores over time**")
    st.markdown("---")

    # Info box
    st.info("""
    ### 📊 How History Works
    1. **Save Audits:** After running a content audit in Tab 4, click "💾 Save to History" to store the results in GitHub.
    2. **Refresh Data:** Click the button below to load your audit history from GitHub.
    3. **Visualize Trends:** See how your content scores change over time across different keywords and models.
    """)

    st.markdown("---")

    # Refresh button
    if st.button("🔄 Refresh Data", key="refresh_history"):
        if not st.session_state.github_token:
            st.warning("⚠️ Please enter your GitHub Personal Token in the sidebar.")
        elif not st.session_state.github_repo:
            st.warning("⚠️ Please enter your GitHub Repo Name in the sidebar.")
        else:
            with st.spinner("📥 Loading history from GitHub..."):
                success, result = load_from_github(
                    st.session_state.github_repo,
                    st.session_state.github_token
                )

                if success:
                    df_history = result

                    if df_history.empty:
                        st.warning("📭 No audit history found. Run some audits in Tab 4 and save them!")
                    else:
                        st.success(f"✅ Loaded {len(df_history)} audit records")

                        # Display summary metrics
                        st.markdown("### 📊 Summary Metrics")

                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            avg_score = df_history['score'].mean()
                            st.metric("Average Score", f"{avg_score:.1f}/100")

                        with metric_col2:
                            total_audits = len(df_history)
                            st.metric("Total Audits", total_audits)

                        with metric_col3:
                            unique_keywords = df_history['keyword'].nunique()
                            st.metric("Unique Keywords", unique_keywords)

                        with metric_col4:
                            if 'date' in df_history.columns:
                                latest_date = df_history['date'].max()
                                st.metric("Latest Audit", latest_date.strftime('%Y-%m-%d'))
                            else:
                                st.metric("Latest Audit", "N/A")

                        st.markdown("---")

                        # Line chart: Date vs Score
                        st.markdown("### 📈 Score Trend Over Time")

                        if 'date' in df_history.columns and len(df_history) > 0:
                            # Create line chart
                            fig = px.line(
                                df_history,
                                x='date',
                                y='score',
                                color='keyword',
                                markers=True,
                                title='Content Scores Over Time',
                                labels={'date': 'Date', 'score': 'Score', 'keyword': 'Keyword'},
                                hover_data=['model', 'brand']
                            )

                            fig.update_layout(
                                hovermode='x unified',
                                xaxis_title='Date',
                                yaxis_title='Score (0-100)',
                                legend_title='Keywords',
                                height=500,
                                template='plotly_white'
                            )

                            fig.update_yaxis(range=[0, 100])

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Not enough data to display trend chart")

                        st.markdown("---")

                        # Raw dataframe
                        st.markdown("### 📋 Raw Data")

                        # Display dataframe with formatting
                        st.dataframe(
                            df_history,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Download full history
                        csv_history = df_history.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full History as CSV",
                            data=csv_history,
                            file_name=f"audit_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                else:
                    st.error(f"❌ {result}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>🌍 <strong>GEO/AIO Brand Monitor v2.0 - Ultimate Edition</strong> | Advanced LLM Brand Monitoring Platform</p>
        <p style='font-size: 0.9rem;'>Monitor your brand's visibility, sentiment, and competitor positioning across AI models with actionable GEO insights</p>
    </div>
    """,
    unsafe_allow_html=True
)
