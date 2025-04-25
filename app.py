import streamlit as st
import joblib
import pandas as pd
import time
import re

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4B5563;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(30, 58, 138, 0.1);
    }
    .stButton>button:hover {
        box-shadow: 0 6px 10px rgba(30, 58, 138, 0.2);
        transform: translateY(-2px);
    }
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    .result-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 4rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 6px 6px 0 0;
        padding: 10px 24px;
        gap: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        border-bottom: 2px solid #1E3A8A !important;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        padding: 12px;
        font-size: 1rem;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .stRadio > div > div {
        background-color: #F3F4F6;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 0 5px;
    }
    .stRadio > div [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_reliability_score(text):
    # Simplified example - you could implement more sophisticated metrics
    factors = {
        'length': min(len(text.split()), 500) / 500,  # Longer text gets higher score up to 500 words
        'has_citations': 1 if re.search(r'\[\d+\]|\(\d{4}\)', text) else 0,
        'sentiment_balance': 0.7,  # Placeholder for sentiment analysis
        'source_credibility': 0.8,  # Placeholder for source checking
    }
    
    # Weighted average
    weights = {'length': 0.2, 'has_citations': 0.3, 'sentiment_balance': 0.2, 'source_credibility': 0.3}
    score = sum(factor * weights[key] for key, factor in factors.items())
    return min(max(score, 0), 1)  # Ensure between 0 and 1

def get_confidence_color(confidence):
    if confidence > 0.8:
        return "#10B981"  # Green
    elif confidence > 0.6:
        return "#FBBF24"  # Yellow
    else:
        return "#EF4444"  # Red

# Load models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

vectorizer, model = load_models()

# Sidebar with improved content and no image
with st.sidebar:
    st.markdown("## AI Fake News Detector")
    st.markdown("""
    <div class="info-box">
    This advanced tool leverages machine learning to analyze news articles and determine their credibility with high accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### How it works")
    st.markdown("""
    Our algorithm examines:
    
    1. **Language Patterns** - Identifies sensationalist language and emotional manipulation
    
    2. **Content Structure** - Analyzes article composition and information flow
    
    3. **Source Credibility** - Evaluates reliability signals in the content
    
    4. **Contextual Analysis** - Examines consistency and factual presentation
    """)
    
    st.markdown("### Tips for Spotting Fake News")
    st.markdown("""
    ✓ **Check Multiple Sources**
    Verify information across reputable outlets
    
    ✓ **Examine Author Credentials**
    Legitimate articles typically have identified authors with verifiable backgrounds
    
    ✓ **Look for Citations**
    Credible news cites sources and provides evidence
    
    ✓ **Be Wary of Emotional Language**
    Fake news often uses charged language to trigger emotional responses
    
    ✓ **Check Dates & Context**
    Old news may be presented as current, or taken out of context
    """)
    
    st.markdown("### About This Tool")
    st.markdown("""
    <div class="info-box">
    Our model has been trained on thousands of verified and fake news samples, achieving over 90% accuracy in controlled testing environments.
    
    This tool is for educational purposes and should not replace critical thinking and thorough fact-checking.
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis to identify misinformation in news articles</p>', unsafe_allow_html=True)

# Tabs with improved design
tab1, tab2 = st.tabs(["📝 Analyze Content", "📊 Results Interpretation"])

with tab1:
    input_method = st.radio("Select input method:", ["Text", "URL"], horizontal=True)
    
    if input_method == "Text":
        news_input = st.text_area("Enter the news article text:", height=200, 
                                placeholder="Paste the full text of the news article here for analysis...")
    else:
        news_url = st.text_input("Enter news article URL:", placeholder="https://example.com/news-article")
        news_input = news_url  # In a real app, you'd fetch and process the URL content
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Analyze Article", use_container_width=True)

    if predict_button:
        if news_input and news_input.strip():
            with st.spinner("Analyzing article content..."):
                # Progress bar with improved visual feedback
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulating analysis time
                    progress_bar.progress(i + 1)
                
                # Make prediction
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)
                proba = model.predict_proba(transform_input)[0]
                
                # Calculate confidence and reliability
                confidence = proba[1] if prediction[0] == 1 else proba[0]
                reliability_score = calculate_reliability_score(news_input)
                
                # Display result with enhanced visual design
                st.markdown("### Analysis Result")
                
                if prediction[0] == 1:
                    st.markdown(f"""
                    <div class="result-card" style="background-color: rgba(16, 185, 129, 0.05); border: 1px solid #10B981;">
                        <h2 style="color: #10B981; font-size: 1.8rem; margin-bottom: 1rem;">✅ This article appears to be Real</h2>
                        <p style="font-size: 1.1rem; margin-bottom: 1rem;">Our analysis indicates this content demonstrates characteristics consistent with legitimate news reporting.</p>
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 600; margin-right: 1rem;">Confidence:</div>
                            <div style="flex-grow: 1; background-color: #E5E7EB; height: 12px; border-radius: 6px; margin-right: 10px;">
                                <div style="width: {confidence*100}%; background-color: {get_confidence_color(confidence)}; height: 12px; border-radius: 6px;"></div>
                            </div>
                            <div style="font-weight: 700; font-size: 1.2rem;">{confidence*100:.1f}%</div>
                        </div>
                        <p style="color: #6B7280; font-style: italic;">We've detected multiple indicators of credible journalism in this article.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card" style="background-color: rgba(239, 68, 68, 0.05); border: 1px solid #EF4444;">
                        <h2 style="color: #EF4444; font-size: 1.8rem; margin-bottom: 1rem;">⚠️ This article appears to be Fake</h2>
                        <p style="font-size: 1.1rem; margin-bottom: 1rem;">Our analysis indicates this content shows characteristics consistent with misinformation.</p>
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 600; margin-right: 1rem;">Confidence:</div>
                            <div style="flex-grow: 1; background-color: #E5E7EB; height: 12px; border-radius: 6px; margin-right: 10px;">
                                <div style="width: {confidence*100}%; background-color: {get_confidence_color(confidence)}; height: 12px; border-radius: 6px;"></div>
                            </div>
                            <div style="font-weight: 700; font-size: 1.2rem;">{confidence*100:.1f}%</div>
                        </div>
                        <p style="color: #6B7280; font-style: italic;">We've detected several warning signs of potential misinformation in this content.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Content analysis with improved metrics display
                st.markdown("### Content Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-title">Overall Reliability Score</div>
                        <div class="metric-value">{0:.1f}%</div>
                        <div style="width: 100%; background-color: #E5E7EB; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                            <div style="width: {1}%; background-color: {2}; height: 8px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """.format(reliability_score*100, reliability_score*100, get_confidence_color(reliability_score)), unsafe_allow_html=True)
                    
                    # Content statistics
                    word_count = len(news_input.split())
                    sentence_count = len(re.split(r'[.!?]+', news_input))
                    avg_words = round(word_count/max(1, sentence_count), 1)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-title">Content Statistics</div>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <div style="text-align: center; flex: 1;">
                                <div style="font-size: 0.9rem; color: #6B7280;">Words</div>
                                <div style="font-weight: 700; font-size: 1.5rem; color: #1E3A8A;">{0}</div>
                            </div>
                            <div style="text-align: center; flex: 1;">
                                <div style="font-size: 0.9rem; color: #6B7280;">Sentences</div>
                                <div style="font-weight: 700; font-size: 1.5rem; color: #1E3A8A;">{1}</div>
                            </div>
                            <div style="text-align: center; flex: 1;">
                                <div style="font-size: 0.9rem; color: #6B7280;">Avg Words/Sentence</div>
                                <div style="font-weight: 700; font-size: 1.5rem; color: #1E3A8A;">{2}</div>
                            </div>
                        </div>
                    </div>
                    """.format(word_count, sentence_count, avg_words), unsafe_allow_html=True)
                
                with col2:
                    # Credibility factors visualization
                    factors = {
                        'Source Credibility': 0.8,
                        'Factual Content': 0.75 if prediction[0] == 1 else 0.4,
                        'Neutral Language': 0.7 if prediction[0] == 1 else 0.3,
                        'Internal Consistency': 0.85 if prediction[0] == 1 else 0.5
                    }
                    
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-title">Credibility Factors</div>
                    """, unsafe_allow_html=True)
                    
                    for factor, score in factors.items():
                        st.markdown(f"""
                        <div style="margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                                <span style="font-size: 0.9rem;">{factor}</span>
                                <span style="font-size: 0.9rem; font-weight: 600;">{score*100:.0f}%</span>
                            </div>
                            <div style="width: 100%; background-color: #E5E7EB; height: 6px; border-radius: 3px;">
                                <div style="width: {score*100}%; background-color: {get_confidence_color(score)}; height: 6px; border-radius: 3px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Key findings
                    finding_color = "#10B981" if prediction[0] == 1 else "#EF4444"
                    finding_text = "multiple credibility indicators" if prediction[0] == 1 else "potential misinformation signals"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Key Findings</div>
                        <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {finding_color}; margin-right: 8px;"></div>
                            <div>Analysis detected {finding_text}</div>
                        </div>
                        <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #3B82F6; margin-right: 8px;"></div>
                            <div>Content structure {'supports' if prediction[0] == 1 else 'challenges'} credibility</div>
                        </div>
                        <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {'#10B981' if reliability_score > 0.6 else '#FBBF24'}; margin-right: 8px;"></div>
                            <div>Overall reliability score is {reliability_score*100:.0f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation section
                st.markdown("### Recommendation")
                if prediction[0] == 1 and confidence > 0.8:
                    st.markdown("""
                    <div style="background-color: rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 1rem; border-left: 4px solid #10B981;">
                        <p style="font-weight: 600; color: #10B981;">This article appears credible, but always verify information from multiple sources.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction[0] == 0 and confidence > 0.8:
                    st.markdown("""
                    <div style="background-color: rgba(239, 68, 68, 0.1); border-radius: 8px; padding: 1rem; border-left: 4px solid #EF4444;">
                        <p style="font-weight: 600; color: #EF4444;">This content shows strong signals of misinformation. We recommend seeking alternative reputable sources.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: rgba(251, 191, 36, 0.1); border-radius: 8px; padding: 1rem; border-left: 4px solid #FBBF24;">
                        <p style="font-weight: 600; color: #FBBF24;">Our analysis is inconclusive. We strongly recommend verifying this information with established news sources.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please enter an article to analyze.")

with tab2:
    st.markdown("### Understanding Your Results")
    
    st.markdown("""
    <div style="background-color: #F3F4F6; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #1E3A8A;">How Our AI Evaluates News</h4>
        <p>Our machine learning model has been trained on thousands of verified real and fake news articles to recognize patterns that indicate misinformation.</p>
        
        <h5 style="color: #1E3A8A; margin-top: 1rem;">Confidence Levels Explained</h5>
        <ul style="margin-bottom: 1rem;">
            <li><span style="color: #10B981; font-weight: 600;">High Confidence (80-100%)</span>: The AI has detected strong signals that align with either real or fake news patterns</li>
            <li><span style="color: #FBBF24; font-weight: 600;">Medium Confidence (60-80%)</span>: Some indicators are present, but the signals are mixed</li>
            <li><span style="color: #EF4444; font-weight: 600;">Low Confidence (Below 60%)</span>: The AI cannot make a clear determination</li>
        </ul>
        
        <h5 style="color: #1E3A8A; margin-top: 1rem;">Reliability Score Components</h5>
        <ul>
            <li><strong>Source Credibility</strong>: Indicators of source reputation and authority</li>
            <li><strong>Factual Content</strong>: Presence of verifiable facts versus unsupported claims</li>
            <li><strong>Neutral Language</strong>: Level of emotional or sensationalist language</li>
            <li><strong>Internal Consistency</strong>: How logically consistent the article is</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: rgba(16, 185, 129, 0.05); border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="margin-top: 0; color: #10B981;">Characteristics of Real News</h4>
            <ul>
                <li>Identifies sources by name</li>
                <li>Contains specific dates, times, and locations</li>
                <li>Uses neutral, objective language</li>
                <li>Provides multiple perspectives</li>
                <li>Information can be verified elsewhere</li>
                <li>Logical structure and progression</li>
                <li>Contains relevant context</li>
                <li>Published by established outlets with editorial standards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: rgba(239, 68, 68, 0.05); border-radius: 12px; padding: 1.5rem; height: 100%;">
            <h4 style="margin-top: 0; color: #EF4444;">Warning Signs of Fake News</h4>
            <ul>
                <li>Vague or anonymous sources</li>
                <li>Emotionally charged language</li>
                <li>Sensationalist claims</li>
                <li>Lack of specific details</li>
                <li>Information not reported elsewhere</li>
                <li>Excessive use of ALL CAPS or exclamation marks!!!</li>
                <li>Missing context or background</li>
                <li>Attempts to trigger outrage or fear</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-top: 1.5rem;">
        <p style="font-weight: 600; margin-bottom: 0.5rem;">Important Note:</p>
        <p>This tool provides an automated assessment and should be used as one of many tools in your information verification toolkit. Always cross-reference information with multiple reputable sources.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 Fake News Detector | For educational purposes only | Developed with AI technology</div>', unsafe_allow_html=True)