# app.py
import os
import time
import json
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
port = int(os.environ.get("PORT", 8501))

# ------------------------------
# Config / Load key
# ------------------------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(
    page_title="AI Resume Matcher Pro", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Advanced Modern CSS (same as before)
# ------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
      --primary: #6366f1;
      --secondary: #8b5cf6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-tertiary: #334155;
      --text-primary: #f8fafc;
      --text-secondary: #cbd5e1;
      --text-muted: #64748b;
      --border: #334155;
      --shadow: rgba(0, 0, 0, 0.3);
    }
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: var(--text-primary);
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px var(--shadow);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .card {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px var(--shadow);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px var(--shadow);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: var(--primary);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .skill-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
        white-space: nowrap;
        min-width: fit-content;
    }
    
    .skill-matched {
        background: linear-gradient(45deg, var(--success), #059669);
        color: white;
    }
    
    .skill-missing {
        background: linear-gradient(45deg, var(--danger), #dc2626);
        color: white;
    }
    
    .skill-extra {
        background: linear-gradient(45deg, var(--warning), #d97706);
        color: white;
    }
    
    .skill-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px var(--shadow);
    }
    
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .no-skills {
        color: var(--text-muted);
        font-style: italic;
        padding: 1rem;
        text-align: center;
        background: var(--bg-tertiary);
        border-radius: 8px;
    }
    
    .progress-container {
        background: var(--bg-tertiary);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    .recommendation-item {
        background: var(--bg-secondary);
        border-left: 4px solid var(--primary);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-excellent { background: var(--success); color: white; }
    .status-good { background: var(--warning); color: white; }
    .status-poor { background: var(--danger); color: white; }
    
    .animation-fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎯 AI Resume Matcher Pro</h1>
        <p>Advanced AI-powered resume analysis and job matching with actionable insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Sidebar Configuration
# ------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    model_options = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    selected_model = st.selectbox("🤖 AI Model", model_options, index=0)
    
    # Analysis depth
    analysis_depth = st.select_slider(
        "📊 Analysis Depth",
        options=["Quick", "Standard", "Deep", "Comprehensive"],
        value="Standard"
    )
    
    # Temperature setting
    temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.1, 0.1)
    
    st.markdown("---")
    st.markdown("### 📈 Session Stats")
    
    # Session state for stats
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'best_match' not in st.session_state:
        st.session_state.best_match = 0
    
    st.metric("Analyses Run", st.session_state.analysis_count)
    st.metric("Best Match %", f"{st.session_state.best_match}%")

# ------------------------------
# API Key Validation
# ------------------------------
if not GROQ_KEY:
    st.error("🔑 GROQ_API_KEY not found. Please set it in your `.env` file.", icon="⚠️")
    st.stop()

# ------------------------------
# LLM Setup
# ------------------------------
llm = ChatGroq(api_key=GROQ_KEY, model=selected_model, temperature=temperature)

# ------------------------------
# Enhanced Prompts with Better Structure
# ------------------------------
# Enhanced comparison prompt with structured output
response_schemas = [
    ResponseSchema(name="skills_matched", description="Array of skills present in both JD and resume - return as comma-separated string"),
    ResponseSchema(name="skills_missing", description="Array of critical skills missing from resume - return as comma-separated string"),
    ResponseSchema(name="skills_extra", description="Array of additional skills in resume not required by JD - return as comma-separated string"),
    ResponseSchema(name="experience_match", description="How well experience aligns with requirements"),
    ResponseSchema(name="education_match", description="Education alignment assessment"),
    ResponseSchema(name="overall_match_percentage", description="Overall match percentage (0-100) as integer"),
    ResponseSchema(name="selection_probability", description="Likelihood of selection (High/Medium/Low)"),
    ResponseSchema(name="strength_areas", description="Top 3 candidate strengths as comma-separated string"),
    ResponseSchema(name="improvement_areas", description="Top 3 areas needing improvement as comma-separated string"),
    ResponseSchema(name="specific_recommendations", description="Actionable improvement steps as comma-separated string"),
    ResponseSchema(name="interview_preparation", description="Suggested interview focus areas as comma-separated string"),
    ResponseSchema(name="salary_competitiveness", description="Salary negotiation position assessment")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt_compare = PromptTemplate(
    input_variables=["job_description", "resume_text", "format_instructions", "depth"],
    template="""
You are a senior HR consultant providing comprehensive resume-job matching analysis.
Analysis depth: {depth}

IMPORTANT: For all array fields (skills_matched, skills_missing, skills_extra, strength_areas, improvement_areas, specific_recommendations, interview_preparation), return the values as comma-separated strings, NOT as arrays. 

For example:
- CORRECT: "Python, JavaScript, React, Node.js"
- WRONG: ["Python", "JavaScript", "React", "Node.js"]

Perform detailed analysis:

1. **Skill Matching**: 
   - Extract complete skill names (not individual characters)
   - Match technical skills, soft skills, tools, technologies
   - Consider synonyms (e.g., "JS" = "JavaScript", "React.js" = "React")

2. **Experience Analysis**: Compare years, relevance, industry alignment

3. **Education Assessment**: Degree requirements vs candidate qualifications

4. **Overall Scoring**: Weighted scoring based on critical vs nice-to-have requirements

Return analysis following this exact format:
{format_instructions}

Job Description:
{job_description}

Resume:
{resume_text}
"""
)

# ------------------------------
# Utility Functions
# ------------------------------
def pdf_to_text(file) -> str:
    try:
        reader = PdfReader(file)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")
        return ""

def parse_comma_separated(value):
    """Parse comma-separated string into list of clean strings"""
    if not value:
        return []
    
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    
    # Handle string input
    if isinstance(value, str):
        # Remove brackets if present
        value = re.sub(r'[\[\]"]', '', value)
        # Split by comma and clean
        items = [item.strip() for item in value.split(',') if item.strip()]
        # Filter out single characters (likely parsing errors)
        items = [item for item in items if len(item) > 1]
        return items
    
    return []

def create_match_gauge(percentage):
    """Create a modern gauge chart for match percentage"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match Score", 'font': {'size': 20, 'color': '#f8fafc'}},
        delta = {'reference': 70, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#cbd5e1'},
            'bar': {'color': "#6366f1"},
            'steps': [
                {'range': [0, 50], 'color': "#1e293b"},
                {'range': [50, 80], 'color': "#334155"},
                {'range': [80, 100], 'color': "#475569"}
            ],
            'threshold': {
                'line': {'color': "#10b981", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc', 'family': 'Inter'},
        height=300
    )
    
    return fig

def render_skill_chips(skills, chip_type="matched"):
    """Render skills as interactive chips with proper parsing"""
    # Parse skills if they're in string format
    if isinstance(skills, str):
        skills = parse_comma_separated(skills)
    elif not isinstance(skills, list):
        skills = []
    
    if not skills:
        st.markdown('<div class="no-skills">No skills identified</div>', unsafe_allow_html=True)
        return
    
    # Create chips HTML
    chips_html = '<div class="skills-container">'
    for skill in skills:
        skill_clean = str(skill).strip()
        if skill_clean and len(skill_clean) > 1:  # Avoid single characters
            chips_html += f'<span class="skill-chip skill-{chip_type}">{skill_clean}</span>'
    chips_html += '</div>'
    
    st.markdown(chips_html, unsafe_allow_html=True)

def get_status_badge(percentage):
    """Get status badge based on percentage"""
    if percentage >= 80:
        return '<span class="status-badge status-excellent">Excellent Match</span>'
    elif percentage >= 60:
        return '<span class="status-badge status-good">Good Match</span>'
    else:
        return '<span class="status-badge status-poor">Needs Improvement</span>'

# ------------------------------
# Main Interface
# ------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Job Description")
    job_description = st.text_area(
        "Paste the complete job description",
        height=350,
        placeholder="Paste the full job description here...\n\nInclude:\n• Role responsibilities\n• Required skills\n• Experience requirements\n• Education requirements\n• Company culture details",
        help="The more detailed the job description, the better the analysis will be."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Resume")
    
    resume_input_method = st.radio(
        "How would you like to provide your resume?",
        ["📁 Upload PDF", "✍️ Paste Text"],
        horizontal=True
    )
    
    resume_text = ""
    
    if resume_input_method == "📁 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF format)",
            type=["pdf"],
            help="Upload a clear, well-formatted PDF resume for best results"
        )
        
        if uploaded_file:
            with st.spinner("🔍 Extracting text from PDF..."):
                resume_text = pdf_to_text(uploaded_file)
                time.sleep(0.5)  # Visual feedback
            
            if resume_text:
                st.success(f"✅ PDF processed successfully! Extracted {len(resume_text)} characters.")
                with st.expander("📝 Review extracted text"):
                    st.text_area("Extracted content", resume_text, height=200)
            else:
                st.error("❌ Failed to extract text from PDF. Please try pasting the text manually.")
    
    else:
        resume_text = st.text_area(
            "Paste your resume text",
            height=350,
            placeholder="Paste your complete resume here...\n\nInclude:\n• Contact information\n• Professional summary\n• Work experience\n• Education\n• Skills\n• Projects/Achievements",
            help="Include all sections of your resume for comprehensive analysis"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Analysis Buttons
# ------------------------------
st.markdown("### 🚀 Analysis Controls")
button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

with button_col2:
    compare_btn = st.button(
        "⚡ Complete Analysis",
        use_container_width=True,
        type="primary",
        help="Run comprehensive matching analysis"
    )

with button_col3:
    clear_btn = st.button(
        "🗑️ Clear All",
        use_container_width=True,
        help="Clear all inputs and results"
    )

if clear_btn:
    st.rerun()

# ------------------------------
# Complete Analysis
# ------------------------------
if compare_btn:
    if not job_description or not resume_text:
        st.error("❌ Both job description and resume are required for analysis.")
    else:
        with st.spinner("🧠 Running comprehensive AI analysis..."):
            progress_bar = st.progress(0)
            
            try:
                progress_bar.progress(50)
                system_msg = SystemMessage(content="""You are a senior HR consultant. 
                CRITICAL: For all list/array fields, return comma-separated strings, NOT JSON arrays.
                Example: "Python, JavaScript, React" NOT ["Python", "JavaScript", "React"]
                Return only valid JSON following the exact schema provided.""")
                
                human_msg = HumanMessage(content=prompt_compare.format(
                    job_description=job_description,
                    resume_text=resume_text,
                    format_instructions=format_instructions,
                    depth=analysis_depth
                ))
                
                response = llm.invoke([system_msg, human_msg])
                raw_output = response.content
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                
                # Parse the structured output
                try:
                    analysis_result = output_parser.parse(raw_output)
                except Exception as parse_error:
                    st.error(f"❌ Failed to parse AI response: {parse_error}")
                    with st.expander("🔧 Debug - Raw AI Response"):
                        st.code(raw_output, language="json")
                    st.stop()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()
        
        # Update session stats
        st.session_state.analysis_count += 1
        match_percentage = int(analysis_result.get("overall_match_percentage", 0))
        if match_percentage > st.session_state.best_match:
            st.session_state.best_match = match_percentage
        
        st.success("✅ Comprehensive analysis completed!")
        
        # ------------------------------
        # Results Display
        # ------------------------------
        st.markdown("## 📊 Analysis Results")
        
        # Parse skills with better handling
        skills_matched = parse_comma_separated(analysis_result.get("skills_matched", ""))
        skills_missing = parse_comma_separated(analysis_result.get("skills_missing", ""))
        skills_extra = parse_comma_separated(analysis_result.get("skills_extra", ""))
        
        # Overall metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">✅ Matched Skills</div>
                <div class="metric-value">{len(skills_matched)}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">❌ Missing Skills</div>
                <div class="metric-value">{len(skills_missing)}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">➕ Extra Skills</div>
                <div class="metric-value">{len(skills_extra)}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">🎯 Match Score</div>
                <div class="metric-value">{match_percentage}%</div>
                {get_status_badge(match_percentage)}
            </div>
            ''', unsafe_allow_html=True)
        
        # Match score visualization
        st.plotly_chart(create_match_gauge(match_percentage), use_container_width=True)
        
        # Progress bar
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.progress(match_percentage / 100)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        analysis_col1, analysis_col2 = st.columns([1, 1])
        
        with analysis_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Skills Analysis")
            
            st.markdown("#### ✅ **Matched Skills**")
            render_skill_chips(skills_matched, "matched")
            
            st.markdown("#### ❌ **Missing Critical Skills**")
            render_skill_chips(skills_missing, "missing")
            
            st.markdown("#### ➕ **Additional Skills**")
            render_skill_chips(skills_extra, "extra")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 **Overall Assessment**")
            
            selection_prob = analysis_result.get("selection_probability", "Unknown")
            exp_match = analysis_result.get("experience_match", "Not assessed")
            edu_match = analysis_result.get("education_match", "Not assessed")
            
            st.markdown(f"""
            **Selection Probability:** `{selection_prob}`
            
            **Experience Match:** {exp_match}
            
            **Education Match:** {edu_match}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Strengths and improvements
        strength_col, improvement_col = st.columns([1, 1])
        
        with strength_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 💪 **Key Strengths**")
            strengths = parse_comma_separated(analysis_result.get("strength_areas", ""))
            for i, strength in enumerate(strengths, 1):
                st.markdown(f"**{i}.** {strength}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with improvement_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎯 **Improvement Areas**")
            improvements = parse_comma_separated(analysis_result.get("improvement_areas", ""))
            for i, improvement in enumerate(improvements, 1):
                st.markdown(f"**{i}.** {improvement}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 **Actionable Recommendations**")
        recommendations = parse_comma_separated(analysis_result.get("specific_recommendations", ""))
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'''
            <div class="recommendation-item">
                <strong>{i}.</strong> {rec}
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interview preparation
        interview_prep = parse_comma_separated(analysis_result.get("interview_preparation", ""))
        if interview_prep:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎤 **Interview Preparation**")
            for i, prep in enumerate(interview_prep, 1):
                st.markdown(f"**{i}.** {prep}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Salary insights
        salary_info = analysis_result.get("salary_competitiveness", "")
        if salary_info:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 💰 **Salary Negotiation Position**")
            st.markdown(salary_info)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export results
        st.markdown("### 📄 **Export Results**")
        export_col1, export_col2 = st.columns([1, 1])
        
        with export_col1:
            # Prepare export data
            export_data = {
                "analysis_date": datetime.now().isoformat(),
                "match_percentage": match_percentage,
                "selection_probability": selection_prob,
                "skills_matched": skills_matched,
                "skills_missing": skills_missing,
                "skills_extra": skills_extra,
                "analysis_result": analysis_result
            }
            
            st.download_button(
                label="📊 Download Analysis Report (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with export_col2:
            if st.button("🔄 **Run New Analysis**", use_container_width=True):
                st.rerun()
        
        # Debug information
        with st.expander("🔧 **Debug Information**"):
            st.code(raw_output, language="json")
            st.json({
                "parsed_skills_matched": skills_matched,
                "parsed_skills_missing": skills_missing,
                "parsed_skills_extra": skills_extra
            })

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: var(--text-muted); padding: 2rem;">
        <p>🎯 <strong>AI Resume Matcher Pro</strong> | Powered by Groq & LangChain</p>
        <p>Built with ❤️ for job seekers and recruiters</p>
    </div>
    """,
    unsafe_allow_html=True
)
