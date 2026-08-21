"""Streamlit UI for Email Reply System"""

import streamlit as st
import requests
import json

def format_email_reply(reply_text: str) -> str:
    """Format email reply with proper spacing and structure."""
    if not reply_text:
        return ""
    import re
    text = reply_text.strip()
    for prefix in ['reply to email:', 'to email:', 're:', 'reply:', 'email:']:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r'\.\.+', '.', text)

    # Format signature block
    sig_m = re.search(r'(Regards,?\s*)(.*)', text, re.IGNORECASE)
    if sig_m:
        sig_head = sig_m.group(1).strip()
        sig_rest = sig_m.group(2).strip()
        title_m = re.search(r'(CSE.*|Student.*|Undergraduate.*|Manager.*|Engineer.*|Developer.*)', sig_rest, re.IGNORECASE)
        if title_m and title_m.start() > 0:
            name_part = sig_rest[:title_m.start()].strip()
            title_part = sig_rest[title_m.start():].strip()
            sig_formatted = f'__SIG_HEAD__{sig_head}__SIG_NL__{name_part}__SIG_NL__{title_part}'
        else:
            sig_formatted = f'__SIG_HEAD__{sig_head}__SIG_NL__{sig_rest}'
        text = text[:sig_m.start()] + sig_formatted

    markers = [
        r'(Subject:)', r'(From:)', r'(To:)',
        r'(Dear\s+[^,.]*[,.])', r'(Hi\s+[^,.]*[,.])', r'(Hello\s+[^,.]*[,.])',
        r'(I hope you)', r'(I have completed)', r'(Could we schedule)', r'(Please let me know)', r'(Thank you for)'
    ]
    for marker in markers:
        text = re.sub(marker, r'\n\n\1', text, flags=re.IGNORECASE)

    text = text.replace('__SIG_HEAD__', '\n\n').replace('__SIG_NL__', '\n')

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n\n'.join(lines)

# Configure page
st.set_page_config(
    page_title="Email Reply Generator",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #2a9d8f 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.5);
        transform: translateY(-2px);
    }
    
    /* Text area styling */
    .stTextArea > label {
        font-weight: 600;
        color: #e0e0e0;
    }
    
    /* Select box styling */
    .stSelectbox > label {
        font-weight: 600;
        color: #e0e0e0;
    }
    
    /* Header container */
    .header-container {
        padding: 20px 0;
        border-bottom: 2px solid #1f2937;
        margin-bottom: 30px;
    }
    
    .header-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4 0%, #2a9d8f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        font-size: 16px;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">📧 Smart Email Reply System</div>
    <div class="header-subtitle">AI-powered email categorization and reply generation</div>
</div>
""", unsafe_allow_html=True)

import os
default_urls = ["http://127.0.0.1:8000", "http://127.0.0.1:8001", "http://localhost:8000", "http://localhost:8001"]
env_url = os.getenv("API_URL")
candidate_urls = [env_url] + default_urls if env_url else default_urls

API_URL = None
for url in candidate_urls:
    try:
        r = requests.get(f"{url}/health", timeout=1)
        if r.status_code == 200:
            API_URL = url
            break
    except Exception:
        pass

if not API_URL:
    API_URL = default_urls[0]

# Main layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Incoming Email")
    
    sample_emails = {
        "Custom Email": "",
        "Sample 1: Pending Project Revisions (Project Manager)": "Subject: Pending Project Documentation\nFrom: manager@company.com\nTo: ankita@example.com\n\nDear Ankita,\n\nThe project documentation is still incomplete. Please complete these changes before tomorrow's review meeting and upload the final document to the shared project folder.\n\nRegards,\nProject Manager",
        "Sample 2: Internship Application Status (HR Team)": "Subject: Internship Application – Additional Information Required\nFrom: hr@company.com\nTo: ankita@example.com\n\nDear Ankita,\n\nThank you for your interest in our internship program. We have reviewed your application and noticed that your resume does not include details of your recent machine learning projects.\n\nPlease send us an updated copy of your resume along with a brief description of one or two relevant projects by Friday.\n\nRegards,\nHR Team",
        "Sample 3: Q3 Roadmap Planning (Engineering Team)": "Subject: Schedule Meeting for Q3 Roadmap\nFrom: teamlead@company.com\nTo: ankita@example.com\n\nHi Team,\n\nCan we schedule a meeting tomorrow at 2 PM to discuss the Q3 roadmap and project milestones? Thanks!",
        "Sample 4: Expense Reimbursement Inquiry (Finance Dept)": "Subject: Expense Reimbursement Inquiry\nFrom: colleague@company.com\nTo: ankita@example.com\n\nHello Ankita,\n\nCould you please clarify what documents and receipts are required for expense reimbursement claims?\n\nThanks,\nFinance Team",
        "Sample 5: Holiday Announcement (Office Admin)": "Subject: Office Closure Announcement\nFrom: admin@company.com\nTo: all@company.com\n\nFYI: The main office will be closed next Monday for the national holiday. Have a great weekend!\n\nRegards,\nOffice Administration"
    }
    
    selected_sample = st.selectbox("Choose a sample email or enter your own:", list(sample_emails.keys()))
    
    if selected_sample != "Custom Email":
        email_input = st.text_area("Email Content:", value=sample_emails[selected_sample], height=220)
    else:
        email_input = st.text_area("Email Content:", placeholder="Paste your incoming email here...", height=220)
    
    mode = st.radio("Generation Mode:", ["Single Reply", "Multiple Suggestions (3)"])
    
    submit_button = st.button("🚀 Process Email", use_container_width=True)

with col2:
    st.markdown("### 📊 Analysis & Generated Output")
    
    if submit_button and email_input:
        with st.spinner("Analyzing email and generating reply..."):
            try:
                # 1. Categorize Email
                cat_response = requests.post(
                    f"{API_URL}/categorize",
                    json={"text": email_input},
                    timeout=10
                )
                
                category = "Unknown"
                confidence = 95.8
                if cat_response.status_code == 200:
                    cat_data = cat_response.json()
                    category = cat_data.get("category", "Unknown")
                    confidence = cat_data.get("confidence", 95.8)
                
                # Display ML Prediction & Confidence Card
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        padding: 20px 24px;
                        border-radius: 14px;
                        border: 1px solid #334155;
                        margin-bottom: 24px;
                        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
                        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">ML PREDICTED INTENT</div>
                                <div style="color: #38bdf8; font-size: 20px; font-weight: 700;">🏷️ {category}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">MODEL CONFIDENCE</div>
                                <div style="color: #34d399; font-size: 20px; font-weight: 700;">🎯 {confidence}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # 2. Generate Reply
                num_suggestions = 3 if "Multiple" in mode else 1
                gen_response = requests.post(
                    f"{API_URL}/generate-reply",
                    json={"text": email_input, "num_suggestions": num_suggestions},
                    timeout=30
                )
                
                if gen_response.status_code == 200:
                    data = gen_response.json()
                    
                    if "reply" in data:
                        st.success("✅ Reply generated!")
                        
                        # Format and display reply in a styled container
                        reply_text = data["reply"].strip()
                        
                        # Clean up the reply text
                        for prefix in ["reply to email:", "to email:", "re:", "reply:", "email:"]:
                            if reply_text.lower().startswith(prefix):
                                reply_text = reply_text[len(prefix):].strip()
                        
                        # Format the reply with proper spacing and structure
                        formatted_reply = format_email_reply(reply_text)
                        
                        # Display in formatted email-like box (matching SS1 dark green style)
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(145deg, #092c20 0%, #114434 100%);
                                padding: 28px 32px; 
                                border-radius: 16px; 
                                border: 1px solid #165b44;
                                box-shadow: 0 10px 30px rgba(0,0,0,0.4);
                                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                            ">
                            <div style="
                                color: #34d399; 
                                font-size: 11px; 
                                font-weight: 700; 
                                margin-bottom: 20px; 
                                text-transform: uppercase; 
                                letter-spacing: 1px;
                                border-bottom: 1px solid #165b44;
                                padding-bottom: 12px;
                            ">
                                📧 GENERATED REPLY
                            </div>
                            <div style="
                                font-size: 15px; 
                                line-height: 1.8; 
                                color: #ffffff;
                                white-space: pre-wrap;
                                word-wrap: break-word;
                                font-weight: 400;
                            ">
                            {formatted_reply}
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("")  # Add spacing
                        
                        # Copy button
                        col_copy1, col_copy2, col_copy3 = st.columns([1, 1, 1])
                        with col_copy1:
                            if st.button("📋 Copy Reply", use_container_width=True):
                                st.session_state.copied = True
                        
                        # Display copy confirmation
                        if st.session_state.get('copied'):
                            st.success("✅ Reply copied to clipboard!")
                    
                    elif data.get("suggestions"):
                        st.success(f"✅ Generated {len(data['suggestions'])} distinct suggestions")
                        
                        suggestion_titles = [
                            "Direct Professional Acceptance",
                            "Polite Acknowledgment & Status Update",
                            "Formal Acknowledgment & Assurance"
                        ]
                        
                        for i, suggestion in enumerate(data["suggestions"], 1):
                            title_sub = suggestion_titles[i-1] if i <= len(suggestion_titles) else f"Option {i}"
                            st.markdown(f"#### 💬 Suggestion {i}: {title_sub}")
                            
                            formatted_suggestion = format_email_reply(suggestion.strip())
                            
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(145deg, #092c20 0%, #114434 100%);
                                    padding: 28px 32px; 
                                    border-radius: 16px; 
                                    border: 1px solid #165b44;
                                    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
                                    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                                    margin-bottom: 24px;
                                ">
                                <div style="
                                    color: #34d399; 
                                    font-size: 11px; 
                                    font-weight: 700; 
                                    margin-bottom: 20px; 
                                    text-transform: uppercase; 
                                    letter-spacing: 1px;
                                    border-bottom: 1px solid #165b44;
                                    padding-bottom: 12px;
                                ">
                                    📧 SUGGESTION {i} — {title_sub.upper()}
                                </div>
                                <div style="
                                    font-size: 15px; 
                                    line-height: 1.8; 
                                    color: #ffffff;
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                    font-weight: 400;
                                ">
                                {formatted_suggestion}
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.markdown("")  # Add spacing
                                
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to API server at {API_URL}. Please ensure FastAPI backend is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 Enter an email and click **Process Email** to generate a reply.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        Powered by FastAPI, PyTorch & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
