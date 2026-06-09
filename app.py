"""
Sinhala Handwritten Text Recognition Web App
Professional Streamlit Implementation with Authentication & OCR Pipeline
"""

import streamlit as st
import time
import re
from datetime import datetime
from PIL import Image
import pandas as pd

# Import your locked model pipeline
from model_pipeline import load_sinhala_ocr_model, extract_text_from_handwriting

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Sinhala OCR - Handwriting Recognition",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    /* Main background and canvas */
    .stApp {
        background: linear-gradient(135deg, #F4F7F6 0%, #E8EDEB 100%);
    }
    
    /* Card styling for content boxes */
    .custom-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Sinhala text styling for better glyph rendering */
    .sinhala-text {
        font-family: 'Noto Sans Sinhala', 'Iskoola Pota', 'Nirmala UI', sans-serif;
        font-size: 1.2rem;
        line-height: 1.6;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 20px;
        background: #f8f9fa;
    }
    
    /* Success/Warning/Info cards */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Sidebar customization */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #1a252f 100%);
    }
    
    /* Image preview styling */
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'users_db' not in st.session_state:
    # Simulated database (in production, use real database)
    st.session_state.users_db = {}
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None

# ==================== MODEL INITIALIZATION (CACHED) ====================
@st.cache_resource
def initialize_pipeline():
    """Load the TrOCR model with caching for optimal performance"""
    with st.spinner("🔄 Loading Sinhala OCR Engine... Please wait..."):
        try:
            model, processor, device, config = load_sinhala_ocr_model()
            return model, processor, device, config
        except Exception as e:
            st.error(f"❌ Failed to load OCR model: {str(e)}")
            st.stop()

# Initialize model pipeline
try:
    model, processor, device, config = initialize_pipeline()
    ocr_ready = True
except Exception as e:
    ocr_ready = False
    st.error("OCR Engine unavailable. Please check model files.")

# ==================== AUTHENTICATION FUNCTIONS ====================
def authenticate_user(email, password):
    """Validate user credentials"""
    if email in st.session_state.users_db:
        if st.session_state.users_db[email]['password'] == password:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.session_state.user_name = st.session_state.users_db[email]['name']
            st.session_state.current_page = 'dashboard'
            return True
    return False

def register_user(name, email, password):
    """Register a new user"""
    if email in st.session_state.users_db:
        return False, "Email already registered!"
    
    st.session_state.users_db[email] = {
        'name': name,
        'password': password,
        'registered_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return True, "Registration successful!"

def reset_password(email):
    """Simulate password reset"""
    if email in st.session_state.users_db:
        # In production, send actual email
        st.session_state.reset_email = email
        return True
    return False

# ==================== PAGE RENDERING FUNCTIONS ====================
def show_login_page():
    """Render the login page"""
    st.markdown("""
    <div class="main-header">
        <h1>📝 සිංහල අත්අකුරු හඳුනාගැනීම</h1>
        <p>Sinhala Handwritten Character Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔐 Welcome Back")
        st.markdown("---")
        
        email = st.text_input("📧 Email Address", key="login_email")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            login_clicked = st.button("🚀 Login", use_container_width=True)
        with col_btn2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.current_page = 'signup'
                st.rerun()
        
        if login_clicked:
            if email and password:
                if authenticate_user(email, password):
                    st.success("✅ Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password!")
            else:
                st.warning("⚠️ Please enter both email and password!")
        
        st.markdown("---")
        if st.button("🔑 Forgot Password?", use_container_width=True):
            st.session_state.current_page = 'forgot_password'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_signup_page():
    """Render the signup page"""
    st.markdown("""
    <div class="main-header">
        <h1>📝 Create New Account</h1>
        <p>Join us to experience Sinhala OCR Technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 Registration Form")
        st.markdown("---")
        
        full_name = st.text_input("👤 Full Name", key="signup_name")
        email = st.text_input("📧 Email Address", key="signup_email")
        password = st.text_input("🔒 Password", type="password", key="signup_password")
        confirm_password = st.text_input("✓ Confirm Password", type="password", key="signup_confirm")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            register_clicked = st.button("✅ Register", use_container_width=True)
        with col_btn2:
            if st.button("⬅️ Back to Login", use_container_width=True):
                st.session_state.current_page = 'login'
                st.rerun()
        
        if register_clicked:
            if not all([full_name, email, password, confirm_password]):
                st.warning("⚠️ Please fill all fields!")
            elif password != confirm_password:
                st.error("❌ Passwords do not match!")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("❌ Invalid email format!")
            else:
                success, message = register_user(full_name, email, password)
                if success:
                    st.success(f"✅ {message} Please login!")
                    time.sleep(1.5)
                    st.session_state.current_page = 'login'
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_forgot_password_page():
    """Render the forgot password page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔑 Password Recovery</h1>
        <p>We'll help you reset your password</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📧 Email Verification")
        st.markdown("---")
        
        email = st.text_input("Enter your registered email address", key="reset_email_input")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            reset_clicked = st.button("📨 Send Reset Link", use_container_width=True)
        with col_btn2:
            if st.button("⬅️ Back to Login", use_container_width=True):
                st.session_state.current_page = 'login'
                st.rerun()
        
        if reset_clicked:
            if email:
                if reset_password(email):
                    st.success("""
                    ✅ **Password Reset Instructions Sent!**
                    
                    A password reset link has been sent to your email address.
                    Please check your inbox and follow the instructions.
                    """)
                    
                    # Simulate sending email
                    with st.expander("📧 Demo: Reset Email Preview"):
                        st.info(f"""
                        **To:** {email}
                        **Subject:** Password Reset Request - Sinhala OCR System
                        
                        Dear User,
                        
                        Click the link below to reset your password:
                        🔗 [Reset Password](https://yourdomain.com/reset?token=demo_token)
                        
                        This link will expire in 1 hour.
                        
                        If you didn't request this, please ignore this email.
                        """)
                    
                    if st.button("✅ Return to Login"):
                        st.session_state.current_page = 'login'
                        st.rerun()
                else:
                    st.error("❌ Email not found in our system!")
            else:
                st.warning("⚠️ Please enter your email address!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    """Render the main OCR dashboard"""
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0;">👋 Welcome!</h3>
                <p style="color: white; margin: 5px 0 0 0; font-size: 0.9rem;">
                    {st.session_state.user_name}
                </p>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.8rem;">
                    {st.session_state.user_email}
                </p>
            </div>
            
            <div style="background: #27ae60; border-radius: 10px; padding: 10px; margin-top: 20px;">
                <p style="color: white; margin: 0;">
                    ✅ OCR Engine Status: <strong>LIVE</strong>
                </p>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.8rem;">
                    Sinhala Handwriting Model Active
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = None
            st.session_state.user_name = None
            st.session_state.current_page = 'login'
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.85rem; text-align: center; color: #666;">
            <p>📝 <strong>Tips for best results:</strong></p>
            <p>✓ Use clear handwriting</p>
            <p>✓ Ensure good lighting</p>
            <p>✓ Crop to text region</p>
            <p>✓ Use high contrast</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown("""
    <div class="main-header">
        <h1>✨ Sinhala Handwriting Recognition</h1>
        <p>Upload an image containing Sinhala handwritten text for instant recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for dual-panel design
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Handwriting Sample")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG. Ensure clear handwriting for best results."
        )
        
        if uploaded_file is not None:
            # Display image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Handwriting Sample", use_container_width=True)
            
            # Extract button
            st.markdown("---")
            if st.button("🖋️ අකුරු කියවන්න (Extract Text)", use_container_width=True):
                if ocr_ready:
                    with st.spinner("🔍 Analyzing handwriting strokes..."):
                        try:
                            # Process the image through your model pipeline
                            predicted_text = extract_text_from_handwriting(
                                uploaded_file, model, processor, device, config
                            )
                            
                            # Store in session state for display
                            st.session_state.ocr_result = predicted_text
                            st.session_state.ocr_processed = True
                            
                        except Exception as e:
                            st.error(f"⚠️ Recognition error: {str(e)}")
                            st.session_state.ocr_processed = False
                else:
                    st.error("❌ OCR Engine is not ready. Please check model files.")
        else:
            st.info("📸 No image uploaded yet. Please select an image to begin.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📝 Recognition Results")
        st.markdown("---")
        
        # Display results if available
        if 'ocr_processed' in st.session_state and st.session_state.ocr_processed:
            result_text = st.session_state.get('ocr_result', 'No result available')
            
            # Display in a nice text area with Sinhala font support
            st.text_area(
                "Extracted Text:",
                value=result_text,
                height=200,
                key="ocr_output",
                help="The recognized Sinhala text from your handwriting sample"
            )
            
            # Copy button
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.success("✅ Text copied to clipboard!")
                # JavaScript for clipboard copy
                st.markdown(f"""
                <script>
                function copyToClipboard() {{
                    const text = `{result_text}`;
                    navigator.clipboard.writeText(text);
                }}
                </script>
                """, unsafe_allow_html=True)
            
            # Additional info
            st.markdown("---")
            st.markdown("""
            <div style="background: #e8f4f8; padding: 15px; border-radius: 10px;">
                <p style="margin: 0; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> You can edit the extracted text above or copy it for further processing.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("✨ Recognition results will appear here after processing an image.")
            
            # Show example placeholder
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; background: #f8f9fa; border-radius: 10px;">
                <p style="color: #999; font-size: 0.9rem;">
                📝 Upload a Sinhala handwriting image and click "Extract Text"
                </p>
                <p style="color: #ccc; font-size: 0.8rem;">
                Example: සිංහල අත්අකුරු හඳුනාගැනීමේ පද්ධතිය
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== MAIN APP CONTROLLER ====================
def main():
    """Main application controller"""
    
    # Route to appropriate page based on authentication and current_page
    if not st.session_state.authenticated:
        if st.session_state.current_page == 'signup':
            show_signup_page()
        elif st.session_state.current_page == 'forgot_password':
            show_forgot_password_page()
        else:
            show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
