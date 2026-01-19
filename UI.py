import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
import base64
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert image to base64
def get_image_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .stApp {
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #f7f7f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #10a37f;
        max-width: 60%;
        margin-left: auto;
        margin-right: 10px;
        text-align: left;
    }
    
    .assistant-message {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #667085;
        max-width: 60%;
        margin-left: 10px;
        margin-right: auto;
        text-align: left;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: white;
        border: 1px solid #4d4d4f;
        border-radius: 5px;
        padding: 10px;
        text-align: left;
    }
    
    .stButton>button:hover {
        background-color: #2d2d30;
    }
    
    /* Send button styling */
    button[kind="primary"] {
        background-color: #10a37f !important;
        color: white !important;
        border: none !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #0d8c6d !important;
    }
    
    /* Remove document button styling */
    button[key="remove_doc_top"] {
        background-color: #ef4444 !important;
        color: white !important;
    }
    
    button[key="remove_doc_top"]:hover {
        background-color: #dc2626 !important;
    }
    
    /* Input box */
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    
    /* Delete button styling */
    button[data-testid*="del_"] {
        padding: 5px !important;
        min-height: 30px !important;
        height: 30px !important;
    }
</style>
""", unsafe_allow_html=True)

# Data directory for saving conversations
DATA_DIR = Path("./legal_ai_data")
CONVERSATIONS_DIR = DATA_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

class ConversationManager:
    """Manages conversation history and persistence"""
    
    def __init__(self):
        self.conversations_file = DATA_DIR / "conversations_index.json"
        self.load_conversations()
    
    def load_conversations(self):
        """Load all conversations from disk"""
        try:
            if self.conversations_file.exists():
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading conversations: {str(e)}")
        return []
    
    def save_conversations(self, conversations):
        """Save conversations index to disk"""
        try:
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving conversations: {str(e)}")
    
    def get_conversation(self, conv_id):
        """Load specific conversation"""
        try:
            conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading conversation {conv_id}: {str(e)}")
        return None
    
    def save_conversation(self, conv_id, messages):
        """Save specific conversation"""
        try:
            conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving conversation {conv_id}: {str(e)}")
    
    def create_conversation(self, title, agent_type):
        """Create new conversation"""
        conversations = self.load_conversations()
        new_conv = {
            'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'title': title,
            'agent_type': agent_type,
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat()
        }
        conversations.insert(0, new_conv)
        self.save_conversations(conversations)
        self.save_conversation(new_conv['id'], [])
        return new_conv['id']
    
    def delete_conversation(self, conv_id):
        """Delete conversation"""
        try:
            conversations = self.load_conversations()
            conversations = [c for c in conversations if c['id'] != conv_id]
            self.save_conversations(conversations)
            
            conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
            if conv_file.exists():
                conv_file.unlink()
        except Exception as e:
            st.error(f"Error deleting conversation: {str(e)}")
    
    def update_conversation_title(self, conv_id, new_title):
        """Update conversation title"""
        conversations = self.load_conversations()
        for conv in conversations:
            if conv['id'] == conv_id:
                conv['title'] = new_title
                conv['updated'] = datetime.now().isoformat()
                break
        self.save_conversations(conversations)
    
    def add_message(self, conv_id, role, content, agent_type=None):
        """Add message to conversation"""
        messages = self.get_conversation(conv_id) or []
        messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'agent_type': agent_type
        })
        self.save_conversation(conv_id, messages)
        
        # Update conversation timestamp
        conversations = self.load_conversations()
        for conv in conversations:
            if conv['id'] == conv_id:
                conv['updated'] = datetime.now().isoformat()
                break
        self.save_conversations(conversations)
    
    def generate_title_summary(self, first_message):
        """Generate a short summary title from the first message"""
        import hashlib
        
        # Simple summarization - take first 40 chars and clean it up
        title = first_message.strip()
        
        # Remove common question words at the start
        for word in ["what", "how", "why", "when", "where", "who", "can", "is", "are", "do", "does", "could", "would", "should"]:
            if title.lower().startswith(word + " "):
                title = title[len(word)+1:]
                break
        
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        
        # Truncate to reasonable length
        if len(title) > 40:
            title = title[:40].rsplit(' ', 1)[0] + "..."
        
        # Add unique identifier to ensure uniqueness
        unique_hash = hashlib.md5(f"{first_message}{datetime.now().isoformat()}".encode()).hexdigest()[:4]
        
        return f"{title}" if title else f"Chat {unique_hash}"
    
    def is_greeting(self, message):
        """Check if message is a greeting"""
        greetings = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'hola', 'howdy', 'sup', 'yo', 'hiya', 'whats up',
            "what's up", 'how are you', 'how do you do'
        ]
        message_lower = message.lower().strip().strip('!.,?')
        return message_lower in greetings or any(message_lower.startswith(g) for g in greetings)

# Mock chatbot class for testing (replace with your actual implementation)
class LegalResearchChatbot:
    """Mock Legal Research Chatbot - Replace with your actual implementation"""
    
    def answer(self, question):
        """Mock answer method - Replace with your actual chatbot logic"""
        return f"This is a mock response to: '{question}'. Please integrate your actual LegalResearchChatbot implementation."

# Initialize session state
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None

if 'current_agent' not in st.session_state:
    st.session_state.current_agent = 'chatbot'

if 'chatbot' not in st.session_state:
    try:
        # Try to import the actual chatbot
        from legal_chatbot import LegalResearchChatbot as ActualChatbot
        st.session_state.chatbot = ActualChatbot()
    except ImportError:
        st.warning("Using mock chatbot. Please add your legal_chatbot.py file with LegalResearchChatbot class.")
        st.session_state.chatbot = LegalResearchChatbot()

if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = {}

if 'current_document' not in st.session_state:
    st.session_state.current_document = None

if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# Agent configurations (easy to extend)
AGENTS = {
    'chatbot': {
        'name': '💬 Legal Research Chatbot',
        'description': 'Ask legal questions and get research-backed answers',
        'icon': '💬',
        'status': 'active',
        'instance': None
    },
    'document_upload': {
        'name': '📄 Document Upload',
        'description': 'Upload and analyze legal documents and case files',
        'icon': '📄',
        'status': 'coming_soon',
        'instance': None
    },
    'insights': {
        'name': '📊 Legal Insights',
        'description': 'Get analytics and insights from your legal database',
        'icon': '📊',
        'status': 'coming_soon',
        'instance': None
    },
    'case_analyzer': {
        'name': '🔍 Case Analyzer',
        'description': 'Deep analysis of legal cases and precedents',
        'icon': '🔍',
        'status': 'coming_soon',
        'instance': None
    }
}

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except ImportError:
                st.error("PyPDF2 not installed. Install with: pip install PyPDF2")
                return None
        
        elif file_extension in ['docx', 'doc']:
            try:
                import docx
                doc = docx.Document(uploaded_file)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                return text
            except ImportError:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return None
        
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def render_sidebar():
    """Render sidebar with conversations and agent selector"""
    with st.sidebar:
        st.title("⚖️ Legal AI Assistant")
        
        # Agent selector
        st.markdown("### AI Agents")
        
        for agent_id, agent_info in AGENTS.items():
            if agent_info['status'] == 'active':
                if st.button(
                    f"{agent_info['icon']} {agent_info['name'].replace(agent_info['icon'] + ' ', '')}",
                    key=f"agent_{agent_id}"
                ):
                    st.session_state.current_agent = agent_id
                    st.rerun()
        
        st.markdown("---")
        
        # New conversation button
        if st.button("➕ New Conversation"):
            # Just reset current conversation, don't create empty one
            st.session_state.current_conversation_id = None
            st.session_state.current_document = None
            st.rerun()
        
        st.markdown("### Conversations")
        
        # List conversations
        conversations = st.session_state.conversation_manager.load_conversations()
        
        if not conversations:
            st.info("No conversations yet. Start a new one!")
        else:
            for conv in conversations:
                cols = st.columns([5, 1])
                
                with cols[0]:
                    # Conversation button
                    agent_type = conv.get('agent_type', 'chatbot')
                    agent_icon = AGENTS.get(agent_type, {}).get('icon', '💬')
                    conv_title = conv['title'][:30] + "..." if len(conv['title']) > 30 else conv['title']
                    
                    if st.button(
                        f"{agent_icon} {conv_title}",
                        key=f"conv_{conv['id']}"
                    ):
                        st.session_state.current_conversation_id = conv['id']
                        st.session_state.current_agent = conv.get('agent_type', 'chatbot')
                        st.rerun()
                
                with cols[1]:
                    # Delete button
                    if st.button("🗑️", key=f"del_{conv['id']}"):
                        st.session_state.conversation_manager.delete_conversation(conv['id'])
                        if st.session_state.current_conversation_id == conv['id']:
                            st.session_state.current_conversation_id = None
                        st.rerun()
        
        st.markdown("---")
        st.caption("Built for AI Audit & Compliance")

def render_chatbot_interface():
    """Render the chatbot interface"""
    
    conv_id = st.session_state.current_conversation_id
    messages = []
    
    if conv_id:
        messages = st.session_state.conversation_manager.get_conversation(conv_id) or []
    
    # Display header
    st.markdown("## 💬 Legal Research Chatbot")
    st.markdown("Ask any legal question and get research-backed answers.")
    
    # Show attached document info at top
    if st.session_state.current_document:
        current_doc = st.session_state.uploaded_documents.get(st.session_state.current_document)
        if current_doc:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.success(f"📄 Document attached: **{current_doc['name']}**")
            with col2:
                if st.button("❌ Remove", key="remove_doc_top"):
                    st.session_state.current_document = None
                    st.rerun()
    
    st.markdown("---")
    
    # Display messages
    if not messages:
        st.info("👋 Welcome! Ask me any legal question to get started. Click ➕ to attach documents.")
    else:
        chat_container = st.container()
        with chat_container:
            for msg in messages:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>Assistant</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input area at bottom
    st.markdown("---")
    
    # Container for plus button and chat input
    input_container = st.container()
    
    with input_container:
        # Plus button and file uploader in same row
        col_plus, col_input, col_send = st.columns([0.5, 5.5, 0.5])
        
        with col_plus:
            if st.button("➕", key="toggle_uploader", help="Attach document"):
                st.session_state.show_uploader = not st.session_state.show_uploader
                st.rerun()
        
        # Show file uploader when toggled (above the input)
        if st.session_state.show_uploader:
            st.markdown("")  # Small spacing
            uploaded_file = st.file_uploader(
                "Choose a file to analyze",
                type=['pdf', 'docx', 'doc', 'txt'],
                key="chat_file_upload",
                help="Upload PDF, DOCX, DOC, or TXT files"
            )
            
            if uploaded_file is not None:
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.current_document != file_key:
                    with st.spinner("Processing document..."):
                        text = extract_text_from_file(uploaded_file)
                        if text:
                            st.session_state.uploaded_documents[file_key] = {
                                'name': uploaded_file.name,
                                'text': text,
                                'uploaded_at': datetime.now().isoformat()
                            }
                            st.session_state.current_document = file_key
                            st.session_state.show_uploader = False
                            st.success(f"✅ Document attached!")
                            st.rerun()
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            form_cols = st.columns([6, 0.5])
            
            with form_cols[0]:
                user_input = st.text_input(
                    "Your question:",
                    key=f"user_input_{st.session_state.input_key}",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            
            with form_cols[1]:
                send_button = st.form_submit_button("➤", type="primary")
            
            if send_button and user_input:
                # Create conversation on first message if it doesn't exist
                if not conv_id:
                    conv_id = st.session_state.conversation_manager.create_conversation(
                        "New Chat",
                        'chatbot'
                    )
                    st.session_state.current_conversation_id = conv_id
                
                process_user_input(conv_id, user_input, messages)

def process_user_input(conv_id, user_input, messages):
    """Process user input and get bot response"""
    try:
        # Add user message
        st.session_state.conversation_manager.add_message(
            conv_id,
            'user',
            user_input,
            'chatbot'
        )
        
        # Get bot response
        with st.spinner("Thinking..."):
            try:
                # If document is attached, include context
                if st.session_state.current_document:
                    current_doc = st.session_state.uploaded_documents.get(st.session_state.current_document)
                    if current_doc:
                        context_prompt = f"""Based on the following document, please answer the user's question.

Document: {current_doc['name']}

Document Content (first 4000 chars):
{current_doc['text'][:4000]}

User Question: {user_input}

Please provide a clear and concise answer."""
                        response = st.session_state.chatbot.answer(context_prompt)
                    else:
                        response = st.session_state.chatbot.answer(user_input)
                else:
                    response = st.session_state.chatbot.answer(user_input)
                
                # Add assistant message
                st.session_state.conversation_manager.add_message(
                    conv_id,
                    'assistant',
                    response,
                    'chatbot'
                )
                
                # Update conversation title logic
                # Count only user messages (not assistant responses)
                user_messages = [msg for msg in messages if msg['role'] == 'user']
                
                # If this is the first message and it's a greeting, don't set title yet
                if len(user_messages) == 0:
                    if st.session_state.conversation_manager.is_greeting(user_input):
                        # Keep default title for now
                        pass
                    else:
                        # First message and not a greeting, set title
                        title = st.session_state.conversation_manager.generate_title_summary(user_input)
                        st.session_state.conversation_manager.update_conversation_title(conv_id, title)
                
                # If this is the second user message, set title based on this message
                elif len(user_messages) == 1:
                    # Check if first message was a greeting
                    first_msg = user_messages[0]['content']
                    if st.session_state.conversation_manager.is_greeting(first_msg):
                        # Set title based on second message
                        title = st.session_state.conversation_manager.generate_title_summary(user_input)
                        st.session_state.conversation_manager.update_conversation_title(conv_id, title)
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.session_state.conversation_manager.add_message(
                    conv_id,
                    'assistant',
                    error_msg,
                    'chatbot'
                )
                st.error(error_msg)
        
        # Clear input and rerun
        st.session_state.input_key += 1
        st.rerun()
        
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_coming_soon(agent_name, description):
    """Render coming soon page for future agents"""
    st.markdown(f"## {agent_name}")
    st.markdown(description)
    st.markdown("---")
    
    st.info("🚧 This agent is under development and will be available soon!")
    
    st.markdown("### Planned Features:")
    
    if "Document Upload" in agent_name:
        st.markdown("""
        - 📤 Upload PDF, DOCX, and TXT legal documents
        - 🔍 Automatic document parsing and extraction
        - 🗂️ Organize documents by case, type, or category
        - 🔗 Link documents to relevant cases in the database
        - 🤖 AI-powered document summarization
        """)
    
    elif "Insights" in agent_name:
        st.markdown("""
        - 📊 Database analytics and statistics
        - 📈 Trend analysis across cases and regulations
        - 🗺️ Jurisdiction coverage maps
        - 🎯 Topic clustering and theme extraction
        - 📋 Generate compliance reports
        """)
    
    elif "Case Analyzer" in agent_name:
        st.markdown("""
        - 🔍 Deep dive into specific cases
        - ⚖️ Compare similar cases across jurisdictions
        - 📖 Extract legal precedents and principles
        - 🔗 Find related cases automatically
        - 📝 Generate case briefs and summaries
        """)

def main():
    """Main application"""
    
    # Render sidebar
    render_sidebar()
    
    # Render main content based on selected agent
    if st.session_state.current_agent == 'chatbot':
        render_chatbot_interface()
    
    elif st.session_state.current_agent == 'document_upload':
        render_coming_soon(
            AGENTS['document_upload']['name'],
            AGENTS['document_upload']['description']
        )
    
    elif st.session_state.current_agent == 'insights':
        render_coming_soon(
            AGENTS['insights']['name'],
            AGENTS['insights']['description']
        )
    
    elif st.session_state.current_agent == 'case_analyzer':
        render_coming_soon(
            AGENTS['case_analyzer']['name'],
            AGENTS['case_analyzer']['description']
        )

if __name__ == "__main__":
    main()