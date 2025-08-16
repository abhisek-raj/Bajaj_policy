import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="Bajaj Insurance Policy Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* ================================
   🌌 Violet-Red-Blue Dark Theme
   ================================ */
:root {
    /* Primary Colors */
    --primary: #7c4dff;           /* Vibrant Violet */
    --primary-dark: #5e35b1;      /* Deep Violet */
    --primary-light: #b388ff;     /* Light Violet */
    --accent: #ff1744;            /* Bright Red */
    --accent2: #2979ff;           /* Bright Blue */
    
    /* Background Colors */
    --background: #0b0b0f;        /* Nearly Black */
    --surface: #1a1a27;           /* Very Dark Navy-Black */
    --surface-light: #2c2c3f;     /* Dark Slate Blue */
    --surface-highlight: #3f3f5c; /* Slate Blue Highlight */
    
    /* Text Colors */
    --text: #e8e6f1;              /* Soft Off-White */
    --text-secondary: #b3a8c8;    /* Muted Lavender Gray */
    --text-tertiary: #817ca8;     /* Medium Purple Gray */
    
    /* Status Colors */
    --success: #4caf50;           /* Green */
    --warning: #ffb300;           /* Amber */
    --error: #ff1744;             /* Bright Red */
    --info: #2979ff;              /* Bright Blue */
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-dark) 100%);
    --gradient-accent: linear-gradient(135deg, var(--accent) 0%, var(--primary) 100%);
    
    /* Shadow & Effects */
    --shadow-sm: 0 4px 6px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 10px 15px rgba(0, 0, 0, 0.5);
    --shadow-lg: 0 20px 25px rgba(0, 0, 0, 0.7);
    
    /* Transitions */
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ================================
   🌌 Main App Styling
   ================================ */
.stApp {
    background-color: var(--background);
    color: var(--text);
    background-image:
        radial-gradient(at 0% 0%, rgba(124, 77, 255, 0.1) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(255, 23, 68, 0.1) 0px, transparent 50%);
    transition: background-color 0.4s ease-in-out;
}

/* Typography */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text);
    line-height: 1.5;
}

/* ================================
   🎯 Buttons
   ================================ */
.stButton > button {
    background: var(--gradient-primary);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: var(--shadow-md);
    background: var(--primary-dark);
}

/* ================================
   📊 Cards
   ================================ */
.metric-card, .stat-card {
    background: var(--surface);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid var(--surface-highlight);
    margin: 0.5rem 0;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}

.metric-card:hover, .stat-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary);
}

/* ================================
   📝 Inputs & Form Fields
   ================================ */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[role="button"] {
    background-color: var(--surface-light) !important;
    color: var(--text) !important;
    border: 1px solid var(--surface-highlight) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    transition: var(--transition);
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(124, 77, 255, 0.4) !important;
}

/* ================================
   📑 Tabs
   ================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 0 4px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--surface-light);
    color: var(--text-secondary);
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    transition: var(--transition);
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary) !important;
    color: #fff !important;
    border-color: var(--primary-light) !important;
    box-shadow: var(--shadow-sm);
}

/* ================================
   📌 Sidebar
   ================================ */
.css-1d391kg {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--surface-highlight);
}

/* ================================
   💬 Chat Messages
   ================================ */
.stChatMessage {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    animation: fadeIn 0.4s ease-in-out;
}

.stChatMessage[data-testid="user"] {
    background: var(--primary-dark);
    color: white;
}

.stChatMessage[data-testid="assistant"] {
    background: var(--surface-light);
    color: var(--text);
}

/* ================================
   ⚠ Alerts
   ================================ */
.stAlert {
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid;
}

.stAlert-success { border-left-color: var(--success); background-color: rgba(76, 175, 80, 0.15); }
.stAlert-warning { border-left-color: var(--warning); background-color: rgba(255, 179, 0, 0.15); }
.stAlert-error { border-left-color: var(--error); background-color: rgba(255, 23, 68, 0.15); }
.stAlert-info { border-left-color: var(--info); background-color: rgba(41, 121, 255, 0.15); }

/* ================================
   📜 Scrollbar
   ================================ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--surface);
}
::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 4px;
    transition: var(--transition);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary-light);
}

/* ================================
   ✨ Animations
   ================================ */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)
import os
from dotenv import load_dotenv
load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"
hf_token = os.getenv("HF_TOKEN")

# Multilingual support
LANGUAGES = {
    "English": {
        "title": "Bajaj Insurance Policy Assistant",
        "upload_doc": "Upload Policy Document",
        "chat_placeholder": "Ask about your insurance policy (e.g., '46M, knee surgery, Pune, 3-month policy')",
        "dashboard_title": "Bajaj Policy Dashboard",
        "total_policies": "Total Policies",
        "active_policies": "Active Policies",
        "total_premium": "Total Premium",
        "claims_approved": "Claims Approved",
        "language": "Language",
        "select_language": "Select Language",
        "search_db": "Search Bajaj Insurance Policy",
        "vector_explorer": "Bajaj Policy Explorer",
        "analytics": "Analytics Dashboard",
        "search_language": "Search Language",
        "select_search_language": "Select search language:",
        "enter_search_query": "Enter your search query",
        "search_placeholder": "e.g., knee surgery, policy terms, coverage limits, exclusions"
    },
    "Hindi": {
        "title": "बजाज बीमा पॉलिसी सहायक",
        "upload_doc": "पॉलिसी दस्तावेज़ अपलोड करें",
        "chat_placeholder": "अपनी बीमा पॉलिसी के बारे में पूछें (जैसे, '46M, घुटने की सर्जरी, पुणे, 3-महीने की पॉलिसी')",
        "dashboard_title": "बजाज पॉलिसी डैशबोर्ड",
        "total_policies": "कुल पॉलिसियां",
        "active_policies": "सक्रिय पॉलिसियां",
        "total_premium": "कुल प्रीमियम",
        "claims_approved": "अनुमोदित दावे",
        "language": "भाषा",
        "select_language": "भाषा चुनें",
        "search_db": "बजाज बीमा पॉलिसी खोजें",
        "vector_explorer": "बजाज पॉलिसी एक्सप्लोरर",
        "analytics": "विश्लेषण डैशबोर्ड",
        "search_language": "खोज भाषा",
        "select_search_language": "खोज भाषा चुनें:",
        "enter_search_query": "अपना खोज प्रश्न दर्ज करें",
        "search_placeholder": "जैसे, घुटने की सर्जरी, पॉलिसी शर्तें, कवरेज सीमाएं, अपवर्जन"
    },
    "Marathi": {
        "title": "बजाज विमा पॉलिसी सहाय्यक",
        "upload_doc": "पॉलिसी दस्तऐवज अपलोड करा",
        "chat_placeholder": "तुमच्या विमा पॉलिसीबद्दल विचारा (उदा., '46M, गुडघ्याची शस्त्रक्रिया, पुणे, 3-महिन्यांची पॉलिसी')",
        "dashboard_title": "बजाज पॉलिसी डॅशबोर्ड",
        "total_policies": "एकूण पॉलिसी",
        "active_policies": "सक्रिय पॉलिसी",
        "total_premium": "एकूण प्रीमियम",
        "claims_approved": "मंजूर दावे",
        "language": "भाषा",
        "select_language": "भाषा निवडा",
        "search_db": "बजाज विमा पॉलिसी शोधा",
        "vector_explorer": "बजाज पॉलिसी एक्सप्लोरर",
        "analytics": "विश्लेषण डॅशबोर्ड",
        "search_language": "शोध भाषा",
        "select_search_language": "शोध भाषा निवडा:",
        "enter_search_query": "तुमचा शोध प्रश्न प्रविष्ट करा",
        "search_placeholder": "उदा., गुडघ्याची शस्त्रक्रिया, पॉलिसी अटी, कव्हरेज मर्यादा, अपवर्जन"
    },
    "Spanish": {
        "title": "Asistente de Pólizas de Seguro Bajaj",
        "upload_doc": "Subir Documento de Póliza",
        "chat_placeholder": "Pregunta sobre tu póliza de seguro (ej., '46M, cirugía de rodilla, Pune, póliza de 3 meses')",
        "dashboard_title": "Panel de Pólizas Bajaj",
        "total_policies": "Total de Pólizas",
        "active_policies": "Pólizas Activas",
        "total_premium": "Prima Total",
        "claims_approved": "Reclamos Aprobados",
        "language": "Idioma",
        "select_language": "Seleccionar Idioma",
        "search_db": "Buscar Póliza de Seguro Bajaj",
        "vector_explorer": "Explorador de Pólizas Bajaj",
        "analytics": "Panel de Análisis",
        "search_language": "Idioma de Búsqueda",
        "select_search_language": "Seleccionar idioma de búsqueda:",
        "enter_search_query": "Ingresa tu consulta de búsqueda",
        "search_placeholder": "ej., cirugía de rodilla, términos de póliza, límites de cobertura, exclusiones"
    },
    "French": {
        "title": "Assistant de Police d'Assurance Bajaj",
        "upload_doc": "Télécharger le Document de Police",
        "chat_placeholder": "Posez des questions sur votre police d'assurance (ex., '46M, chirurgie du genou, Pune, police de 3 mois')",
        "dashboard_title": "Tableau de Bord des Polices Bajaj",
        "total_policies": "Total des Polices",
        "active_policies": "Polices Actives",
        "total_premium": "Prime Totale",
        "claims_approved": "Réclamations Approuvées",
        "language": "Langue",
        "select_language": "Sélectionner la Langue",
        "search_db": "Rechercher Police d'Assurance Bajaj",
        "vector_explorer": "Explorateur de Polices Bajaj",
        "analytics": "Tableau de Bord Analytique",
        "search_language": "Langue de Recherche",
        "select_search_language": "Sélectionner la langue de recherche:",
        "enter_search_query": "Entrez votre requête de recherche",
        "search_placeholder": "ex., chirurgie du genou, termes de police, limites de couverture, exclusions"
    },
    "German": {
        "title": "Bajaj Versicherungspolice Assistent",
        "upload_doc": "Police Dokument hochladen",
        "chat_placeholder": "Fragen Sie nach Ihrer Versicherungspolice (z.B. '46M, Knieoperation, Pune, 3-Monats-Police')",
        "dashboard_title": "Bajaj Police Dashboard",
        "total_policies": "Gesamte Policen",
        "active_policies": "Aktive Policen",
        "total_premium": "Gesamtprämie",
        "claims_approved": "Genehmigte Ansprüche",
        "language": "Sprache",
        "select_language": "Sprache auswählen",
        "search_db": "Bajaj Versicherungspolice suchen",
        "vector_explorer": "Bajaj Police Explorer",
        "analytics": "Analytics Dashboard",
        "search_language": "Suchsprache",
        "select_search_language": "Suchsprache auswählen:",
        "enter_search_query": "Geben Sie Ihre Suchanfrage ein",
        "search_placeholder": "z.B., Knieoperation, Policenbedingungen, Deckungsgrenzen, Ausschlüsse"
    },
    "Chinese": {
        "title": "Bajaj保险政策助手",
        "upload_doc": "上传保险单文件",
        "chat_placeholder": "询问您的保险单（例如，'46M，膝关节手术，浦那，3个月保单'）",
        "dashboard_title": "Bajaj保单仪表板",
        "total_policies": "总保单数",
        "active_policies": "活跃保单",
        "total_premium": "总保费",
        "claims_approved": "已批准索赔",
        "language": "语言",
        "select_language": "选择语言",
        "search_db": "搜索Bajaj保险单",
        "vector_explorer": "Bajaj保单浏览器",
        "analytics": "分析仪表板",
        "search_language": "搜索语言",
        "select_search_language": "选择搜索语言:",
        "enter_search_query": "输入您的搜索查询",
        "search_placeholder": "例如，膝关节手术，保单条款，承保限额，除外责任"
    },
    "Japanese": {
        "title": "Bajaj保険証券アシスタント",
        "upload_doc": "保険証券書類をアップロード",
        "chat_placeholder": "保険証券について質問する（例：'46M、膝手術、プネー、3ヶ月証券'）",
        "dashboard_title": "Bajaj証券ダッシュボード",
        "total_policies": "総証券数",
        "active_policies": "アクティブ証券",
        "total_premium": "総保険料",
        "claims_approved": "承認済み請求",
        "language": "言語",
        "select_language": "言語を選択",
        "search_db": "Bajaj保険証券を検索",
        "vector_explorer": "Bajaj証券エクスプローラー",
        "analytics": "分析ダッシュボード",
        "search_language": "検索言語",
        "select_search_language": "検索言語を選択:",
        "enter_search_query": "検索クエリを入力",
        "search_placeholder": "例、膝手術、証券条件、カバレッジ制限、除外事項"
    },
    "Arabic": {
        "title": "مساعد بوليصة التأمين باجاج",
        "upload_doc": "رفع وثيقة البوليصة",
        "chat_placeholder": "اسأل عن بوليصة التأمين الخاصة بك (مثل، '46M، جراحة الركبة، بونه، بوليصة 3 أشهر')",
        "dashboard_title": "لوحة تحكم بوليصة باجاج",
        "total_policies": "إجمالي البوالص",
        "active_policies": "البوالص النشطة",
        "total_premium": "إجمالي الأقساط",
        "claims_approved": "المطالبات المعتمدة",
        "language": "اللغة",
        "select_language": "اختر اللغة",
        "search_db": "البحث عن بوليصة تأمين باجاج",
        "vector_explorer": "مستكشف بوليصة باجاج",
        "analytics": "لوحة تحكم التحليلات",
        "search_language": "لغة البحث",
        "select_search_language": "اختر لغة البحث:",
        "enter_search_query": "أدخل استعلام البحث",
        "search_placeholder": "مثل، جراحة الركبة، شروط البوليصة، حدود التغطية، الاستثناءات"
    },
    "Russian": {
        "title": "Помощник по страховым полисам Bajaj",
        "upload_doc": "Загрузить документ полиса",
        "chat_placeholder": "Спросите о вашем страховом полисе (например, '46M, операция на колене, Пуна, полис на 3 месяца')",
        "dashboard_title": "Панель полисов Bajaj",
        "total_policies": "Всего полисов",
        "active_policies": "Активные полисы",
        "total_premium": "Общая премия",
        "claims_approved": "Одобренные претензии",
        "language": "Язык",
        "select_language": "Выбрать язык",
        "search_db": "Поиск страхового полиса Bajaj",
        "vector_explorer": "Исследователь полисов Bajaj",
        "analytics": "Панель аналитики",
        "search_language": "Язык поиска",
        "select_search_language": "Выбрать язык поиска:",
        "enter_search_query": "Введите поисковый запрос",
        "search_placeholder": "например, операция на колене, условия полиса, лимиты покрытия, исключения"
    },
    "Portuguese": {
        "title": "Assistente de Apólice de Seguro Bajaj",
        "upload_doc": "Carregar Documento da Apólice",
        "chat_placeholder": "Pergunte sobre sua apólice de seguro (ex., '46M, cirurgia no joelho, Pune, apólice de 3 meses')",
        "dashboard_title": "Painel de Apólices Bajaj",
        "total_policies": "Total de Apólices",
        "active_policies": "Apólices Ativas",
        "total_premium": "Prêmio Total",
        "claims_approved": "Sinistros Aprovados",
        "language": "Idioma",
        "select_language": "Selecionar Idioma",
        "search_db": "Pesquisar Apólice de Seguro Bajaj",
        "vector_explorer": "Explorador de Apólices Bajaj",
        "analytics": "Painel de Análise",
        "search_language": "Idioma de Pesquisa",
        "select_search_language": "Selecionar idioma de pesquisa:",
        "enter_search_query": "Digite sua consulta de pesquisa",
        "search_placeholder": "ex., cirurgia no joelho, termos da apólice, limites de cobertura, exclusões"
    }
}

# Generate sample Bajaj policy data for 5 years
def generate_bajaj_policy_data():
    """Generate sample Bajaj policy data for the last 5 years"""
    data = []
    current_year = datetime.now().year
    
    for year in range(current_year - 4, current_year + 1):
        for month in range(1, 13):
            # Generate realistic policy data
            policies_count = random.randint(150, 300)
            active_policies = random.randint(120, 280)
            total_premium = random.randint(5000000, 15000000)
            claims_approved = random.randint(80, 200)
            
            data.append({
                'Year': year,
                'Month': month,
                'Month_Name': datetime(year, month, 1).strftime('%B'),
                'Total_Policies': policies_count,
                'Active_Policies': active_policies,
                'Total_Premium': total_premium,
                'Claims_Approved': claims_approved,
                'Policy_Types': {
                    'Health': random.randint(40, 80),
                    'Motor': random.randint(30, 60),
                    'Life': random.randint(20, 50),
                    'Property': random.randint(10, 30)
                }
            })
    
    return pd.DataFrame(data)

# -------- Load default vectorstore --------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# -------- Fetch data from vector database --------
def fetch_from_vector_db(query_text, k=5):
    """Fetch documents from vector database based on similarity search"""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return None, "Failed to load vector store"
        
        # Perform similarity search
        docs_and_scores = vectorstore.similarity_search_with_score(query_text, k=k)
        
        results = []
        for i, (doc, score) in enumerate(docs_and_scores):
            result = {
                "rank": i + 1,
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            results.append(result)
        
        return results, None
    except Exception as e:
        return None, str(e)

# -------- Custom prompt --------
def set_custom_prompt():
    custom_prompt_template = """
    You are an insurance policy expert. Use the provided context to answer the user's query about insurance coverage.
    Return a structured JSON response with:
    - "decision": "approved" or "rejected"
    - "amount": float or null
    - "justification": Concise, one-line summary (e.g., "Yes, knee surgery is covered under Clause 2.3 of the policy.")
    - "clauses_referenced": List of objects with "file", "page", and "clause"
    
    If the answer is not in the context, say so and return "rejected" with null amount.
    
    Context: {context}
    Query: {question}
    
    Output in JSON format:
    {{
        "decision": "approved" or "rejected",
        "amount": float or null,
        "justification": "string",
        "clauses_referenced": [
            {{"file": "string", "page": "string", "clause": "string"}}
        ]
    }}
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# -------- Load LLM --------
def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=512,
        model_kwargs={}
    )
    return llm

# -------- Dashboard Page --------
def dashboard_page(translations):
    # Main header with gradient
    st.markdown(f"""
    <div class="main-header">
        <h1>� {translations['dashboard_title']}</h1>
        <p>Comprehensive overview of Bajaj insurance policies and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_bajaj_policy_data()
    current_year = datetime.now().year
    
    # Top row - Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Total Policies</div>
            <div class="stat-value">12,845</div>
            <div style="color: #38b000; font-size: 0.9rem;">↑ 12.5% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Active Policies</div>
            <div class="stat-value">9,432</div>
            <div style="color: #38b000; font-size: 0.9rem;">↑ 8.2% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Total Premium</div>
            <div class="stat-value">₹42.8M</div>
            <div style="color: #38b000; font-size: 0.9rem;">↑ 6.7% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Claims Approved</div>
            <div class="stat-value">1,284</div>
            <div style="color: #ff9e00; font-size: 0.9rem;">↓ 2.3% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Middle row - Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### � Policy Growth (Last 12 Months)")
        
        # Sample monthly data
        months = pd.date_range(end=datetime.now(), periods=12, freq='ME').strftime('%b %Y')
        policy_counts = [random.randint(800, 1200) for _ in range(12)]
        premium_amounts = [random.randint(2500000, 4200000) for _ in range(12)]
        
        # Create area chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=policy_counts,
            fill='tozeroy',
            mode='lines+markers',
            name='Policies Sold',
            line=dict(color='#9d4edd', width=3),
            marker=dict(size=8, color='#7b2cbf'),
            hovertemplate='%{y:,.0f} policies<extra></extra>'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Top Performing Plans")
        
        # Sample plan data
        plans = ['Health Shield', 'Motor Protect', 'Life Secure', 'Home Guard', 'Travel Safe']
        policies_sold = [3420, 2870, 2150, 1890, 1560]
        
        fig = go.Figure(go.Bar(
            x=policies_sold,
            y=plans,
            orientation='h',
            marker_color='#7b2cbf',
            text=[f"{x:,}" for x in policies_sold],
            textposition='auto'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=30, b=20),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom row - Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Claim Settlement Ratio")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 94.7,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Month", 'font': {'size': 14}},
            delta = {'reference': 92.3, 'increasing': {'color': "#38b000"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#7b2cbf"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': '#ff6b6b'},
                    {'range': [70, 90], 'color': '#ffd166'},
                    {'range': [90, 100], 'color': '#06d6a0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 94.7
                }
            }
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Policy Distribution by Type")
        
        # Sample policy type data
        policy_types = ['Health', 'Motor', 'Life', 'Home', 'Travel']
        distribution = [35, 28, 22, 10, 5]
        
        fig = go.Figure(go.Pie(
            labels=policy_types,
            values=distribution,
            hole=0.5,
            marker_colors=['#7b2cbf', '#9d4edd', '#c77dff', '#e0aaff', '#f3e5ff'],
            textinfo='label+percent',
            textposition='inside',
            hovertemplate='%{label}: %{percent}' 
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 🚀 Quick Actions")
        
        # Quick action buttons
        if st.button("📄 Issue New Policy", use_container_width=True):
            st.session_state.page = "💬 Chat Bot"
            st.rerun()
            
        if st.button("📊 Generate Report", use_container_width=True):
            st.toast("Generating monthly report...", icon="📊")
            
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
            
        st.markdown("---")
        
        # Recent activities
        st.markdown("### 📝 Recent Activities")
        
        activities = [
            {"time": "2 min ago", "action": "New policy issued", "details": "Health Shield - ₹25,000"},
            {"time": "15 min ago", "action": "Claim approved", "details": "#CL-2023-04521 - ₹18,750"},
            {"time": "1 hour ago", "action": "Policy renewed", "details": "Motor Protect - ₹12,340"},
            {"time": "3 hours ago", "action": "New customer onboarded", "details": "John D. - Corporate Plan"},
            {"time": "5 hours ago", "action": "Document verified", "details": "Policy #POL-2023-1245"}
        ]
        
        for activity in activities:
            with st.container():
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 0.75rem; background: var(--surface-light); border-radius: 8px;">
                    <div style="font-weight: 600;">{activity['action']}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary);">{activity['details']}</div>
                    <div style="font-size: 0.7rem; color: var(--primary-light);">{activity['time']}</div>
                </div>
                """, unsafe_allow_html=True)
    
# -------- Chat Bot Page --------
def chat_bot_page(translations):
    # Main header with gradient
    st.markdown(f"""
    <div class="main-header">
        <h1>💬 {translations['title']}</h1>
        <p>Get instant answers about your insurance policies using AI-powered assistance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Bajaj Insurance Assistant. How can I help you today?"}
        ]
    
    # Sidebar for settings and info
    with st.sidebar:
        st.markdown("### ⚙️ Chat Settings")
        
        # Model selection
        model = st.selectbox(
            "Select AI Model",
            ["Bajaj Insurance Expert", "General Insurance", "Claims Specialist"],
            index=0
        )
        
        # Temperature slider
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1,
                              help="Higher values make responses more creative but less focused")
        
        # Language selection
        language = st.selectbox(
            "Chat Language",
            ["English", "हिंदी", "मराठी", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം"],
            index=0
        )
        
        st.markdown("---")
        
        # Upload policy document
        st.markdown("### 📄 Upload Policy Document")
        uploaded_file = st.file_uploader(
            "Upload your policy document (PDF, DOCX, or images)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I assist you now?"}]
            st.rerun()
    
    # Main chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar="🛡️"):
                    st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your insurance policy..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Simulate AI response (in a real app, this would call your AI model)
        with st.spinner("Thinking..."):
            # Simulate processing time
            time.sleep(1)
            
            # Sample responses based on query
            if "coverage" in prompt.lower():
                response = """
                Your Bajaj Health Shield policy provides comprehensive coverage including:
                - In-patient hospitalization up to ₹10,00,000
                - Pre and post-hospitalization expenses
                - Day care procedures
                - Ambulance charges up to ₹5,000
                
                For specific coverage details, please refer to your policy document or ask about a particular treatment.
                """
            elif "claim" in prompt.lower():
                response = """
                To file a claim:
                1. Notify us within 48 hours of hospitalization
                2. Submit the claim form along with:
                   - Doctor's prescription
                   - Hospital bills and receipts
                   - Discharge summary
                   
                You can track your claim status online or through our mobile app.
                """
            elif "premium" in prompt.lower():
                response = """
                Your annual premium is ₹25,000 (including GST).
                
                Payment options:
                - Full payment (2% discount available)
                - Semi-annual
                - Quarterly (processing fee applies)
                - Monthly (credit card required)
                
                Next premium due: 15th October 2023
                """
            else:
                response = """
                I'm your Bajaj Insurance Assistant. I can help you with:
                - Policy details and coverage
                - Claim assistance
                - Premium payments
                - Policy renewal
                - Network hospitals
                
                Please let me know how I can assist you today!
                """
            
            # Format the response with structured output
            def format_response(response_text):
                """
                Format the LLM response with proper markdown styling and structure
                Supports: sections, lists, tables, code blocks, and key-value pairs
                """
                if not response_text:
                    return ""
                    
                formatted_lines = []
                in_code_block = False
                in_table = False
                
                # Split into lines for processing
                lines = response_text.split('\n')
                i = 0
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Handle code blocks
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        formatted_lines.append(line)
                        i += 1
                        continue
                        
                    if in_code_block:
                        formatted_lines.append(line)
                        i += 1
                        continue
                        
                    # Handle empty lines
                    if not line:
                        formatted_lines.append('')
                        i += 1
                        continue
                        
                    # Handle section headers
                    if line.lower().startswith(('policy', 'coverage', 'premium', 'exclusion', 'claim', 'benefit', 'limit')) and \
                       line.lower().endswith(':'):
                        icon = '📋'  # Default icon
                        if 'coverage' in line.lower():
                            icon = '🛡️'
                        elif 'premium' in line.lower():
                            icon = '💰'
                        elif 'exclu' in line.lower():
                            icon = '❌'
                        elif 'claim' in line.lower():
                            icon = '📝'
                        formatted_lines.append(f"### {icon} {line}\n")
                        i += 1
                        continue
                        
                    # Handle lists
                    if line.startswith(('-', '*', '•')):
                        # Handle multi-line list items
                        list_item = [line]
                        j = i + 1
                        while j < len(lines) and lines[j].startswith('  '):
                            list_item.append(lines[j].strip())
                            j += 1
                        
                        # Format the list item
                        bullet = '•' if line.startswith(('-', '•')) else line[0]
                        content = ' '.join([x.lstrip('-*• ').strip() for x in list_item])
                        formatted_lines.append(f"{bullet} {content}")
                        i = j
                        continue
                        
                    # Handle key-value pairs
                    if ':' in line and len(line) < 100:  # Simple heuristic for key-value
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Special formatting for important fields
                        if any(x in key.lower() for x in ['amount', 'limit', 'sum', 'premium']):
                            formatted_lines.append(f"**{key}:** <span style='color: #4cc9f0'>{value}</span>")
                        elif any(x in key.lower() for x in ['coverage', 'included', 'yes']):
                            formatted_lines.append(f"**{key}:** <span style='color: #52b788'>✓ {value}</span>")
                        elif any(x in key.lower() for x in ['excluded', 'no', 'not covered']):
                            formatted_lines.append(f"**{key}:** <span style='color: #ef476f'>✗ {value}</span>")
                        else:
                            formatted_lines.append(f"**{key}:** {value}")
                        i += 1
                        continue
                        
                    # Handle tables (simple markdown tables)
                    if '|' in line and any(c.isalpha() for c in line.replace('|', '')):
                        if not in_table:
                            in_table = True
                            formatted_lines.append('')
                        formatted_lines.append(line)
                        i += 1
                        continue
                    else:
                        if in_table:
                            formatted_lines.append('')
                            in_table = False
                        
                    # Regular text
                    formatted_lines.append(line)
                    i += 1
                
                # Join lines and clean up
                result = '\n'.join(formatted_lines)
                
                # Add some spacing after sections
                for section in ['###', '##', '#']:
                    result = result.replace(f"{section} ", f"\n{section} ")
                
                return result

            # Format the response
            formatted_response = format_response(response)

            # Display the response with proper formatting
            with st.chat_message("assistant"):
                st.markdown(formatted_response, unsafe_allow_html=True)

            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
    
    # Help section below chat
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - Ask about your policy coverage, claims, or premium details
        - Type 'help' to see what I can assist you with
        - Upload your policy document for specific information
        - Our customer support is available 24/7 at 1800-123-4567
        """)

# -------- Vector Explorer Page --------
def vector_explorer_page(translations):
    st.markdown(f"""
    <div class="main-header">
        <h1>🔍 {translations['vector_explorer']}</h1>
        <p>Search and explore the contents of your insurance policy database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search Interface
    st.markdown("### 🔍 Search Bajaj Insurance Policy")
    
    # Language selector for search
    st.markdown(f"#### 🌐 {translations['search_language']}")
    search_language = st.selectbox(
        translations['select_search_language'],
        options=["English", "Hindi", "Marathi", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic", "Russian", "Portuguese"],
        index=0,
        help="Choose the language for your search query"
    )
    
    # Language-specific placeholders
    placeholders = {
        "English": translations['search_placeholder'],
        "Hindi": LANGUAGES["Hindi"]['search_placeholder'],
        "Marathi": LANGUAGES["Marathi"]['search_placeholder'],
        "Spanish": LANGUAGES["Spanish"]['search_placeholder'],
        "French": LANGUAGES["French"]['search_placeholder'],
        "German": LANGUAGES["German"]['search_placeholder'],
        "Chinese": LANGUAGES["Chinese"]['search_placeholder'],
        "Japanese": LANGUAGES["Japanese"]['search_placeholder'],
        "Arabic": LANGUAGES["Arabic"]['search_placeholder'],
        "Russian": LANGUAGES["Russian"]['search_placeholder'],
        "Portuguese": LANGUAGES["Portuguese"]['search_placeholder']
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            f"{translations['enter_search_query']} ({search_language}):",
            placeholder=placeholders[search_language]
        )
    
    with col2:
        k_results = st.number_input("Results:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search Bajaj Insurance Policy", use_container_width=True):
        if search_query:
            with st.spinner("🔍 Searching Bajaj Insurance Policy..."):
                # Store search results in session state
                st.session_state.search_results = fetch_from_vector_db(search_query, k_results)
                st.session_state.show_all_results = False
        else:
            st.warning("⚠️ Please enter a search query.")
    
    # Display search results from session state if available
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results[0]:
        results, error = st.session_state.search_results
        
        if error:
            st.markdown(f"""
            <div class="error-message">
                ❌ Error: {error}
            </div>
            """, unsafe_allow_html=True)
        elif results:
            st.markdown(f"""
            <div class="success-message">
                ✅ Found {len(results)} result{'s' if len(results) > 1 else ''} for "{search_query}"
            </div>
            """, unsafe_allow_html=True)
            
            # Always show the best result
            best_result = results[0]
            policy_title = best_result.get('metadata', {}).get('title', 'Untitled Policy')
            with st.expander(f"Policy: {policy_title} (Score: {best_result['score']:.4f})", expanded=True):
                st.markdown("**📝 Content:**")
                st.text(best_result['content'])
                
                # Show metadata if available
                if 'metadata' in best_result and best_result['metadata']:
                    st.markdown("**📋 Metadata:**")
                    st.json(best_result['metadata'])
            
            # Show button to display all results if there are more than 1
            if len(results) > 1 and not st.session_state.get('show_all_results', False):
                if st.button("🔍 Show All Results", key="show_all_btn"):
                    st.session_state.show_all_results = True
                    st.rerun()
            
            # Show remaining results if requested
            if st.session_state.get('show_all_results', False) and len(results) > 1:
                st.markdown("### 🔍 Additional Results")
                for i, result in enumerate(results[1:], 2):  # Skip first result as it's already shown
                    policy_title = result.get('metadata', {}).get('title', 'Untitled Policy')
                    with st.expander(f"📄 Result {i} (Score: {result['score']:.4f})"):
                        st.markdown("**📝 Content:**")
                        st.text(result['content'])
                        
                        # Show metadata if available
                        if 'metadata' in result and result['metadata']:
                            st.markdown("**📋 Metadata:**")
                            st.json(result['metadata'])
        else:
            st.warning("⚠️ No results found for your query.")
    
    # Database Information
    st.markdown("### 📊 Database Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                st.markdown("""
                <div class="metric-card">
                    <h3>✅ Database Status</h3>
                    <p>Connected and ready for queries</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>❌ Database Status</h3>
                    <p>Failed to load database</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Database Status</h3>
                <p>Error: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📁 Database Path</h3>
            <p>{DB_FAISS_PATH}</p>
        </div>
        """, unsafe_allow_html=True)

# -------- Analytics Page --------
def analytics_page(translations):
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 {translations['analytics']}</h1>
        <p>Advanced analytics and insights about insurance policy performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    df = generate_bajaj_policy_data()
    
    # Key Performance Indicators
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Growth rate
        current_year = datetime.now().year
        current_policies = df[df['Year'] == current_year]['Total_Policies'].sum()
        previous_policies = df[df['Year'] == current_year - 1]['Total_Policies'].sum()
        growth_rate = ((current_policies - previous_policies) / previous_policies) * 100
        
        st.metric(
            label="Policy Growth Rate",
            value=f"{growth_rate:.1f}%",
            delta=f"{growth_rate:.1f}%"
        )
    
    with col2:
        # Average premium
        avg_premium = df['Total_Premium'].mean()
        st.metric(
            label="Average Premium",
            value=f"₹{avg_premium:,.0f}",
            delta="₹5,000"
        )
    
    with col3:
        # Claims approval rate
        total_claims = df['Claims_Approved'].sum()
        total_policies = df['Total_Policies'].sum()
        approval_rate = (total_claims / total_policies) * 100
        
        st.metric(
            label="Claims Approval Rate",
            value=f"{approval_rate:.1f}%",
            delta="2.5%"
        )
    
    # Advanced Charts
    st.markdown("### 📊 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal analysis
        monthly_avg = df.groupby('Month')['Total_Policies'].mean().reset_index()
        fig_seasonal = px.bar(monthly_avg, x='Month', y='Total_Policies',
                              title='Seasonal Policy Trends',
                              color='Total_Policies')
        fig_seasonal.update_layout(height=400)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        # Year-over-year comparison
        yearly_comparison = df.groupby('Year').agg({
            'Total_Policies': 'sum',
            'Total_Premium': 'sum',
            'Claims_Approved': 'sum'
        }).reset_index()
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Scatter(x=yearly_comparison['Year'], 
                                           y=yearly_comparison['Total_Policies'],
                                           name='Policies', mode='lines+markers'))
        fig_comparison.add_trace(go.Scatter(x=yearly_comparison['Year'], 
                                           y=yearly_comparison['Claims_Approved'],
                                           name='Claims', mode='lines+markers'))
        fig_comparison.update_layout(title='Policies vs Claims Over Time', height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)

# -------- Sidebar Navigation --------
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🛡️ Bajaj Insurance AI</h2>
            <p>Your Smart Policy Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            ["🔍 Vector Explorer", "🏠 Dashboard", "💬 Chat Bot", "📊 Analytics"]
        )
        
        st.markdown("---")
        
        # Language Selection
        st.markdown("### 🌐 Language")
        selected_language = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            index=0
        )
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Error")
        except:
            st.warning("⚠️ Database Loading...")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        temperature = st.slider("AI Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 1000, 512, 50)
        
        return page, selected_language, temperature, max_tokens

# -------- Main App --------
def main():
    # Sidebar navigation
    page, selected_language, temperature, max_tokens = sidebar()
    
    # Get translations for selected language
    translations = LANGUAGES[selected_language]
    
    # Page routing
    if page == "🔍 Vector Explorer":
        vector_explorer_page(translations)
    elif page == "🏠 Dashboard":
        dashboard_page(translations)
    elif page == "💬 Chat Bot":
        chat_bot_page(translations)
    elif page == "📊 Analytics":
        analytics_page(translations)

if __name__ == "__main__":
    main()
