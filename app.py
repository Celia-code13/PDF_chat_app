# Importaciones necesarias para la ejecución de la app
import streamlit as st
import fitz 
import openai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
import io
import base64
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Cargar variables de entorno (API Key de OpenAI)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()  

# Extracción de texto y detección de imágenes en el PDF // Se incluye manejo de errores en el procesamiento del pdf
def process_pdf(pdf_bytes):
    text = ""
    images_detected = False

    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
                if page.get_images(full=True):
                    images_detected = True
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
        st.stop()
    
    return text, images_detected

# Fragmentación del texto para facilitar el trabajo con embeddings
def splitting(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

# Base de datos de embeddings mediante FAISS
def vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Desarrollo del chatbot utilizando LangChain + OpenAI (gpt-4o)
def create_chatbot(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    custom_prompt = PromptTemplate(
        template="""
        Eres un asistente que responde preguntas basándote en el contexto proporcionado.
        - Si la respuesta es explícita en el fragmento, respóndela directamente.
        - Si la pregunta requiere inferencia, analiza el tono, la intención y el contenido para dar una respuesta razonada.
        - Si la información no está en el documento, indica que no puedes responder con certeza.

        CONTEXTO:
        {context}

        PREGUNTA:
        {question}

        RESPUESTA:
        """,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

# Extracción de imágenes si el usuario lo solicita
def extract_images(pdf_bytes):
    images = []
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        for page in doc:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
    return images

# Función para analizar imágenes y responder preguntas sobre ellas (GPT-4o)
def analyze_images(image, user_question):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en Biología. "
                    "Responde con precisión la pregunta del usuario basándote en la imagen proporcionada. "
                    "Si la imagen no contiene suficiente información para responder, di que no puedes garantizar la respuesta."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Pregunta: {user_question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content


## INTERFAZ STREAMLIT ##

# Función para cargar animaciones 
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None


lottie_animation = load_lottie_url("https://lottie.host/39ac68dc-09b2-40ed-b7e0-6e4dcc8f9b63/6iOmblej1a.json")
st.set_page_config(page_title="Chat con tu PDF", page_icon="📄")
col1, col2 = st.columns([2, 8])  

with col1:
    if lottie_animation:
        st_lottie(lottie_animation, width=120, height=120, key="lottie_header")
    

with col2:
    
    st.markdown("""
<div style="display: flex; align-items: center;">
    <h1 style="margin: 0;">Asistente de análisis de archivos <span style="color: #CC0000;">PDF</span> basado en IA</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="color:#ADD8E6; font-size:18px; font-weight:normal; margin-top:15px; margin-bottom:30px;">
📄 Sube un archivo <b>PDF</b> y obtén respuestas precisas basadas en su contenido. <br>
🤖 Extrae y procesa las imágenes contenidas en el documento. ¡También puedes preguntarle a nuestra IA sobre ellas!
</div>
""", unsafe_allow_html=True)


# Subida del PDF
uploaded_pdf = st.file_uploader("Puedes subir tu archivo PDF aquí:", type="pdf")

if uploaded_pdf is not None:
    if "pdf_bytes" not in st.session_state:
        st.session_state["pdf_bytes"] = uploaded_pdf.read()

    pdf_bytes = st.session_state["pdf_bytes"]

    if not pdf_bytes or len(pdf_bytes) < 100:
        st.error("❌ El archivo PDF está vacío o no se ha cargado correctamente. Por favor, sube un archivo válido.")
        st.stop()

    with st.spinner("Procesando archivo..."):
        text, images_detected = process_pdf(pdf_bytes)
        chunks = splitting(text)
        vectorstore = vector_store(chunks)
        qa_chain = create_chatbot(vectorstore)

    st.success("¡PDF procesado con éxito!")

    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Preguntas sobre el texto
    st.header("📜 Pregunta cualquier información de interés sobre el contenido de tu PDF:")
    question = st.text_input("🔍 Quiero saber...")
    if st.button("Preguntar"):
        with st.spinner("Generando respuesta..."):
            response = qa_chain.invoke({"query": question})
            st.write("**Respuesta:**", response["result"])

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    # Manejo de imágenes
    if images_detected:
        if st.button("📷 El PDF contiene imágenes. ¿Quieres extraerlas?"):
            with st.spinner("Extrayendo imágenes..."):
                st.session_state["extracted_images"] = extract_images(pdf_bytes)
            st.success("¡Imágenes extraídas correctamente!")

    if "extracted_images" in st.session_state and st.session_state["extracted_images"]:
        st.header("🖼️ Análisis de Imágenes:")
        selected_image = st.selectbox("Selecciona una imagen para visualizar:", list(range(len(st.session_state["extracted_images"]))))
        st.image(st.session_state["extracted_images"][selected_image], caption="Imagen seleccionada", use_container_width=True)

        image_question = st.text_input("🔎 Puedes hacer preguntas sobre las imágenes extraídas ¿Qué quieres saber sobre esta imagen?")
        if st.button(" Preguntar "):
            with st.spinner("Generando respuesta..."):
                response = analyze_images(st.session_state["extracted_images"][selected_image], image_question)
                st.write("**Respuesta:**", response)     