# 📄 Asistente de Análisis de PDFs con IA

Este proyecto es una aplicación interactiva que permite **subir un archivo PDF** y **realizar preguntas sobre su contenido** utilizando **GPT-4o** de OpenAI y **LangChain**. Además, si el PDF contiene imágenes, la aplicación puede **extraerlas y permitir preguntas específicas sobre ellas**.

## 🚀 Características

- 📄 **Carga de PDF**: Procesa el contenido del documento y lo divide en fragmentos optimizados para su análisis.
- 🤖 **Chat Inteligente**: Responde preguntas basadas en el contenido del PDF.
- 🖼️ **Análisis de Imágenes**: Extrae imágenes del documento y permite hacer preguntas sobre ellas.
- 🔎 **Búsqueda eficiente**: Usa **embeddings** y **FAISS** para mejorar la precisión de las respuestas.

---

## 📦 Instalación

1️⃣ **Clona el repositorio**
```sh
git clone https://github.com/Celia-code13/PDF_chat_app.git
cd PDF_chat_app
```

2️⃣ **Instala las dependencias necesarias**
```sh
pip install -r requirements.txt
```

3️⃣ **Configura tu API Key de OpenAI**

***¡OJO! Debes incluir tu API Key de OpenAI en el **archivo .env.example** para poder ejecutar la app.***
```sh
OPENAI_API_KEY=tu_clave_de_api_aquí
```

4️⃣ **Ejecuta la aplicación**
```sh
streamlit run app.py
```
