import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import torch
import re
import time
from datetime import datetime
import unicodedata

# -------------------------------
# 🔧 Configuración Inicial Mejorada
# -------------------------------
st.set_page_config(
    page_title="Homologador Inteligente RAMEDICAS",
    page_icon="⚕️",  # Icono más relacionado con salud
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 📦 Carga de Modelo y Datos Optimizada
# -------------------------------
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def load_model():
    """Carga el modelo SentenceTransformer con manejo de errores."""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Modelo multilingüe mejorado
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner="Procesando base de datos de productos...")
def load_data():
    """Carga los datos con múltiples fuentes posibles y validación."""
    DATA_SOURCES = [
        "https://docs.google.com/spreadsheets/d/1Y9SgliayP_J5Vi2SdtZmGxKwf1iY7ma/export?format=xlsx",
        "data/local_backup.xlsx"  # Fuente alternativa local
    ]
    
    for source in DATA_SOURCES:
        try:
            if source.startswith('http'):
                df = pd.read_excel(source)
            else:
                df = pd.read_excel(source)
            
            # Validación de columnas esenciales
            required_columns = {'nomart', 'codart'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Faltan columnas requeridas: {required_columns - set(df.columns)}")
                
            return df
        except Exception as e:
            st.warning(f"No se pudo cargar desde {source}: {str(e)}")
            continue
    
    st.error("No se pudo cargar la base de datos desde ninguna fuente")
    st.stop()

@st.cache_resource
def compute_embeddings(_model, df):
    """Calcula embeddings con normalización y manejo de batch."""
    processed_names = df['nomart'].apply(preprocess_name).tolist()
    # Procesamiento por lotes para grandes volúmenes de datos
    batch_size = 128
    embeddings = []
    for i in range(0, len(processed_names), batch_size):
        batch = processed_names[i:i + batch_size]
        embeddings.append(_model.encode(batch, convert_to_tensor=True))
    return torch.cat(embeddings)

# -------------------------------
# 🛠️ Funciones Auxiliares Mejoradas
# -------------------------------
def preprocess_name(name: str) -> str:
    """Preprocesamiento avanzado de texto con normalización Unicode."""
    if not isinstance(name, str):
        name = str(name)
    
    # Normalización Unicode (elimina acentos y caracteres especiales)
    name = unicodedata.normalize('NFKD', name.lower()).encode('ASCII', 'ignore').decode('ASCII')
    
    # Lista de reemplazos más completa
    replacements = [
        (r'[\/\\]', ' '),  # Barras
        (r'[\+\-\*]', ' '),  # Operadores matemáticos
        (r'[\(\[].*?[\)\]]', ''),  # Elimina contenido entre paréntesis/corchetes
        (r'[^a-zA-Z0-9\s]', ''),  # Elimina caracteres especiales
        (r'\b\d+[a-zA-Z]*\b', ''),  # Elimina códigos sueltos
        (r'\s+', ' '),  # Espacios múltiples a uno
    ]
    
    for pattern, repl in replacements:
        name = re.sub(pattern, repl, name)
    
    # Eliminación de stopwords específicas del dominio
    stopwords = {'de', 'la', 'el', 'en', 'y', 'para', 'con', 'sin', 'x'}
    words = [word for word in name.split() if word not in stopwords and len(word) > 2]
    
    return ' '.join(words).strip()

def find_matches(query: str, df: pd.DataFrame, embeddings, model, top_k=5, threshold=0.65):
    """Búsqueda mejorada con múltiples coincidencias y filtrado inteligente."""
    try:
        processed_query = preprocess_name(query)
        query_embedding = model.encode(processed_query, convert_to_tensor=True)
        
        # Cálculo de similitud con GPU si está disponible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scores = util.pytorch_cos_sim(query_embedding.to(device), embeddings.to(device))[0]
        
        # Obtener top_k coincidencias
        top_indices = torch.topk(scores, k=min(top_k, len(df))).indices.cpu().numpy()
        top_scores = torch.topk(scores, k=min(top_k, len(df))).values.cpu().numpy()
        
        # Filtrar y formatear resultados
        results = []
        for idx, score in zip(top_indices, top_scores):
            if score >= threshold:
                result = {
                    'Consulta': query,
                    'Producto RAMEDICAS': df.iloc[idx]['nomart'],
                    'Código': df.iloc[idx]['codart'],
                    'Similitud': f"{score:.1%}",
                    'Puntuación': float(score)  # Para ordenamiento
                }
                results.append(result)
        
        return results if results else [{
            'Consulta': query,
            'Producto RAMEDICAS': "NO ENCONTRADO",
            'Código': "N/A",
            'Similitud': "0%",
            'Puntuación': 0.0
        }]
    except Exception as e:
        st.error(f"Error procesando '{query}': {str(e)}")
        return [{
            'Consulta': query,
            'Producto RAMEDICAS': "ERROR",
            'Código': "N/A",
            'Similitud': "0%",
            'Puntuación': 0.0
        }]

def to_excel(df):
    """Genera archivo Excel con formato profesional."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
        
        # Formateo del Excel
        workbook = writer.book
        worksheet = writer.sheets['Resultados']
        
        # Formato para encabezados
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4F81BD',
            'font_color': 'white',
            'border': 1
        })
        
        # Aplicar formatos
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Autoajustar columnas
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(max_len, 50))
    
    output.seek(0)
    return output

# -------------------------------
# 🎨 Interfaz de Usuario Mejorada
# -------------------------------
def show_sidebar():
    """Barra lateral con configuración y estadísticas."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=RAMEDICAS", width=150)
        st.title("Configuración")
        
        # Controles de umbral y resultados
        threshold = st.slider("Umbral de similitud", 0.5, 1.0, 0.7, 0.05)
        max_results = st.slider("Máximo de resultados por consulta", 1, 10, 3)
        
        # Estadísticas
        st.markdown("---")
        st.subheader("Estadísticas")
        if 'df' in st.session_state:
            st.write(f"📊 Productos en base: {len(st.session_state.df):,}")
        if 'last_search' in st.session_state:
            st.write(f"⏱ Última búsqueda: {st.session_state.last_search}")
        
        # Información de la aplicación
        st.markdown("---")
        st.markdown("""
        **🔍 Homologador Inteligente v2.0**  
        Usa IA para encontrar equivalencias de productos.  
        Desarrollado por RAMEDICAS S.A.S.  
        """)
        
        return threshold, max_results

def main():
    """Función principal de la aplicación."""
    # Cargar datos y modelo
    with st.spinner("Inicializando sistema..."):
        model = load_model()
        df = load_data()
        embeddings = compute_embeddings(model, df)
        
        # Guardar en sesión para acceso rápido
        st.session_state.df = df
        st.session_state.embeddings = embeddings
        st.session_state.model = model
    
    # Sidebar
    threshold, max_results = show_sidebar()
    
    # Cabecera mejorada
    st.image("https://via.placeholder.com/800x150?text=RAMEDICAS+HOMOLOGADOR", use_column_width=True)
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    .result-box { border-left: 5px solid #4F81BD; padding: 10px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    # Pestañas para diferentes modos de entrada
    tab1, tab2, tab3 = st.tabs(["📝 Entrada Manual", "📄 Subir Archivo", "⚙️ Búsqueda Avanzada"])
    
    with tab1:
        st.subheader("Entrada Manual")
        input_text = st.text_area("Ingrese nombres de productos (uno por línea):", height=150)
        process_btn = st.button("Procesar Entrada Manual", type="primary")
        
        if process_btn and input_text:
            process_queries(input_text.split('\n'), threshold, max_results)
    
    with tab2:
        st.subheader("Subir Archivo Excel o CSV")
        uploaded_file = st.file_uploader("Seleccione archivo", type=['xlsx', 'csv'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                # Mostrar vista previa
                st.write("Vista previa (primeras 5 filas):")
                st.dataframe(upload_df.head())
                
                # Seleccionar columna a procesar
                col_to_process = st.selectbox("Seleccione columna a homologar", upload_df.columns)
                
                if st.button("Procesar Archivo", type="primary"):
                    queries = upload_df[col_to_process].astype(str).tolist()
                    process_queries(queries, threshold, max_results, upload_df)
            except Exception as e:
                st.error(f"Error procesando archivo: {str(e)}")
    
    with tab3:
        st.subheader("Búsqueda Avanzada")
        advanced_query = st.text_input("Consulta avanzada:")
        if st.button("Buscar", key="adv_search"):
            if advanced_query:
                results = find_matches(advanced_query, df, embeddings, model, 10, threshold)
                display_results(pd.DataFrame(results))
            else:
                st.warning("Ingrese un término de búsqueda")

def process_queries(queries, threshold, max_results, context_df=None):
    """Procesa múltiples consultas y muestra resultados."""
    start_time = time.time()
    queries = [q.strip() for q in queries if q.strip()]
    
    if not queries:
        st.warning("No hay consultas válidas para procesar")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    # Procesamiento con barra de progreso
    for i, query in enumerate(queries):
        status_text.text(f"Procesando {i+1}/{len(queries)}: {query[:50]}...")
        progress_bar.progress((i + 1) / len(queries))
        
        matches = find_matches(query, st.session_state.df, st.session_state.embeddings, 
                             st.session_state.model, max_results, threshold)
        results.extend(matches)
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    # Ordenar por puntuación descendente
    results_df = results_df.sort_values('Puntuación', ascending=False)
    
    # Mostrar estadísticas
    elapsed_time = time.time() - start_time
    found_count = len(results_df[results_df['Puntuación'] > threshold])
    
    st.success(f"""
    ✅ Procesamiento completado:  
    - Consultas procesadas: {len(queries)}  
    - Coincidencias encontradas: {found_count} ({found_count/len(queries):.0%})  
    - Tiempo total: {elapsed_time:.2f} segundos  
    - Tiempo promedio por consulta: {elapsed_time/len(queries):.2f} segundos  
    """)
    
    # Mostrar resultados
    display_results(results_df)
    
    # Guardar timestamp de última búsqueda
    st.session_state.last_search = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def display_results(results_df):
    """Muestra resultados con formato profesional."""
    # Mostrar tabla resumen
    st.subheader("Resultados de Homologación")
    st.dataframe(results_df.style.highlight_max(subset=['Puntuación'], color='lightgreen'), 
                use_container_width=True)
    
    # Mostrar detalles de los mejores resultados
    best_match = results_df.iloc[0]
    if best_match['Puntuación'] > 0:
        st.markdown(f"""
        <div class="result-box">
            <h4>Mejor coincidencia encontrada:</h4>
            <p><strong>Consulta:</strong> {best_match['Consulta']}</p>
            <p><strong>Producto equivalente:</strong> {best_match['Producto RAMEDICAS']}</p>
            <p><strong>Código:</strong> {best_match['Código']}</p>
            <p><strong>Nivel de confianza:</strong> {best_match['Similitud']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Botón de descarga
    excel_data = to_excel(results_df)
    st.download_button(
        label="📥 Exportar a Excel",
        data=excel_data,
        file_name=f"homologacion_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descargue los resultados completos en formato Excel"
    )

if __name__ == "__main__":
    main()
