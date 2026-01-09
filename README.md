
# 🧠 Carga de Conocimiento

Esta aplicación procesa documentos PDF, los convierte en vectores y los sube a una base de datos vectorial.

---

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/tuusuario/nombre-del-repo.git
cd nombre-del-repo
````

### 2. Crea y activa el entorno virtual

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar el proyecto

```bash
python app/main.py
```

---

## CLI 

La CLI expone 7 comandos para procesar documentos PDF, desde extracción básica hasta carga completa en base de datos vectorial.

**Requisitos previos**:
- Activar venv e instalar dependencias
- Colocar PDFs en `company_files/` (o subcarpetas para empresas)
- Configurar variables AWS en `.env` y `settings.py` para embeddings y Weaviate

---

### 1) `extract` — Extracción de texto (debug)

Extrae texto **sin normalizar** de un PDF específico. Útil para inspección rápida del contenido extraído.

```bash
python -m app.main extract <ruta_o_nombre.pdf> [--max-pages N]
```

**Parámetros:**
* `ruta_o_nombre.pdf`: archivo dentro de `company_files/` o ruta relativa
* `--max-pages`: límite de páginas a procesar (opcional)

**Ejemplo:**
```bash
python -m app.main extract documento.pdf --max-pages 5
```

**Salida:** Muestra el texto extraído de cada página sin procesamiento adicional.

---

### 2) `chunk-global` — Chunking de un PDF

Ejecuta el pipeline completo de **extracción → normalización → chunking global** sobre un archivo.

```bash
python -m app.main chunk-global <ruta_o_nombre.pdf> \
  [--max-pages N] \
  [--target 512] [--overlap 64] [--min-toks 50] \
  [--sep "\n\n\f\n\n"]
```

**Parámetros de chunking:**
* `--target`: tokens por chunk (recomendado 512–768)
* `--overlap`: tokens de solapamiento entre chunks (≈10–15% del target)
* `--min-toks`: umbral mínimo de tokens por chunk
* `--sep`: separador entre páginas en el texto unido

**Salida:** Información detallada de cada chunk generado con offsets de páginas y caracteres.

---

### 3) `chunk-global-all` — Chunking masivo con reporte

Procesa **todos los PDFs** de `company_files/` con validaciones de calidad y genera reporte JSONL.

```bash
python -m app.main chunk-global-all \
  [--max-pages N] [--non-recursive] \
  [--target 512] [--overlap 64] [--min-toks 50] [--sep "\n\n\f\n\n"] \
  [--q-min-toks 50] [--q-min-chars 200] [--q-alpha-min 0.30] [--q-uniq-min 0.10] \
  [--no-report] [--dry-run]
```

**Validaciones de calidad (Quality Gates):**
* `--q-min-toks`: mínimo de tokens por chunk aceptado
* `--q-min-chars`: mínimo de caracteres por chunk  
* `--q-alpha-min`: ratio mínimo de caracteres alfabéticos (0–1)
* `--q-uniq-min`: diversidad léxica mínima (palabras únicas / total)

**Opciones adicionales:**
* `--non-recursive`: no buscar en subdirectorios
* `--no-report`: no generar archivo JSONL
* `--dry-run`: solo validar, no procesar realmente

**Reporte:** Genera `data/output/chunk-global-report-YYYYMMDD-HHMMSS.jsonl` con métricas detalladas.

---

### 4) `embed-dry` — Embeddings sin almacenar

Ejecuta **extracción → normalización → chunking → embeddings** sobre un archivo **sin guardar en base de datos**.

```bash
python -m app.main embed-dry <ruta_o_nombre.pdf> \
  [--max-pages N] \
  [--target 512] [--overlap 64] [--min-toks 50] \
  [--sep "\n\n\f\n\n"] \
  [--q-min-toks 50] [--q-min-chars 200] [--q-alpha-min 0.30] [--q-uniq-min 0.10]
```

**Propósito:** Testear embeddings y validar configuración antes de subir a Weaviate.

**Salida:** Información de chunks generados y dimensión de vectores obtenidos.

---

### 5) `embed-weaviate` — Carga individual a Weaviate

Procesa **un PDF individual** y lo carga en Weaviate con embeddings de AWS Bedrock.

```bash
python -m app.main embed-weaviate <ruta_o_nombre.pdf> \
  [--max-pages N] \
  [--target 512] [--overlap 64] [--min-toks 50] \
  [--sep "\n\n\f\n\n"] \
  [--q-min-toks 50] [--q-min-chars 200] [--q-alpha-min 0.30] [--q-uniq-min 0.10] \
  [--doc-id ID_PERSONALIZADO] [--company-id EMPRESA]
```

**Parámetros específicos:**
* `--doc-id`: ID personalizado del documento (por defecto: nombre del archivo)
* `--company-id`: identificador de empresa (por defecto: "default_company")

**Funcionalidad:**
* Genera embeddings con AWS Bedrock Titan
* Almacena en colección configurada en `WEAVIATE_COLLECTION`
* Soporte para upsert (actualiza si ya existe)

---

### 6) `embed-weaviate-company` — Carga masiva por empresa

Procesa **todos los PDFs de una carpeta de empresa** y los carga en una colección específica de Weaviate.

```bash
python -m app.main embed-weaviate-company <nombre_empresa> \
  [--max-pages N] \
  [--target 512] [--overlap 64] [--min-toks 50] \
  [--sep "\n\n\f\n\n"] \
  [--q-min-toks 50] [--q-min-chars 200] [--q-alpha-min 0.30] [--q-uniq-min 0.10]
```

**Comportamiento:**
* **Carpeta origen:** `company_files/<nombre_empresa>/`
* **Colección destino:** `<nombre_empresa>` (mismo nombre)
* **Company ID:** `<nombre_empresa>`
* **Procesamiento:** Todos los PDFs de la carpeta

**Ejemplo:**
```bash
python -m app.main embed-weaviate-company Testv2 --max-pages 200 --target 512
```

**Ventajas:**
* Organización automática por empresa
* Colecciones separadas en Weaviate
* Procesamiento masivo eficiente
* Soporte para upsert al re-ejecutar

---

### 7) `bedrock-check` — Verificación de conectividad

Prueba la conexión con AWS Bedrock embebiendo un texto de prueba.

```bash
python -m app.main bedrock-check --text "Texto de prueba para embeddings"
```

**Propósito:** Validar configuración de AWS y conectividad con Bedrock antes de procesamiento masivo.

**Salida:** Dimensión del vector y primeros valores del embedding generado.

---

## Flujo de trabajo recomendado

1. **Verificar conectividad:** `bedrock-check`
2. **Prueba individual:** `extract` → `chunk-global` → `embed-dry`
3. **Carga individual:** `embed-weaviate`
4. **Carga masiva por empresa:** `embed-weaviate-company`
5. **Procesamiento completo:** `chunk-global-all` para métricas

## Configuración

**Variables de entorno clave (.env):**
```
AWS_PROFILE=tu-perfil
BEDROCK_REGION=us-east-2
BEDROCK_MODEL_ID=amazon.titan-embed-text-v2:0
WEAVIATE_URL=https://tu-cluster.weaviate.network
WEAVIATE_API_KEY=tu-api-key
WEAVIATE_COLLECTION=CloudPointComputing
```