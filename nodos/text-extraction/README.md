# n8n-nodes-mistral-ocr

Nodo de n8n para extraer texto de documentos PDF usando Mistral OCR.

## Descripción

Este nodo permite extraer texto de archivos PDF utilizando la API de Mistral OCR. Soporta múltiples formatos de entrada y salida, haciéndolo flexible para diferentes casos de uso.

## Características

- Extracción de texto de PDFs usando Mistral OCR
- Múltiples tipos de entrada:
  - Datos binarios de nodos anteriores
  - Strings Base64
  - URLs de documentos
- Múltiples formatos de salida:
  - Array de markdown (una página por elemento)
  - Markdown combinado (todas las páginas en un string)
  - Objeto JSON con metadata
- Límite de páginas configurable
- Opción para incluir imágenes en base64

## Instalación

### Opción 1: Instalación en n8n (Community Node)

1. En n8n, ve a **Settings** > **Community Nodes**
2. Busca `n8n-nodes-mistral-ocr`
3. Haz clic en **Install**

### Opción 2: Instalación Manual

1. Navega a la carpeta de n8n:
```bash
cd ~/.n8n/custom
```

2. Clona o copia este nodo:
```bash
# Si tienes el código en un repositorio
git clone <repository-url>

# O copia la carpeta manualmente
cp -r /path/to/nodos ~/.n8n/custom/n8n-nodes-mistral-ocr
```

3. Instala las dependencias:
```bash
cd ~/.n8n/custom/n8n-nodes-mistral-ocr
npm install
npm run build
```

4. Reinicia n8n

## Configuración

### Credenciales

Antes de usar el nodo, necesitas configurar las credenciales de Mistral API:

1. En n8n, ve a **Credentials** > **New**
2. Busca **Mistral OCR API**
3. Ingresa tu API Key de Mistral AI
4. Guarda las credenciales

Puedes obtener tu API Key en: https://console.mistral.ai/

## Uso

### Parámetros

#### Operation
- **Extract Text from PDF**: Extrae texto de un documento PDF

#### Input Type
- **Binary Data**: Usa datos binarios del nodo anterior (ej: HTTP Request, Read Binary File)
- **Base64 String**: Proporciona un string base64 del PDF
- **URL**: Proporciona una URL al documento PDF

#### Binary Property
- Nombre de la propiedad binaria que contiene el PDF (por defecto: `data`)
- Solo visible cuando Input Type es "Binary Data"

#### Base64 Data
- String base64 del PDF
- Solo visible cuando Input Type es "Base64 String"

#### Document URL
- URL del documento PDF
- Solo visible cuando Input Type es "URL"

#### Model
- Modelo de Mistral OCR a usar (por defecto: `mistral-ocr-latest`)

#### Max Pages
- Número máximo de páginas a extraer (0 = todas las páginas)

#### Output Format
- **Markdown Array**: Retorna un array de strings markdown (uno por página)
- **Combined Markdown**: Retorna un solo string markdown con todas las páginas
- **JSON Object**: Retorna un objeto JSON con páginas y metadata

#### Include Images
- Si se deben incluir imágenes en base64 en la respuesta

### Ejemplos

#### Ejemplo 1: Extraer texto de PDF descargado

```
[HTTP Request] → [Mistral OCR] → [Process Data]
```

Configuración del nodo Mistral OCR:
- Input Type: Binary Data
- Binary Property: data
- Output Format: JSON Object

#### Ejemplo 2: Procesar PDF desde URL

```
[Start] → [Mistral OCR] → [Process Data]
```

Configuración del nodo Mistral OCR:
- Input Type: URL
- Document URL: https://example.com/document.pdf
- Output Format: Combined Markdown

#### Ejemplo 3: Extraer solo primeras 5 páginas

```
[Read Binary File] → [Mistral OCR] → [Save to Database]
```

Configuración del nodo Mistral OCR:
- Input Type: Binary Data
- Max Pages: 5
- Output Format: Markdown Array

## Formato de Salida

### JSON Object
```json
{
  "pages": [
    "# Página 1\n\nContenido...",
    "# Página 2\n\nMás contenido..."
  ],
  "pageCount": 10,
  "extractedPages": 2,
  "model": "mistral-ocr-latest"
}
```

### Markdown Array
```json
{
  "pages": [
    "# Página 1\n\nContenido...",
    "# Página 2\n\nMás contenido..."
  ],
  "pageCount": 10,
  "extractedPages": 2
}
```

### Combined Markdown
```json
{
  "markdown": "# Página 1\n\nContenido...\n\n---\n\n# Página 2\n\nMás contenido...",
  "pageCount": 10,
  "extractedPages": 2
}
```

## Desarrollo

### Requisitos
- Node.js 18+
- npm

### Scripts

```bash
# Compilar
npm run build

# Desarrollo (watch mode)
npm run dev

# Formatear código
npm run format

# Linter
npm run lint
```

### Estructura de archivos

```
nodos/
├── MistralOcr.node.ts           # Nodo principal
├── MistralOcrApi.credentials.ts # Credenciales
├── package.json                  # Configuración del paquete
├── tsconfig.json                 # Configuración de TypeScript
└── README.md                     # Documentación
```

## Basado en

Este nodo está basado en el adapter `mistralocr_text_extractor.py` del proyecto original, proporcionando la misma funcionalidad en formato de nodo n8n.

## Licencia

MIT

## Soporte

Para reportar problemas o solicitar nuevas características, por favor abre un issue en el repositorio.
