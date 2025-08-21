
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

La CLI expone 3 comandos pensados para un flujo simple por archivo y para todos los PDFs de `company_files/`.

**Requisitos previos**:
- Activar venv e instalar dependencias
- Colocar PDFs en `company_files/`
- (Opcional) Ajustar variables en `.env` y `settings.py`

### 1) `extract` — extracción (debug puntual)

Extrae texto **sin normalizar** para un PDF específico. Útil para inspección rápida si la extracción salió bien.

```bash
python -m app.main extract <ruta_o_nombre.pdf> [--max-pages N]
````

* La `ruta` puede ser:

  * solo el nombre dentro de `company_files/`
  * con el prefijo `company_files/`
  * una ruta **absoluta** dentro de esa carpeta
* Ejemplo (Windows):

```bash
python -m app.main extract 8a49ffa4..._original.pdf --max-pages 3
```

### 2) `chunk-global` — pipeline para **un** PDF

Ejecuta **extracción → normalización → chunking global con offsets** sobre un archivo.

```bash
python -m app.main chunk-global <ruta_o_nombre.pdf> \
  [--max-pages N] \
  [--target 512] [--overlap 64] [--min-toks 50] \
  [--sep "\n\n\f\n\n"]
```

* `--target`: tokens por chunk (recomendado 512–768)
* `--overlap`: tokens de solapamiento (≈10–15% del target)
* `--min-toks`: umbral mínimo de tokens por chunk
* `--sep`: separador entre páginas en el texto unido (no suele cambiarse)

**Salida (consola):**

* Páginas detectadas y total de chunks
* Por cada chunk: `pages=start-end`, `toks`, `chars=start-end`, `chunk_id` y un preview del texto

### 3) `chunk-global-all` — pipeline para **todos** los PDFs

Procesa todos los PDFs de `company_files/` (recursivo por defecto), aplica validaciones de calidad y genera **reporte JSONL**.

```bash
python -m app.main chunk-global-all \
  [--max-pages N] [--non-recursive] \
  [--target 512] [--overlap 64] [--min-toks 50] [--sep "\n\n\f\n\n"] \
  [--q-min-toks 50] [--q-min-chars 200] [--q-alpha-min 0.30] [--q-uniq-min 0.10] \
  [--no-report] [--dry-run]
```

* **Quality gates:**

  * `--q-min-toks`: mínimo de tokens por chunk aceptado
  * `--q-min-chars`: mínimo de caracteres por chunk
  * `--q-alpha-min`: razón mínima de caracteres alfabéticos (0–1)
  * `--q-uniq-min`: diversidad léxica mínima (únicas / total palabras)
* **Reporte**:

  * Por defecto genera `data/output/chunk-global-report-YYYYMMDD-HHMMSS.jsonl`
  * Cada línea es un JSON con métricas por archivo: `pages`, `chunks_total`, `chunks_accepted`, `avg_tokens`, `min_tokens`, `max_tokens`, `warnings`…

**Ejemplos:**

```bash
# Procesar todos los PDFs con parámetros recomendados
python -m app.main chunk-global-all --max-pages 200

# Ajustar validaciones de calidad
python -m app.main chunk-global-all --q-alpha-min 0.25 --q-uniq-min 0.08

# Sin reporte, solo consola
python -m app.main chunk-global-all --no-report
```