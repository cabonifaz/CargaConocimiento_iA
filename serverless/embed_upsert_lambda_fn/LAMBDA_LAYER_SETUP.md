# Lambda Layer Setup - Weaviate Client

Guía rápida para crear el Lambda Layer compatible con AWS Lambda.

## El Problema

El error `GLIBC_2.28' not found` ocurre porque `weaviate-client` tiene dependencias con extensiones nativas (Rust) que deben compilarse para el entorno de AWS Lambda (Amazon Linux 2 con GLIBC 2.26).

## La Solución: Usar Docker

Usa la imagen oficial de AWS Lambda para compilar las dependencias correctamente.

## Pasos

### 1. Crear el Layer usando Docker

```bash
cd serverless/embed_upsert_lambda_fn

# Crear directorio
mkdir -p lambda-layer

# Instalar usando imagen de Lambda
docker run --rm \
  --entrypoint "" \
  -v "$PWD/lambda-layer:/layer" \
  -w /layer \
  public.ecr.aws/lambda/python:3.11 \
  pip install weaviate-client==4.16.9 -t python/

# Verificar instalación
ls lambda-layer/python/
# Deberías ver: weaviate/, cryptography/, y otras carpetas
```

### 2. Crear ZIP

```bash
cd lambda-layer
zip -r weaviate-layer.zip python/

# Verificar tamaño (debe ser ~15-20 MB)
ls -lh weaviate-layer.zip
```

### 3. Subir a AWS

```bash
aws lambda publish-layer-version \
  --layer-name weaviate-client-layer \
  --description "Weaviate client v4.16.9 for Python 3.11 (Lambda compatible)" \
  --zip-file fileb://weaviate-layer.zip \
  --compatible-runtimes python3.11 \
  --region us-east-1
```

**Importante**: Guarda el `LayerVersionArn` del output.

### 4. Actualizar Lambda Function

Si ya existe la función Lambda, actualízala para usar el nuevo layer:

```bash
# Eliminar layer anterior (si existe)
aws lambda update-function-configuration \
  --function-name embed-upsert-processor \
  --layers

# Agregar nuevo layer
aws lambda update-function-configuration \
  --function-name embed-upsert-processor \
  --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:weaviate-client-layer:VERSION
```

O desde la consola AWS:
1. Ve a tu función Lambda
2. Scroll down a "Layers"
3. "Add a layer"
4. Selecciona "Custom layers"
5. Selecciona `weaviate-client-layer`
6. Selecciona la versión más reciente
7. "Add"

### 5. Limpiar archivos locales

```bash
cd ..
rm -rf lambda-layer/
```

## Verificar

Prueba la función Lambda desde n8n. El error de GLIBC debe desaparecer.

## Sin Docker?

Si no tienes Docker, usa pip con platform-specific wheels:

```bash
pip install \
  --platform manylinux2014_x86_64 \
  --target=lambda-layer/python \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  --upgrade \
  weaviate-client==4.16.9
```

**Nota**: Este método puede no funcionar perfectamente. Docker es la solución recomendada.

## Recursos

- Imagen Docker oficial: `public.ecr.aws/lambda/python:3.11`
- [AWS Lambda Runtimes](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html)
- [Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)
