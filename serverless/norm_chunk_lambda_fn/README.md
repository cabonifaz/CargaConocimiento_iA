# Lambda: Markdown Normalization & Chunking

Lambda function for normalizing markdown pages and creating semantic chunks with page tracking.

## Overview

This Lambda replicates the exact behavior of the CLI command `ocr-upsert-company-files`, receiving pages as an array and calculating `page_start`/`page_end` for each chunk.

## Invocation

n8n invokes this Lambda directly using AWS credentials (no API Gateway).

**Event structure**:
```json
{
  "pages": [
    "# Page 1 Title\n\nContent of page 1...",
    "# Page 2 Title\n\nContent of page 2...",
    "# Page 3 Title\n\nContent of page 3..."
  ],
  "filename": "document.pdf",
  "target_tokens": 400,
  "min_tokens": 50,
  "enable_quality_filter": true
}
```

**Response** (array of chunks):
```json
[
  {
    "text": "# Title\n\nContent...",
    "token_count": 152,
    "chunk_id": "a3f2e1d8...",
    "type": "content",
    "filename": "document.pdf",
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 580,
    "page_start": 1,
    "page_end": 2
  },
  {
    "text": "# Title\n## Section\n...",
    "token_count": 98,
    "chunk_id": "b4g3f2e1...",
    "type": "content",
    "filename": "document.pdf",
    "chunk_index": 1,
    "char_start": 435,
    "char_end": 820,
    "page_start": 2,
    "page_end": 3
  }
]
```

## Deployment

### 1. Package the Lambda

```bash
cd serverless/norm_chunk_lambda_fn
zip -r function.zip lambda_function.py normalizer.py chunker.py quality.py
```

### 2. Create Lambda Function

```bash
aws lambda create-function \
  --function-name norm-chunk-processor \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 60 \
  --memory-size 512
```

### 3. Update Lambda (when needed)

```bash
zip -r function.zip lambda_function.py normalizer.py chunker.py quality.py
aws lambda update-function-code \
  --function-name norm-chunk-processor \
  --zip-file fileb://function.zip
```

## n8n Workflow Configuration

### 1. Mistral OCR Node (Extract & Split Pages)

The Mistral OCR node must output pages as an array:

```json
{
  "pages": [
    "# Page 1 content...",
    "# Page 2 content...",
    "# Page 3 content..."
  ],
  "filename": "document.pdf"
}
```

**Important**: Configure Mistral OCR to return **individual pages** as an array, not combined text.

### 2. AWS Lambda Node

**Credentials**: AWS access keys with `lambda:InvokeFunction` permission

**Function**: `norm-chunk-processor`

**Payload**:
```json
{
  "pages": "{{ $json.pages }}",
  "filename": "{{ $json.filename }}"
}
```

The Lambda returns an array of chunks. n8n automatically splits this into individual items.

### 3. Loop Over Items Node

After the Lambda node, the chunks are already individual items in n8n. Use **Loop Over Items** to process each chunk.

### 4. Amazon Bedrock Embeddings Node

For each chunk item:

**Model**: `cohere.embed-multilingual-v3`

**Input Text**: `{{ $json.text }}`

**Input Type**: `search_document`

This generates embeddings for the chunk text.

### 5. Weaviate Insert Node

Combine chunk metadata with embeddings:

**Class**: `Documents`

**ID**: `{{ $json.chunk_id }}`

**Properties**:
```json
{
  "text": "{{ $json.text }}",
  "filename": "{{ $json.filename }}",
  "chunk_index": {{ $json.chunk_index }},
  "token_count": {{ $json.token_count }},
  "chunk_type": "{{ $json.type }}",
  "char_start": {{ $json.char_start }},
  "char_end": {{ $json.char_end }},
  "page_start": {{ $json.page_start }},
  "page_end": {{ $json.page_end }}
}
```

**Vector**: `{{ $json.embedding }}`

Note: The `embedding` field comes from the Bedrock node output. The `page_start` and `page_end` fields indicate which pages the chunk spans.

## Processing Flow

```
Mistral OCR (pages array) → Lambda (normalize + chunk) → Bedrock Embeddings → Weaviate
```

The Lambda receives pages as an array (like the CLI), joins them with offsets, and calculates `page_start`/`page_end` for each chunk.

## Features

- Markdown normalization (OCR-specific)
- Semantic chunking with 15% overlap
- Table-to-JSON conversion
- Quality filtering (configurable)
- Hierarchy preservation (H1-H6)
- Page tracking (page_start, page_end)
- CloudWatch logging

## Configuration

Default parameters (can be overridden in event):
- `target_tokens`: 400
- `min_tokens`: 50
- `enable_quality_filter`: true
- `quality_min_tokens`: 50
- `quality_min_chars`: 200
- `quality_min_alpha_ratio`: 0.30
- `quality_min_unique_ratio`: 0.10

## Logs

View logs in CloudWatch:
```bash
aws logs tail /aws/lambda/norm-chunk-processor --follow
```

## Key Differences from n8n Node Code

This Lambda replicates the **exact CLI behavior**:
- Receives **pages array** (not combined text)
- Joins pages with separator `\n\n---PÁGINA---\n\n`
- Tracks page offsets
- Calculates `page_start` and `page_end` for each chunk using `bisect`
- Identical normalization and chunking logic
