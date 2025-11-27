# Lambda: Embeddings & Weaviate Upsert

Lambda function for generating embeddings using AWS Bedrock Cohere Embed Multilingual v3 and upserting chunks to Weaviate.

## Overview

This Lambda function completes the OCR processing pipeline by:
1. Receiving chunks from the `norm_chunk_lambda_fn`
2. Generating embeddings using AWS Bedrock (Cohere Embed Multilingual v3)
3. Upserting chunks with embeddings to Weaviate

This replicates the behavior of the CLI command `ocr-upsert-company-files` for the embedding and Weaviate upsert stages.

## Architecture

```
norm_chunk_lambda_fn → embed_upsert_lambda_fn → Weaviate
        (chunks)              (embed + upsert)
```

## Event Structure

### Input

```json
{
  "chunks": [
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
    }
  ],
  "company_id": "EMPR304",
  "area_id": "AREA1"
}
```

**Required fields:**
- `chunks`: Array of chunk objects from `norm_chunk_lambda_fn` (must contain `filename` field)
- `company_id`: Company identifier (string, no whitespace)
- `area_id`: Area identifier (string, no whitespace)

**Auto-derived fields:**
- `doc_id`: Extracted from `chunks[0].filename` (without .pdf extension)
- `doc_title`: Same as `doc_id`
- `company`: Same as `company_id`
- `area`: Same as `area_id`
- `embedding_model`: From `BEDROCK_MODEL_ID` environment variable (default: `cohere.embed-multilingual-v3`)
- `collection_name`: Always `company_id`

### Output

```json
{
  "chunks_written": 42,
  "collection_name": "304",
  "doc_id": "documento_123",
  "company_id": "304",
  "area_id": "1"
}
```

## Environment Variables

The Lambda function requires these environment variables:

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `WEAVIATE_URL` | Weaviate cluster URL | `https://xxxxx.weaviate.network` | Yes |
| `WEAVIATE_API_KEY` | Weaviate API key | `xxxxx` | Yes |
| `BEDROCK_REGION` | AWS Bedrock region | `us-east-1` | No (defaults to AWS_REGION) |
| `BEDROCK_MODEL_ID` | Bedrock embedding model | `cohere.embed-multilingual-v3` | No (has default) |
| `AWS_REGION` | AWS region (auto-set by Lambda) | `us-east-1` | Auto |

## Configuration

Default values (matching `ocr-upsert-company-files` CLI command):
- **Target tokens per chunk**: 400
- **Min tokens per chunk**: 50
- **Quality filters**:
  - Min tokens: 50
  - Min chars: 200
  - Min alpha ratio: 0.30
  - Min unique ratio: 0.10
- **Embedding model**: `cohere.embed-multilingual-v3`
- **Input type**: `search_document`
- **Distance metric**: Cosine

## Weaviate Schema

The function creates collections with this schema:

### Properties

| Field | Type | Indexed | Filterable | Description |
|-------|------|---------|------------|-------------|
| `text` | TEXT | Yes | No | Full chunk text with hierarchy |
| `bm25_text` | TEXT | Yes | No | Cleaned text for BM25 search |
| `doc_title` | TEXT | Yes | Yes | Document title |
| `section_title` | TEXT | Yes | Yes | Section title (deepest heading) |
| `doc_id` | TEXT | No | Yes | Document identifier |
| `company_id` | TEXT | No | Yes | Company ID (string) |
| `company` | TEXT | No | Yes | Company name |
| `area_id` | TEXT | No | Yes | Area ID (string) |
| `area` | TEXT | No | Yes | Area name |
| `section_path` | TEXT_ARRAY | No | Yes | Hierarchical section path |
| `page_start` | INT | No | Yes | Starting page number |
| `page_end` | INT | No | Yes | Ending page number |
| `embedding_model` | TEXT | No | Yes | Model used for embeddings |
| `embedding_dim` | INT | No | Yes | Embedding vector dimension |
| `chunk_id` | TEXT | No | No | Unique chunk identifier |
| `token_count` | INT | No | No | Token count |
| `char_start` | INT | No | No | Character start position |
| `char_end` | INT | No | No | Character end position |
| `ingested_at` | TEXT | No | No | Ingestion timestamp (ISO format) |

### Vector Configuration
- **Vector type**: Self-provided (BYOV)
- **Index**: HNSW
- **Distance metric**: Cosine
- **BM25 config**: k1=1.3, b=0.75

## Dependencies

The function requires `weaviate-client` which must be packaged in a Lambda Layer.

**Note**: `boto3` is already included in the AWS Lambda runtime, so you don't need to package it.

## Lambda Layer Creation

You need to create a Lambda Layer only for `weaviate-client==4.16.9`.

**⚠️ IMPORTANT**: You must use Docker or platform-specific pip to avoid GLIBC compatibility errors. See `LAMBDA_LAYER_SETUP.md` for troubleshooting.

### Quick Steps (Using Docker - RECOMMENDED)

**Why Docker?** The `weaviate-client` has dependencies with native extensions that must be compiled for AWS Lambda's environment (Amazon Linux 2). Using Docker ensures compatibility.

```bash
# Navigate to the lambda function directory
cd serverless/embed_upsert_lambda_fn

# Create layer directory structure
mkdir -p lambda-layer

# Use Docker to install packages in Lambda-compatible environment
docker run --rm \
  --entrypoint "" \
  -v "$PWD/lambda-layer:/layer" \
  -w /layer \
  public.ecr.aws/lambda/python:3.11 \
  pip install weaviate-client==4.16.9 -t python/

# Create the ZIP file
cd lambda-layer
zip -r weaviate-layer.zip python/

# Upload to AWS Lambda
aws lambda publish-layer-version \
  --layer-name weaviate-client-layer \
  --description "Weaviate client v4.16.9 for Python 3.11 (Lambda compatible)" \
  --zip-file fileb://weaviate-layer.zip \
  --compatible-runtimes python3.11

# Clean up
cd ..
rm -rf lambda-layer/
```

### Alternative: Without Docker (May have compatibility issues)

If you can't use Docker, you can try installing with platform-specific wheels:

```bash
cd serverless/embed_upsert_lambda_fn
mkdir -p lambda-layer/python

# Install for Linux x86_64 (Lambda platform)
pip install \
  --platform manylinux2014_x86_64 \
  --target=lambda-layer/python \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  --upgrade \
  weaviate-client==4.16.9

# Create ZIP and upload (same as above)
cd lambda-layer
zip -r weaviate-layer.zip python/

aws lambda publish-layer-version \
  --layer-name weaviate-client-layer \
  --description "Weaviate client v4.16.9 for Python 3.11" \
  --zip-file fileb://weaviate-layer.zip \
  --compatible-runtimes python3.11
```

**Note**: The platform-specific method may still have issues. Docker method is strongly recommended.

**Important**: Save the Layer ARN from the output. You'll need it when creating the Lambda function.

Example output:
```json
{
    "LayerArn": "arn:aws:lambda:us-east-1:123456789012:layer:weaviate-client-layer",
    "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:weaviate-client-layer:1",
    ...
}
```

Use the `LayerVersionArn` when creating/updating the Lambda function.

### Explanation of Docker Command

```bash
docker run --rm \
  --entrypoint "" \                    # Override default entrypoint to run custom commands
  -v "$PWD/lambda-layer:/layer" \      # Mount local directory to container
  -w /layer \                          # Set working directory
  public.ecr.aws/lambda/python:3.11 \  # Official AWS Lambda Python 3.11 image
  pip install weaviate-client==4.16.9 -t python/
```

This command:
1. **Overrides the entrypoint** (`--entrypoint ""`) to allow running pip directly
2. Uses the **official AWS Lambda Python 3.11 image** (same environment as Lambda)
3. Mounts your local `lambda-layer` directory into the container
4. Installs `weaviate-client` and all dependencies compiled for Lambda's environment
5. Automatically removes the container when done (`--rm`)

The resulting files are compatible with AWS Lambda because they're compiled in the exact same environment.

### Update an Existing Layer

If you need to update the layer version:

```bash
# Rebuild the ZIP with new version
cd lambda-layer
zip -r weaviate-layer.zip python/

# Publish new version (version number auto-increments)
aws lambda publish-layer-version \
  --layer-name weaviate-client-layer \
  --description "Weaviate client v4.16.9 for Python 3.11" \
  --zip-file fileb://weaviate-layer.zip \
  --compatible-runtimes python3.11
```

### Clean Up

After uploading, you can remove the local layer directory:

```bash
cd ..
rm -rf lambda-layer/
```

### Troubleshooting

**GLIBC version error (`GLIBC_2.28' not found`)?**
- This means you installed the layer on your local machine instead of using Docker
- **Solution**: Delete the layer and recreate it using the Docker method above
- The error occurs because `cryptography` (dependency of weaviate-client) has native Rust bindings

**Layer too large?**
- The layer should be ~15-20 MB zipped (weaviate-client + dependencies)
- If much larger, ensure you're only installing weaviate-client (not boto3)

**Import errors in Lambda?**
- Verify layer is attached to the function: Check Lambda Configuration → Layers
- Check the layer is compatible with python3.11
- Ensure the structure is `python/` (not `python/lib/` or `python/site-packages/`)
- Verify you used the Docker method or platform-specific pip install

**Docker not available?**
- Use the platform-specific pip method (shown above)
- Or install Docker Desktop / Docker Engine
- Or use AWS Cloud9 which has Docker pre-installed

**Wrong Python version?**
- Lambda uses Python 3.11
- Make sure the Docker image is `public.ecr.aws/lambda/python:3.11`
- The layer ARN should specify `python3.11` as compatible runtime

## Deployment

### 1. Package the Lambda Function

```bash
cd serverless/embed_upsert_lambda_fn
zip -r function.zip lambda_function.py embedder.py weaviate_client.py bm25_processor.py metadata.py
```

### 2. Create Lambda Function

```bash
aws lambda create-function \
  --function-name embed-upsert-processor \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 300 \
  --memory-size 1024 \
  --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:weaviate-client-layer:VERSION \
  --environment Variables="{WEAVIATE_URL=https://xxxxx.weaviate.network,WEAVIATE_API_KEY=xxxxx,BEDROCK_REGION=us-east-1,BEDROCK_MODEL_ID=cohere.embed-multilingual-v3}"
```

**Important configuration:**
- **Timeout**: 300 seconds (5 minutes) - embeddings can take time for large documents
- **Memory**: 1024 MB - Weaviate client needs memory for batch operations
- **Layers**: Include the Weaviate client layer ARN

### 3. Update Lambda (when needed)

```bash
zip -r function.zip lambda_function.py embedder.py weaviate_client.py bm25_processor.py metadata.py
aws lambda update-function-code \
  --function-name embed-upsert-processor \
  --zip-file fileb://function.zip
```

### 4. Configure IAM Role

The Lambda execution role needs these permissions:

#### Trust Policy (AssumeRole)

Allow Lambda service to assume this role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

#### Permissions Policy

Create a policy with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Sid": "BedrockInvokeModel",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/cohere.embed-multilingual-v3"
      ]
    }
  ]
}
```

**What each permission does:**
- `logs:*`: Write logs to CloudWatch for debugging and monitoring
- `bedrock:InvokeModel`: Call Bedrock to generate embeddings using Cohere model

#### Create Role via AWS CLI

```bash
# 1. Create trust policy file
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 2. Create the role
aws iam create-role \
  --role-name embed-upsert-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# 3. Create permissions policy file
cat > permissions-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Sid": "BedrockInvokeModel",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/cohere.embed-multilingual-v3"
      ]
    }
  ]
}
EOF

# 4. Attach inline policy to role
aws iam put-role-policy \
  --role-name embed-upsert-lambda-role \
  --policy-name embed-upsert-permissions \
  --policy-document file://permissions-policy.json

# 5. Get the role ARN (use this in Lambda creation)
aws iam get-role --role-name embed-upsert-lambda-role --query 'Role.Arn' --output text
```

**Note**: If you need to access Bedrock from a different region or use a different model, update the Resource ARN accordingly.

## n8n Workflow Integration

### Complete Workflow

```
Mistral OCR → norm_chunk_lambda_fn → embed_upsert_lambda_fn → Success
```

### Node Configuration

#### 1. AWS Lambda Node (norm_chunk_lambda_fn)

Already configured in previous step. Outputs chunks array.

#### 2. AWS Lambda Node (embed_upsert_lambda_fn)

**Function**: `embed-upsert-processor`

**Payload:**
```json
{
  "chunks": "{{ $json }}",
  "company_id": "304",
  "area_id": "1"
}
```

**Note**:
- The chunks come directly from the previous `norm_chunk_lambda_fn` step
- `doc_id` is automatically extracted from the `filename` field in chunks
- `collection_name` is automatically set to `company_id`
- `embedding_model` is read from `BEDROCK_MODEL_ID` environment variable

#### 3. Output

The Lambda returns:
```json
{
  "chunks_written": 42,
  "collection_name": "304",
  "doc_id": "documento_123"
}
```

Use this to track processing status or trigger notifications.

## Logs

View logs in CloudWatch:

```bash
aws logs tail /aws/lambda/embed-upsert-processor --follow
```

## Processing Flow

```
1. Receive chunks from norm_chunk_lambda_fn
2. Extract text from each chunk
3. Generate embeddings using Bedrock Cohere
   - Model: cohere.embed-multilingual-v3
   - Input type: search_document
   - Batching: 1 text per API call (Cohere limitation)
4. Process BM25 text for each chunk
5. Create Weaviate metadata with section info
6. Upsert to Weaviate collection
   - Batch size: 100 chunks per batch
   - UUID: Deterministic (doc_id + chunk_id)
   - Fallback: Individual insert/replace on batch failure
```

## Features

- Bedrock Cohere Embed Multilingual v3 integration
- Automatic Weaviate collection creation
- BM25 text optimization for hybrid search
- Section hierarchy extraction
- Deterministic UUID generation
- Batch upsert with fallback
- Comprehensive CloudWatch logging
- Error handling and retry logic

## Error Handling

The function handles common errors:
- **Throttling**: Automatic retry with exponential backoff
- **Batch insert failure**: Falls back to individual inserts
- **Missing environment variables**: Clear error messages
- **Invalid input**: Validation with descriptive errors

## Monitoring

Key CloudWatch metrics to monitor:
- Invocations
- Duration (should be < 300s)
- Errors
- Throttles

Custom log messages:
- "Embedding X chunks..."
- "Generated X embeddings"
- "Upserting X chunks to collection 'Y'"
- "Successfully upserted X chunks"

## Troubleshooting

### Lambda timeout
- Increase timeout (max 900s)
- Reduce chunk batch size in previous step

### Out of memory
- Increase memory (recommended: 1024 MB or higher)

### Bedrock throttling
- Function has built-in retry logic
- Consider requesting higher quota if frequent

### Weaviate connection errors
- Check `WEAVIATE_URL` and `WEAVIATE_API_KEY`
- Verify network connectivity (VPC config if needed)

## Cost Optimization

Approximate costs per document (500 chunks, 400 tokens/chunk):
- Bedrock embeddings: ~$0.02 (200K tokens @ $0.10/1M tokens)
- Lambda execution: ~$0.001 (1 min @ 1024 MB)
- Weaviate: Depends on your plan

Total: ~$0.021 per 500-chunk document

## Differences from CLI

This Lambda replicates the exact behavior of `ocr-upsert-company-files` for embeddings and Weaviate upsert:
- Same embedding model and configuration
- Same Weaviate schema structure
- Same metadata fields
- Same BM25 text processing
- Same quality filtering (applied in previous step)
