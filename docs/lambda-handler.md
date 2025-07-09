# Text to Coordinates Lambda Handler

This document describes the Lambda handler implementation for the `ITextToCoordinatesMapper.fit_transform` functionality.

## Architecture

The implementation follows Clean Architecture principles:

1. **Application Layer** (`src/application/handlers/`)
   - `ITextProcessingHandler.py` - Interface defining the handler contract
   - `TextProcessingHandler.py` - Implementation handling business logic

2. **Infrastructure Layer** (`src/infrastructure/lambda/`)
   - `lambda_handler.py` - AWS Lambda entry point
   - `lambda_factory.py` - Factory for creating handlers with proper dependencies

## API Endpoint

The Lambda function is exposed via API Gateway at:
```
POST /text/coordinates
```

### Request Format

```json
{
  "sentences": [
    "First sentence to process",
    "Second sentence to process"
  ],
  "tag": "optional-model-tag"
}
```

### Response Format

Success (200):
```json
{
  "coordinates": [
    [0.1234, 0.5678],
    [0.2345, 0.6789]
  ],
  "message": "Successfully processed 2 sentences"
}
```

Error (400/500):
```json
{
  "error": "Error message describing what went wrong"
}
```

## Local Testing

Run the example locally:
```bash
python src/main_lambda.py
```

Run unit tests:
```bash
pytest tests/test_text_processing_handler.py -v
```

Run integration tests:
```bash
pytest tests/test_lambda_integration.py -v
```

## Deployment

Deploy using Serverless Framework:
```bash
serverless deploy --stage prod
```

## Environment Variables

- `USE_REDIS_CACHE` - Set to 'true' to use Redis cache (default: 'false')
- `REDIS_HOST` - Redis host (default: 'localhost')
- `REDIS_PORT` - Redis port (default: '6379')
- `REDIS_DB` - Redis database number (default: '0')

## Performance Notes

- The Lambda function reuses handler instances across invocations for better performance
- BERT model loading happens once during cold start
- Use Redis cache in production for model persistence across Lambda containers