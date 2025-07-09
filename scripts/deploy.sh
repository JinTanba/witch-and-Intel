#!/bin/bash

# Deploy script for AWS Lambda NLP function

set -e

# Configuration
FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-nlp-processor}"
RUNTIME="python3.11"
HANDLER="src.lambda_handler.lambda_handler"
TIMEOUT=300
MEMORY_SIZE=512
REGION="${AWS_REGION:-us-east-1}"
ROLE_ARN="${LAMBDA_ROLE_ARN}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Lambda deployment for ${FUNCTION_NAME}${NC}"

# Check if role ARN is provided
if [ -z "$ROLE_ARN" ]; then
    echo -e "${RED}Error: LAMBDA_ROLE_ARN environment variable is not set${NC}"
    echo "Please set it with: export LAMBDA_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE"
    exit 1
fi

# Clean up previous builds
echo -e "${YELLOW}Cleaning up previous builds...${NC}"
rm -rf lambda-package.zip
rm -rf package/

# Create package directory
mkdir -p package

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt -t package/ --platform manylinux2014_x86_64 --only-binary=:all:

# Copy source code
echo -e "${YELLOW}Copying source code...${NC}"
cp -r src package/

# Create deployment package
echo -e "${YELLOW}Creating deployment package...${NC}"
cd package
zip -r ../lambda-package.zip . -x "*.pyc" -x "*__pycache__*"
cd ..

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo -e "${YELLOW}Updating existing function...${NC}"
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda-package.zip \
        --region $REGION
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --handler $HANDLER \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION
else
    echo -e "${YELLOW}Creating new function...${NC}"
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --role $ROLE_ARN \
        --handler $HANDLER \
        --code ZipFile=fileb://lambda-package.zip \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION
fi

echo -e "${GREEN}Deployment completed successfully!${NC}"

# Test the function
echo -e "${YELLOW}Testing the deployed function...${NC}"
aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"action": "tokenize", "text": "Hello from AWS Lambda!"}' \
    --region $REGION \
    response.json

echo -e "${GREEN}Test response:${NC}"
cat response.json
rm response.json

echo -e "${GREEN}Deployment and testing completed!${NC}"