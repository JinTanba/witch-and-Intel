.PHONY: help install test lint format clean deploy local-test

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean up files"
	@echo "  make deploy      - Deploy to AWS Lambda"
	@echo "  make local-test  - Run local Lambda tests"

install:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements-dev.txt

test:
	. venv/bin/activate && pytest tests/ -v

lint:
	. venv/bin/activate && flake8 src/ tests/
	. venv/bin/activate && mypy src/ tests/

format:
	. venv/bin/activate && black src/ tests/
	. venv/bin/activate && isort src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf lambda-package.zip package/
	rm -rf .coverage htmlcov/
	find . -type f -name "*.pyc" -delete

deploy:
	./scripts/deploy.sh

local-test:
	. venv/bin/activate && python local_test.py

setup-aws-role:
	@echo "Creating IAM role for Lambda..."
	@echo "Run: aws iam create-role --role-name lambda-nlp-role --assume-role-policy-document file://scripts/trust-policy.json"
	@echo "Then: aws iam attach-role-policy --role-name lambda-nlp-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"