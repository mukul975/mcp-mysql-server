.PHONY: help install install-dev test lint format clean docs

help:
	@echo "Available commands:"
	@echo "  install     Install production dependencies"
	@echo "  install-dev Install development dependencies"
	@echo "  test        Run tests with pytest"
	@echo "  lint        Run linting with flake8"
	@echo "  format      Format code with black and isort"
	@echo "  clean       Clean up temporary files"
	@echo "  docs        Build documentation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest --cov=mysql_server --cov-report=html --cov-report=term

lint:
	flake8 mysql_server.py
	mypy mysql_server.py

format:
	black mysql_server.py
	isort mysql_server.py

clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

docs:
	mkdocs build

serve-docs:
	mkdocs serve
