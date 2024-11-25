# Description
- This system provides an interactive interface for querying documentation, using advanced natural language processing to understand questions and retrieve relevant information. It processes documents through a sophisticated pipeline that includes sentence splitting, title extraction, and semantic embedding.
## Features

- Document vectorization and semantic search
- Intelligent query processing
- Relevance scoring and document reranking
- Support for both general queries and code-specific questions
- Interactive command-line interface

## Technologies

- LlamaIndex for document processing
- OpenAI embeddings and GPT-4
- Mirascope for LLM application development
- Pydantic for data validation

## Installation
- pip install "mirascope[openai]"
- pip install llama-index
  
## Usage

- Place your documentation in the docs/learn directory
- Run the script:

- python main.py

- Start querying your documentation

## Requirements

- Python 3.8+
- OpenAI API key
