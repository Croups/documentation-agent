# Description
- This system provides an interactive solution for querying documentation, using advanced natural language processing to understand questions and retrieve relevant information. It processes documents through a sophisticated pipeline that includes sentence splitting, title extraction, and semantic embedding.

## Features

- Document vectorization and semantic search
- Intelligent query processing
- Relevance scoring and document reranking
- Support for both general queries and code-specific questions
- Interactive command-line interface
  
## What can you do with this 

- Query Documentation Naturally: Ask questions in plain English about your documentation
- Get Code Examples: Request specific code snippets and implementation examples
- Smart Search: Find relevant information even when your query doesn't match exact keywords
- Process Large Documents: Handle extensive documentation through efficient chunking and embedding
- Contextual Responses: Receive answers that combine information from multiple relevant document sections
- Interactive Usage: Use through a simple command-line interface with continuous interaction

## Example Usage

Question : How can i make a simple openai call with mirascope
Answer :

<img width="620" alt="image" src="https://github.com/user-attachments/assets/5adadd26-147f-4cf8-a403-f74fe6abf886">

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
- Start querying your documentation

## Requirements

- Python 3.8+
- OpenAI API key

This is inspired by Mirascope, LLM Development Library. I strongly suggest you to check mirascope's features
