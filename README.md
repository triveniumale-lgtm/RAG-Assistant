# RAG Assistant – Document Intelligence System

An AI-powered document intelligence system that allows users to interact with their documents using Retrieval-Augmented Generation (RAG).

## Overview

RAG Assistant enables users to upload PDF documents, process their content into smaller chunks, retrieve relevant information and generate AI-powered answers based on the document context.

## Features

- PDF document upload
- Automatic text extraction
- Document chunking
- Vector embeddings
- FAISS vector database
- Semantic similarity search
- Context-aware AI responses
- Interactive Streamlit interface

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- FAISS
- LangChain (if used in the implementation)
- Vector Search
- Retrieval-Augmented Generation (RAG)

## How It Works

1. Upload a PDF document.
2. Extract and split the document into chunks.
3. Convert document chunks into vector embeddings.
4. Store and retrieve relevant information using FAISS.
5. Generate answers using Gemini based on retrieved context.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Configure your Gemini API key securely using Streamlit secrets.

Run the application:

```bash
streamlit run app.py
```

## Purpose

Built as an AI engineering portfolio project to explore document intelligence, semantic search and Retrieval-Augmented Generation.

## Developer

AI Engineering Student
