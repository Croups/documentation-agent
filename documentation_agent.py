import nest_asyncio
nest_asyncio.apply()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

documents = SimpleDirectoryReader("docs/learn").load_data()
vector_store = SimpleVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
        TitleExtractor(),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,
)

nodes = pipeline.run(documents=documents)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

index.storage_context.persist()

from llama_index.core import (
    load_index_from_storage,
)


storage_context = StorageContext.from_defaults(persist_dir="storage")
loaded_index = load_index_from_storage(storage_context)
query_engine = loaded_index.as_query_engine()


from mirascope.core import openai, prompt_template
from pydantic import BaseModel, Field


class Relevance(BaseModel):
    id: int = Field(..., description="The document ID")
    score: int = Field(..., description="The relevance score (1-10)")
    document: str = Field(..., description="The document text")
    reason: str = Field(..., description="A brief explanation for the assigned score")


@openai.call(
    "gpt-4o-mini",
    response_model=list[Relevance],
    json_mode=True,
)
@prompt_template(
    """
You are an advanced document relevance assessment system. Your task is to evaluate a list of documents and determine their relevance to a given question. Follow these instructions carefully:

1. Review the question:
<question>
{query}
</question>

2. Examine the list of documents:
<documents>
{documents}
</documents>

3. For each document, perform the following analysis inside document_evaluation tags:
   a. Quote relevant passages from the document.
   b. Consider arguments for and against the document's relevance.
   c. List specific ways the document contributes to answering the question.
   d. Count the number of relevant points made in the document.
   e. Determine if the document is relevant (score 5 or above).
   f. If relevant, analyze its content in relation to the question.
   g. Consider both direct and indirect relevance.
   h. Assess the informativeness of the content, not just keyword matches.
   i. Think about how this document might contribute to a complete answer.
   j. Assign a relevance score from 5 to 10.
   k. Provide a reason for the assigned score.

It's OK for this section to be quite long.

4. After analyzing all documents, consider how they might work together to answer the question.

5. Present your final assessment in the following format:

<assessment>
  <document>
    <id>[Document ID]</id>
    <score>[Relevance score (5-10)]</score>
    <reason>[Explanation for the score]</reason>
  </document>
  [Repeat for each relevant document]
  <overall_analysis>
    [Brief explanation of how the documents collectively address the question]
  </overall_analysis>
</assessment>

Important guidelines:
- Exclude documents with relevance scores below 5.
- Prioritize positive, affirmative information over negative statements.
- Be thorough in your analysis, considering all aspects of each document's relevance.
- Ensure your reasoning is clear and justified.

Begin your assessment now.
    """
)
def llm_query_rerank(documents: list[dict], query: str): ...

from typing import cast

from llama_index.core import QueryBundle
from llama_index.core.indices.vector_store import VectorIndexRetriever


def get_documents(query: str) -> list[str]:
    """The get_documents tool that retrieves Mirascope documentation based on the
    relevance of the query"""
    query_bundle = QueryBundle(query)
    retriever = VectorIndexRetriever(
        index=cast(VectorStoreIndex, loaded_index),
        similarity_top_k=10,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)
    choice_batch_size = 5
    top_n = 2
    results: list[Relevance] = []
    for idx in range(0, len(retrieved_nodes), choice_batch_size):
        nodes_batch = [
            {
                "id": idx + id,
                "text": node.node.get_text(),  # pyright: ignore[reportAttributeAccessIssue]
                "document_title": node.metadata["document_title"],
                "semantic_score": node.score,
            }
            for id, node in enumerate(retrieved_nodes[idx : idx + choice_batch_size])
        ]
        results += llm_query_rerank(nodes_batch, query)
    results = sorted(results, key=lambda x: x.score or 0, reverse=True)[:top_n]

    return [result.document for result in results]

from typing import Literal
from pydantic import BaseModel, Field
from mirascope.core import openai, prompt_template


class Response(BaseModel):
    classification: Literal["code", "general"] = Field(
        ..., description="The classification of the question"
    )
    content: str = Field(..., description="The response content")


class DocumentationAgent(BaseModel):
    
    @openai.call("gpt-4o-mini", response_model=Response, json_mode=True)
    @prompt_template(
        """
        SYSTEM:
        You are an AI Assistant that is an expert at answering questions about Mirascope.
        Here is the relevant documentation to answer the question.

        First classify the question into one of two types:
            - General Information: Questions about the system or its components.
            - Code Examples: Questions that require code snippets or examples.

        For General Information, provide a summary of the relevant documents if the question is too broad ask for more details. 
        If the context does not answer the question, say that the information is not available or you could not find it.

        For Code Examples, output ONLY code without any markdown, with comments if necessary.
        If the context does not answer the question, say that the information is not available.

        Examples:
            Question: "What is Mirascope?"
            Answer:
            classification: general
            content: A toolkit for building AI-powered applications with Large Language Models (LLMs).

            Question: "How do I make a basic OpenAI call?"
            Answer:
            classification: code
            content: |
                from mirascope.core import openai

                @openai.call("gpt-4o-mini")
                def basic_call() -> str:
                    return "Hello, world!"

                response = basic_call()
                print(response.content)

        Context:
        {context:list}

        USER:
        {question}
        """
    )
    def _call(self, question: str) -> openai.OpenAIDynamicConfig:
        documents = get_documents(question)
        return {"computed_fields": {"context": documents}}

    def _step(self, question: str):
        answer = self._call(question)
        print("(Assistant):", answer.content)

    def run(self):
        while True:
            question = input("(User): ")
            if question == "exit":
                break
            self._step(question)

if __name__ == "__main__":
    agent = DocumentationAgent()
    agent.run()
    
    
