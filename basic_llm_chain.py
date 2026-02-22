"""
Basic LangChain LLM Chain Tutorial
====================================
This script demonstrates the fundamental usage of LangChain with OpenAI's
language model to build a simple LLM chain.

Based on: https://python.langchain.com/docs/tutorials/llm_chain/
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()


def create_llm_chain():
    """
    Creates a basic LangChain LLM chain using OpenAI.

    Returns:
        chain: A LangChain LCEL chain (prompt | llm | parser)
    """
    # 1. Initialize the OpenAI chat model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 2. Define a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question clearly and concisely.",
            ),
            ("human", "{question}"),
        ]
    )

    # 3. Add an output parser to extract the string content
    output_parser = StrOutputParser()

    # 4. Chain the components together using LCEL (LangChain Expression Language)
    chain = prompt | llm | output_parser

    return chain


def run_basic_examples(chain):
    """
    Runs a few example queries through the LLM chain.

    Args:
        chain: A LangChain chain object
    """
    examples = [
        "What is LangChain and what is it used for?",
        "Explain what a Large Language Model (LLM) is in simple terms.",
        "What are the benefits of using RAG (Retrieval-Augmented Generation)?",
    ]

    print("=" * 60)
    print("Basic LangChain LLM Chain Demo")
    print("=" * 60)

    for question in examples:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        response = chain.invoke({"question": question})
        print(f"Answer: {response}")

    print("\n" + "=" * 60)


def run_interactive_session(chain):
    """
    Starts an interactive Q&A session with the LLM chain.

    Args:
        chain: A LangChain chain object
    """
    print("\n" + "=" * 60)
    print("Interactive Q&A Session (type 'exit' to quit)")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue

        response = chain.invoke({"question": question})
        print(f"Answer: {response}")


if __name__ == "__main__":
    # Validate that the API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Please copy .env.example to .env and fill in your API key."
        )

    # Create the chain
    chain = create_llm_chain()

    # Run built-in examples
    run_basic_examples(chain)

    # Optionally, start an interactive session
    run_interactive_session(chain)
