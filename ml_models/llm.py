import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="",
    temperature=0.2,
    max_tokens=50
)   


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert Machine Learning Engineer, Data Scientist, and AI Assistant.
            You have access to the analysis of the user's uploaded dataset.
            Dataset Context:
            {context}

            Rules:
            1. If the user asks about their uploaded dataset, model performance, preprocessing, EDA, feature importance, metrics, or predictions, answer using the dataset context.
            2. If the user asks a general Machine Learning, Data Science, Statistics, or AI question, answer it clearly. When relevant, connect the explanation back to the uploaded dataset.
            3. If the user asks something completely unrelated to Machine Learning, Data Science, AI, or the uploaded dataset, politely explain that you specialize in helping with ML and the analyzed dataset.
            4. Be concise, practical, and educational.
            """
        ),
        (
            "human",
            "{question}"
        )
    ]
)
chain = prompt | llm | StrOutputParser()
def ask_llm(context, question):
    return chain.invoke(
        {
             "context": context,
             "question": question
        }
    )