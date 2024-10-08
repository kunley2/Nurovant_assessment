import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import re
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory,ConversationSummaryMemory
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser,StrOutputParser
from pydantic.v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel
from operator import itemgetter


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


model_id = "kunley2/Llama-3.2-3B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,  bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
    max_new_tokens=512,
    return_full_text=False,
    num_return_sequences=1,
    # eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(
  pipeline=pipe,
  model_kwargs={"temperature": 0.3},
  pipeline_kwargs={"repetition_penalty":1.2}
)

def answer_query(vector_store, input:str) -> dict:
    """ this function is used to get the insight from a user query"""

    template = """
    Answer the question based only on the information provided for you in the context
    context:
    {context}

    Question:
    {question}
    output only the answer that relates to the question
    """
    answer_template = PromptTemplate(
        template=template, input_variables=["question"],
    )

    answer_chain = answer_template | llm | StrOutputParser()
        

    template2 = """
    You are to use the information in the context to ask the user a question about the answer you gave to ensure they understand it. Make sure you only ask a reasonable question.
    think in step by step and ask a question relevant to the context and answer provided:

    Question:
    {question}

    context: {context}

    answer: {answer}
    Output only the question you want to ask the user
    """
    question_template = PromptTemplate(
        template=template2, input_variables=["context","question","answer"],
    )
    question_chain = question_template | llm | StrOutputParser()



    bullet_point = """
    You are to provide a list of bullet points telling how you came to the conclusion of your answer from the question.
    make the list as concise and short as possible

    Question:
    {question}

    context: {context}

    answer: {answer}
    Give your bullet point in a comma separated list
    """
    bullet_template = PromptTemplate(
        template=bullet_point, input_variables=["context","question","answer"],
    )
    bullet_chain = bullet_template | llm | StrOutputParser()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    chain = ({
        "context": itemgetter("question") | retriever,
        "question":itemgetter("question"),
        } |  RunnablePassthrough.assign(
            answer=answer_chain,
            ) |
            RunnablePassthrough.assign(second_question=question_chain)|
            RunnablePassthrough.assign( bullet=bullet_chain)
    )

    response = chain.invoke({'question':input})

    return response["answer"], response['second_question'], response['bullet']


def evaluate_answer(question:str, answer:str, vector_store) -> dict:
    """This function is used to evaluate the understanding of a user to a particular question"""

    evaluate_template = """
    You are given a question and answer and required to use the context to determine if the user understands the topic by writing either True or Force.
    you are to give a one word output either its True or False, True meaning he understands the topic and False meaning he doesnt understand the topic.\n\n
    Ensure you only give your output either its True or False after evaluating the answer

    Question:
    {question}

    context: {context}

    answer: {answer}

    """
    evaluation_prompt_template = PromptTemplate(
        template=evaluate_template, input_variables=["context","question","answer"],
    )
    evaluation_chain = evaluation_prompt_template | llm | StrOutputParser()

    confidence_template = """
    You are given a question and answer and required to use the context to determine if the user answers the question correctly by rating the user on a scale of 1 - 100, 100 meaning he knows the topic well.\n\n
    Ensure you only give a only the rating

    Question:
    {question}

    context: {context}

    answer: {answer}

    """
    confidence_prompt_template = PromptTemplate(
        template=confidence_template, input_variables=["context","question","answer"],
    )
    confidence_chain = confidence_prompt_template | llm | StrOutputParser()

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    chain =  ({
        "context": itemgetter("question") | retriever,
        "question":itemgetter("question"),
        "answer":itemgetter("answer"),
        } |  RunnablePassthrough.assign(
            knowledge=evaluation_chain,
            confidence=confidence_chain,
            )
    )
    response = chain.invoke({"question":question,"answer":answer})
    return response["knowledge"].split('\n\n')[0], response['confidence'].split('\n\n')[0]