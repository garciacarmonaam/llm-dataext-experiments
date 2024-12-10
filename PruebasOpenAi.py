from langchain_openai import ChatOpenAI
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import pandas as pd
import json
import os
import glob
import re
from typing import Optional
from langchain_community.document_loaders import PDFPlumberLoader

class ResultadosInforme(BaseModel):
    nombre: Optional[str] = Field(description="Nombre del paciente")
    edad: Optional[str] = Field(description="Edad del paciente")
    diagnostico: Optional[str] = Field(description="Diagnóstico del paciente que se expone")
    tests: Optional[str] = Field(description="Pruebas diagnósticas hechas hasta el momento")
    medicacion: Optional[str] = Field(description="Medicación que toma el paciente")

parser = PydanticOutputParser(pydantic_object=ResultadosInforme)

folder_path = 'docs'
pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

def extract_number(file_path):
    match = re.search(r'(\d+)', file_path)  # Find the first occurrence of a number
    return int(match.group()) if match else 0  # Return the number, or 0 if not found

sorted_pdf_files = sorted(pdf_files, key=extract_number)

#os.environ["OPENAI_API_KEY"] = -> User must assign its own OPEN AI API KEY

for file in sorted_pdf_files:
    print(file)
    id_caso = extract_number(file)
    loader = PDFPlumberLoader(file)
    docs = loader.load()

    text_splitter = SemanticChunker(OpenAIEmbeddings())
    documents = text_splitter.split_documents(docs)

    embedder = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    prompt_template = """
    En base al contexto dado, responde en el formato indicado

    Contexto: {context}

    Pregunta: {question}

    Formato de las instrucciones: {format_instructions}

    Respuesta útil:"""


    QA_CHAIN_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
        | parser
    )


    resultado = rag_chain.invoke("Obtener información sobre el paciente").json(ensure_ascii=False)


    # In[9]:


    file_path = "ResultadosGPT4o.xlsx"
    df_existente = pd.read_excel(file_path)

    datos_resultado = json.loads(resultado)
    df_nuevo = pd.DataFrame([datos_resultado])
    df_nuevo['id'] = id_caso


    if df_nuevo.empty or df_nuevo.isnull().all(axis=1).all():
        print("Fila vacia")

    df_actualizado = pd.concat([df_existente, df_nuevo], ignore_index=True)

    df_actualizado.to_excel(file_path, index=False)
