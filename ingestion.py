import asyncio
from json import load
import os
import site
import ssl
from typing import Any, Dict, List
from unittest import result
import certifi
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, tavily_crawl, tavily_extract, tavily_map
from langchain_core.documents import Document
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] =certifi.where()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth= 5, max_breath= 20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int =50):
    """Process documents in batches asysnchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore Indexing: Preparing to add {len(documents)} dcouments to vector store",
        Colors.DARKCYAN,
    )

    #create batches
    batches =[
        documents[i : i +batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f" VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)}({len(batch) }documents")

        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    #Process batches consurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results =  await asyncio.gather(*tasks, return_exceptions=True)

    #Count successful batches
    succesful = sum(1 for results in results if results is True)

    if succesful == len(batches):
        log_success(
            f"VectorStore Indexing: All bacthes processed successfully! ({succesful}/{len(batches)}) "
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {succesful}/ {len(batches)} batches succesfully"
        )
    
            

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://docs.langchain.com/oss/python/langchain/overview",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 5,
        "extract_depth": "advanced",
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result['url']})for result in res["results"] if result.get("raw_content")]
    
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URL's from documentation site"
    )

    #Split dcouments into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents  with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 4000, chunk_overlap = 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} dcouments"
    )

    # Process Dcouments asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success(" Dcoumentation ingestion pipeline finished successfully!")
    log_info(" summary: ", Colors.BOLD)
    log_info(f"  . Dcouments extracted:  {len(all_docs)}")
    log_info(f"  . Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())





