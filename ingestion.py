import asyncio
import os
import ssl
import requests
from typing import List, Dict, Any
import certifi 
from bs4 import BeautifulSoup

from dotenv import load_dotenv

print(certifi.where())

#--------------------Langchain imports--------------------#
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap 
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from logger import (Colors, log_info, log_error, log_warning, log_success, log_header)

load_dotenv()

# SSL Setup
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# HuggingFace Embeddings
if hf_token := os.getenv("HF_TOKEN"):
    log_success("HuggingFace API token found. Using HuggingFaceEmbeddings with API access.", Colors.GREEN)     
    os.environ["HF_TOKEN"] = hf_token
else:
    log_warning("No HuggingFace API token found. Using HuggingFaceEmbeddings without API access.", Colors.YELLOW)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Pinecone Setup
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
indexes_response = pinecone.list_indexes()
index_names = [idx.name for idx in indexes_response]

if index_name not in index_names:
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT")),
    )
    log_success(f"Pinecone index '{index_name}' created successfully.")
else: 
    log_info(f"Pinecone index '{index_name}' already exists. Skipping creation.", Colors.YELLOW)

vector_store = PineconeVectorStore(index=pinecone.Index(index_name), embedding=embeddings)

# Tavily Setup - Text focused
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl(extract=tavily_extract, map=tavily_map)


# scraping images from the web pages using BeautifulSoup (for metadata in the documents)
def scrape_images_from_url(url: str, query: str = " ") -> List[str]:
    """Scrape relevant technical image URLs from a given webpage using BeautifulSoup."""
    try: 
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        image_urls = []

        technique_keywords = ["punch", "hook", "uppercut", "jab", "cross", "stance", "footwork", 
                    "technique", "drill", "combo", "form", "boxing", "judo", "throw", 
                    "ukemi", "seoi", "ouchi", "gari", "strike", "movement"]
        
        bad_keywords = ["book", "ebook", "guide", "program", "buy", "sale", "promo", "advert", 
                        "banner", "logo", "icon", "avatar", "profile"]

        # Look for common image tags and attributes on Judo/MMA content sites
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original") or img.get("data-lazy")
            
            if not src:
                continue

            if src.startswith("//"):
                src = "https:" + src
            elif not src.startswith("http"):
                src = requests.compat.urljoin(url, src)
            
            
            src_lower = src.lower()
            alt = (img.get("alt") or "").lower()

            # Must be a real image file 
            if not any(ext in src_lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                continue

            # Skip obvious ads and book covers by filtering out images with bad keywords in the URL or alt text
            if any(bad_keyword in src_lower or bad_keyword in alt for bad_keyword in bad_keywords):
                continue

            # Strong preference for images with relevant keywords in alt text or URL
            if any(good_keyword in src_lower or good_keyword in alt for good_keyword in technique_keywords):
                if src not in image_urls:
                    image_urls.append(src)
            
            # Fallback : Keep reasonably sized images that are above a certain dimension threshold, even if they don't have keywords (to capture useful technique photos that may not be labeled well)
            elif img.get("width") and int(img.get("width", 0)) > 400:
                if src not in image_urls:
                    image_urls.append(src)

            # Limit to top 5 most likely relevant images to avoid overwhelming the user, and to ensure we stay within token limits when we send these back in the response metadata
            return image_urls[:5]
     
    except Exception as e:
        log_warning(f"Failed to scrape images from {url}: {str(e)}", Colors.YELLOW)
        return []



def index_documents(documents: List[Document], batch_size: int = 200):
    log_header("VECTOR STORAGE PHASE")
    log_info(f" VectorStore Indexing: Preparing to add {len(documents)} chunks to Pinecone.", Colors.DARKCYAN)

    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f" Split into {len(batches)} batches of {batch_size} chunks each.", Colors.DARKCYAN)

    for i, batch in enumerate(batches, 1):
        try:
            vector_store.add_documents(batch)
            log_success(f"Indexed batch {i}/{len(batches)} ({len(batch)} documents).")
        except Exception as e:
            log_error(f"Error indexing batch {i}/{len(batches)}: {e}")

    log_success("All batches indexed successfully.")


async def main():
    
    log_header("DOCUMENTATION INGESTION PIPELINE - CRAWLING PHASE")
    log_info("TavilyCrawl: Starting crawl for MMA/Judo content (Text-focused + Light Image Capture)", Colors.PURPLE)

    urls = [
        "https://blog.joinfightcamp.com/training/six-6-basic-boxing-punches/",
        "https://blog.joinfightcamp.com/training/how-to-aggressively-close-distance-in-boxing/",
        "https://blog.joinfightcamp.com/training/how-to-throw-long-boxing-combos-without-losing-steam",
        "https://blog.joinfightcamp.com/training/my-first-time-sparring-a-pro-boxer-fightcamps-coach-pj/",
        "https://blog.joinfightcamp.com/training/shake-up-cardio-unique-ways-boxers-build-fight-endurance",
        "https://blog.joinfightcamp.com/wellness/a-boxers-guide-to-meal-frequency/",
        "https://blog.joinfightcamp.com/wellness/mental-benefits-of-boxing/",
        "https://blog.joinfightcamp.com/wellness/how-safe-are-boxing-workouts-what-you-should-know/",
        "https://blog.joinfightcamp.com/training/gym-etiquette-101-respectful-boxing-gym-culture",
        "https://blog.joinfightcamp.com/training/the-jab-5-common-mistakes-how-to-fix-them/",
        "https://blog.joinfightcamp.com/boxing-equipment/6-boxer-jump-rope-tricks-for-beginners-boxing-training/",
        "https://blog.joinfightcamp.com/boxing-equipment/5-things-you-need-to-set-up-your-at-home-boxing-gym/",
        "https://blog.joinfightcamp.com/boxing-equipment/hand-wrapping-methods-for-boxing-kickboxing-mma/",
        "https://blog.joinfightcamp.com/fight-news/common-boxing-match-terms-defined-boxing-101/",
        "https://blog.joinfightcamp.com/fight-news/youtube-boxing-from-influencers-to-boxers",
        "https://blog.joinfightcamp.com/training/shake-up-cardio-unique-ways-boxers-build-fight-endurance/",
        "https://blog.joinfightcamp.com/training/flo-masters-3-reasons-to-shadowbox-with-hand-weights/",
        "https://blog.joinfightcamp.com/training/boxing-cardio-training-running-to-get-in-shape/",
        "https://blog.joinfightcamp.com/training/break-out-of-that-cardio-rut-boxing-cardio-explained/",

        "https://blog.rumbleboxinggym.com/boxing-for-beginners",
        "https://expertboxing.com/the-beginners-guide-to-boxing",
        "https://expertboxing.com/basic-boxing-footwork-drills",
        "https://expertboxing.com/intermediate-boxing-skills",
        "https://expertboxing.com/10-advanced-secrets-to-balance-and-footwork",
        "https://expertboxing.com/5-common-punching-mistakes",
        "https://expertboxing.com/secrets-to-boxing-defense",
        "https://expertboxing.com/muhammad-ali-boxing-footwork-technique",
        "https://expertboxing.com/mastering-the-left-hook",
        "https://expertboxing.com/boxing-defense-techniques",
        "https://expertboxing.com/advanced-slipping-technique-head-movement",
        "https://expertboxing.com/how-to-slip-punches-in-boxing",
        "https://expertboxing.com/advanced-slipping-technique-part-2-body-movement",
        "https://expertboxing.com/how-to-parry-punches",
        "https://expertboxing.com/how-to-slip-punches",
        "https://expertboxing.com/how-to-shoulder-roll",
        "https://expertboxing.com/secret-to-keeping-your-hands-up-in-boxing",
        "https://expertboxing.com/how-to-beat-a-shorter-boxer",
        "https://expertboxing.com/how-to-beat-a-taller-boxer",
        "https://expertboxing.com/boxing-bounce-step-footwork-technique",
        "https://expertboxing.com/why-jumping-rope-is-the-1-footwork-drill-for-beginner-boxers",
        "https://expertboxing.com/10-boxing-footwork-tips",
        "https://expertboxing.com/pro-boxing-tips-for-punching-power-frank-buglioni",
        "https://expertboxing.com/10-pro-boxing-techniques",
        "https://expertboxing.com/how-to-throw-a-jab",
        "https://expertboxing.com/corkscrew-punch-technique",
        "https://expertboxing.com/how-to-throw-an-uppercut",
        "https://expertboxing.com/the-ultimate-boxing-jab-guide",
        "https://expertboxing.com/setting-up-the-left-hook",
        "https://expertboxing.com/5-types-of-jabs",
    ]

    all_docs: List[Document] = []

    for url in urls: 
        log_info(f"Invoking TavilyCrawl for URL: {url}", Colors.CYAN)

        try:
            res = tavily_crawl.invoke({
                "url": url,
                "max_depth": 2, # Limit depth to focus on main content and avoid deep navigation
                "extract_depth": "basic", # Focus on text content, minimal images
                "include_images": True          # Keeping light image capture
            })

            if isinstance(res, Dict) and "error" in res:
                log_error(f"Tavily error for {url}: {res['error']}", Colors.RED)
                await asyncio.sleep(6)
                continue

            if isinstance(res, str):
                log_warning(f"Tavily returned plain text for {url}. Skipping.", Colors.YELLOW)
                await asyncio.sleep(3)
                continue

            results_list = res.get("results", []) if isinstance(res, Dict) else []

            log_info(f"Received {len(results_list)} items from {url}", Colors.CYAN)

            for item in results_list: 
                if not isinstance(item, Dict):
                    continue

                content = item.get("raw_content") or item.get("content") or item.get("markdown")
                source = item.get("url", url)

                # Get Images : Try Tavily First, then use our BeautifulSoup scraper as a fallback for better image capture
                images = item.get("images", [])

                if len(images) == 0:
                    log_info(f"No images from Tavily for {source} - running BeautifulSoup scraper.", Colors.YELLOW)
                    images = scrape_images_from_url(source)

                if content and len(str(content).strip()) > 100:
                    all_docs.append(
                        Document(
                            page_content=str(content).strip(),
                            metadata={
                                "source": source,
                                "images": images,
                                "image_count": len(images)
                            }
                        )
                    )
                    log_success(f"Added: {source[:80]}... | Images: {len(images)}", Colors.GREEN)
                else: 
                    log_warning(f"Skipped: {source[:80]} — insufficient content", Colors.YELLOW)

        except Exception as e:
            log_error(f"Failed to crawl {url}: {str(e)}", Colors.RED)

        await asyncio.sleep(4)

    log_success(f"Successfully collected {len(all_docs)} documents from Tavily crawl.", Colors.GREEN)

    # Chunking & Indexing
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Preparing to chunk {len(all_docs)} documents (4000/200)", Colors.BLUE)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )

    documents_split = text_splitter.split_documents(all_docs)
    log_success(f"Chunked into {len(documents_split)} chunks.")

    index_documents(documents=documents_split, batch_size=200)

    log_header("INGESTION PIPELINE COMPLETE")
    log_success(f"Total documents indexed: {len(all_docs)} | Total Chunks: {len(documents_split)}")
    log_info("You can now use these documents for retrieval and question-answering tasks.", Colors.GREEN)


if __name__ == "__main__":
    asyncio.run(main())