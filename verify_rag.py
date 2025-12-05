from skills_db import KnowledgeBase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("VerifyRAG")

def verify_retrieval():
    kb = KnowledgeBase()
    
    queries = [
        "What is the formula for quadratic equations in KCSE?",
        "Explain Fasihi Simulizi.",
        "Who wrote The River and the Source?",
        "What is the first law of inheritance in Biology?",
        "What is the role of Universal Indicator in Chemistry?",
        "How many counties are there in Kenya?"
    ]
    
    logger.info("Verifying RAG Retrieval...")
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        results = kb.retrieve_knowledge(query, n_results=1)
        if results:
            for res in results:
                logger.info(f"Retrieved: {res.get('text')}")
                logger.info(f"Source: {res.get('metadata', {}).get('source')}")
        else:
            logger.warning("No results found.")

if __name__ == "__main__":
    verify_retrieval()
