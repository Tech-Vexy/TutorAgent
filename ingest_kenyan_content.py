from skills_db import KnowledgeBase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("IngestContent")

def ingest_data():
    kb = KnowledgeBase()
    
    # Sample Kenyan Curriculum Data
    content = [
        {
            "text": "In KCSE Mathematics, Quadratic Equations are solved using the formula: x = (-b ± √(b² - 4ac)) / 2a. This is a fundamental concept in Form 3 Mathematics.",
            "source": "KCSE Mathematics Form 3 Syllabus"
        },
        {
            "text": "Fasihi ya Kiswahili imegawanyika katika sehemu mbili kuu: Fasihi Simulizi na Fasihi Andishi. Fasihi Simulizi inajumuisha ngano, nyimbo, methali, na vitendawili.",
            "source": "KCSE Kiswahili Form 2 Syllabus"
        },
        {
            "text": "The River and the Source by Margaret Ogola is a key set book for KCSE English Literature. It explores the lives of three generations of women.",
            "source": "KCSE English Literature Set Books"
        },
        {
            "text": "In Biology Form 4, Genetics covers Mendelian inheritance. The first law of inheritance is the Law of Segregation.",
            "source": "KCSE Biology Form 4 Syllabus"
        },
        {
            "text": "Chemistry Form 1 introduces the concept of Acids, Bases, and Indicators. Universal indicator changes color depending on the pH of the solution.",
            "source": "KCSE Chemistry Form 1 Syllabus"
        },
        {
            "text": "History and Government Form 2 covers the Constitution of Kenya 2010. It introduced the Devolved Government structure with 47 Counties.",
            "source": "KCSE History Form 2 Syllabus"
        }
    ]
    
    logger.info("Starting ingestion of Kenyan Education Content...")
    
    total_chunks = 0
    for item in content:
        try:
            chunks = kb.add_document(item["text"], metadata={"source": item["source"]})
            total_chunks += chunks
            logger.info(f"Added {chunks} chunks from {item['source']}")
        except Exception as e:
            logger.error(f"Failed to add content from {item['source']}: {e}")
            
    logger.info(f"Ingestion complete. Total chunks added: {total_chunks}")

if __name__ == "__main__":
    ingest_data()
