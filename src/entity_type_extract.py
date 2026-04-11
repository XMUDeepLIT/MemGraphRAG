#!/usr/bin/env python3
"""
Entity Extraction Script
Features:
1. Split text into chunks of fixed token length
2. Extract entities from each chunk using spacy's en_core_web_trf model
3. Save chunks, entities and entity labels to JSON file
"""

import json
import spacy
from typing import List, Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Entity Extractor Class"""
    
    def __init__(self, model_name: str = "en_core_web_trf", chunk_size: int = 512):
        """
        Initialize entity extractor
        
        Args:
            model_name: spacy model name
            chunk_size: number of tokens per chunk
        """
        self.chunk_size = chunk_size
        logger.info(f"Loading spacy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Model {model_name} not installed, please run: python -m spacy download {model_name}")
            raise
        logger.info(f"Model loaded successfully, chunk size: {chunk_size} tokens")
    
    def chunk_text_by_tokens(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks by token length
        
        Args:
            text: input text
            
        Returns:
            List of chunk information, each element contains chunk_id, text, start_char, end_char
        """
        logger.info("Starting text chunking...")
        doc = self.nlp(text)
        chunks = []
        
        current_tokens = []
        current_start = 0
        
        for i, token in enumerate(doc):
            current_tokens.append(token)
            
            # Create a chunk when reaching chunk_size or end of document
            if len(current_tokens) >= self.chunk_size or i == len(doc) - 1:
                if current_tokens:
                    chunk_start = current_tokens[0].idx
                    chunk_end = current_tokens[-1].idx + len(current_tokens[-1].text)
                    chunk_text = text[chunk_start:chunk_end]
                    
                    chunks.append({
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'start_char': chunk_start,
                        'end_char': chunk_end,
                        'num_tokens': len(current_tokens)
                    })
                    
                    current_tokens = []
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    def extract_entities_from_chunk(self, chunk_text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from a chunk
        
        Args:
            chunk_text: chunk text
            
        Returns:
            List of entities, each entity contains text, label, start_char, end_char
        """
        doc = self.nlp(chunk_text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'label_description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Process text: chunk and extract entities
        
        Args:
            text: input text
            
        Returns:
            List containing chunk and entity information
        """
        chunks = self.chunk_text_by_tokens(text)
        
        logger.info("Starting entity extraction...")
        results = []
        
        for chunk in chunks:
            logger.info(f"Processing chunk {chunk['chunk_id']}...")
            entities = self.extract_entities_from_chunk(chunk['text'])
            
            result = {
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['text'],
                'chunk_start_char': chunk['start_char'],
                'chunk_end_char': chunk['end_char'],
                'num_tokens': chunk['num_tokens'],
                'entities': entities,
                'num_entities': len(entities)
            }
            
            results.append(result)
            logger.info(f"  - Found {len(entities)} entities")
        
        logger.info(f"Entity extraction completed, processed {len(results)} chunks")
        return results
    
    def process_file(self, input_path: str, output_path: str):
        """
        Process text file and save results
        
        Args:
            input_path: input text file path
            output_path: output JSON file path
        """
        logger.info(f"Reading file: {input_path}")
        
        # Read input file
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"File length: {len(text)} characters")
        
        # Process text
        results = self.process_text(text)
        
        # Save results
        self.save_results(results, output_path)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save results to JSON file
        
        Args:
            results: extraction results
            output_path: output file path
        """
        logger.info(f"Saving results to: {output_path}")
        
        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        total_entities = sum(r['num_entities'] for r in results)
        entity_labels = {}
        
        for result in results:
            for entity in result['entities']:
                label = entity['label']
                entity_labels[label] = entity_labels.get(label, 0) + 1
        
        # Build output data
        output_data = {
            'metadata': {
                'total_chunks': len(results),
                'total_entities': total_entities,
                'chunk_size': self.chunk_size,
                'entity_labels_count': entity_labels,
                'model': 'en_core_web_trf'
            },
            'chunks': results
        }
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Results saved")
        logger.info(f"  - Total chunks: {len(results)}")
        logger.info(f"  - Total entities: {total_entities}")
        logger.info(f"  - Entity label distribution: {entity_labels}")


def main():
    """Main function"""
    
    # ==================== Configuration Parameters ====================
    
    # Spacy model name
    MODEL_NAME = "en_core_web_trf"
    
    # Number of tokens per chunk
    CHUNK_SIZE = 512
    
    # Input text file path
    INPUT_FILE = "input/input.txt"
    
    # Output JSON file path
    OUTPUT_FILE = "output/output_entities.json"
    
    # ================================================================
    
    try:
        # Create entity extractor
        extractor = EntityExtractor(
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE
        )
        
        # Process file
        extractor.process_file(INPUT_FILE, OUTPUT_FILE)
        
        logger.info("=" * 50)
        logger.info("Processing completed!")
        logger.info("=" * 50)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Tip: Make sure input file exists, or modify INPUT_FILE variable")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


def demo():
    """Demo function - using example text"""
    
    # Example text
    demo_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. Apple is known for 
    its innovative products including the iPhone, iPad, and Mac computers. In 2023, Tim Cook continues 
    to serve as the CEO of Apple. The company is headquartered at One Apple Park Way in Cupertino, 
    and has retail stores worldwide including in New York, London, and Tokyo. Apple's market 
    capitalization often exceeds 2 trillion dollars, making it one of the most valuable companies 
    in the world. The company was originally founded in Los Altos, California before moving to 
    Cupertino. Steve Jobs returned to Apple in 1997 after being ousted in 1985.
    """
    
    logger.info("Running demo mode...")
    
    # Create entity extractor (using smaller chunk size for demo)
    extractor = EntityExtractor(model_name="en_core_web_sm", chunk_size=50)
    
    # Process text
    results = extractor.process_text(demo_text)
    
    # Save results
    extractor.save_results(results, "demo_entities.json")
    
    # Print partial results
    logger.info("\n" + "=" * 50)
    logger.info("Example result preview:")
    logger.info("=" * 50)
    
    for result in results[:2]:  # Show only first 2 chunks
        logger.info(f"\nChunk {result['chunk_id']}:")
        logger.info(f"Text: {result['chunk_text'][:100]}...")
        logger.info(f"Number of entities: {result['num_entities']}")
        
        for entity in result['entities'][:5]:  # Show only first 5 entities per chunk
            logger.info(f"  - {entity['text']} ({entity['label']}): {entity['label_description']}")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demo mode
        demo()
    else:
        # Run main program
        main()