#!/usr/bin/env python3
"""
"""

from typing import List, Dict

from prompt import (
    RELATION_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_USER_PROMPT_TEMPLATE
)


def format_entities_for_prompt(entities: list) -> str:
    """
 
    """
   
    unique_entities = {}
    for ent in entities:
        text = ent.get("text", "")
        label = ent.get("label", "UNKNOWN")
        if text not in unique_entities:
            unique_entities[text] = label
    
    
    lines = []
    for text, label in unique_entities.items():
        lines.append(f"- {text} ({label})")
    
    return "\n".join(lines)


def build_relation_extraction_messages(text: str, entities: list) -> List[Dict[str, str]]:
    """
    
    """
    entities_str = format_entities_for_prompt(entities)
    
    user_content = RELATION_EXTRACTION_USER_PROMPT_TEMPLATE.format(
        text=text,
        entities=entities_str
    )
    
    messages = [
        {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    return messages


if __name__ == "__main__":
    
    test_text = "Apple Inc. was founded by Steve Jobs in 1976."
    test_entities = [
        {"text": "Apple Inc.", "label": "ORG"},
        {"text": "Steve Jobs", "label": "PERSON"},
        {"text": "1976", "label": "DATE"}
    ]
    
    messages = build_relation_extraction_messages(test_text, test_entities)
    
    print("System message:")
    print(messages[0]["content"])
    print("\nUser message:")
    print(messages[1]["content"])
