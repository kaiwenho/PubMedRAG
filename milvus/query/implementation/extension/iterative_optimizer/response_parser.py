"""
Simplified LLM Response Parser for PubMed Extension Pipeline

Handles two types of LLM responses:
1. Sentence generation: {"sentences": [...]}
2. Validation: {"support": "yes|no|maybe", "explanation": "..."}
"""

import json
import re
from typing import Dict, Any, Optional


class SimpleLLMResponseParser:
    """Lightweight parser for LLM JSON responses"""
    
    def parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response and extract JSON
        
        Args:
            response_text: Raw LLM response string
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not response_text or not isinstance(response_text, str):
            return None
        
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = self._extract_from_markdown(response_text)
        
        # Clean escape characters
        response_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', response_text)
        
        try:
            data = json.loads(response_text)
            return self._normalize_keys(data)
        except json.JSONDecodeError:
            # Attempt repair
            repaired = self._attempt_repair(response_text)
            if repaired:
                try:
                    data = json.loads(repaired)
                    return self._normalize_keys(data)
                except:
                    pass
        
        return None
    
    def _extract_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks"""
        # Match ```json...``` or ```...```
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: remove ``` markers
        text = text.strip('`').strip()
        if text.startswith('json'):
            text = text[4:].strip()
        
        return text
    
    def _attempt_repair(self, text: str) -> Optional[str]:
        """Attempt to repair malformed JSON"""
        text = text.strip()
        
        # Add missing closing braces/brackets
        if text.startswith('{') and not text.endswith('}'):
            return text + '}'
        if text.startswith('[') and not text.endswith(']'):
            return text + ']'
        
        return None
    
    def _normalize_keys(self, data: Dict) -> Dict:
        """Normalize keys to lowercase for case-insensitive access"""
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        for key, value in data.items():
            normalized[key.lower()] = value
        
        return normalized


class BatchResponseParser:
    """Parser with batch processing and error tracking"""
    
    def __init__(self):
        self.parser = SimpleLLMResponseParser()
        self.success_count = 0
        self.failure_count = 0
        self.failed_indices = []
    
    def parse_batch(self, responses: list) -> list:
        """
        Parse a batch of responses
        
        Args:
            responses: List of dicts with 'index' and 'response' keys
            
        Returns:
            List of successfully parsed results
        """
        results = []
        self.success_count = 0
        self.failure_count = 0
        self.failed_indices = []
        
        for item in responses:
            index = item.get('index')
            response_text = item.get('response', '')
            
            parsed = self.parser.parse_response(response_text)
            
            if parsed:
                results.append({
                    'index': index,
                    'data': parsed,
                    'status': 'success'
                })
                self.success_count += 1
            else:
                self.failed_indices.append(index)
                self.failure_count += 1
        
        return results
    
    def print_summary(self):
        """Print parsing statistics"""
        total = self.success_count + self.failure_count
        success_rate = (self.success_count / total * 100) if total > 0 else 0
        
        print(f"\nParsing Summary:")
        print(f"  Total: {total}")
        print(f"  Success: {self.success_count}")
        print(f"  Failed: {self.failure_count}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        if self.failed_indices:
            print(f"  Failed Indices: {self.failed_indices[:10]}")
            if len(self.failed_indices) > 10:
                print(f"  ... and {len(self.failed_indices) - 10} more")


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_sentences_response(data: Dict) -> bool:
    """Validate sentence generation response format"""
    if not isinstance(data, dict):
        return False
    
    sentences = data.get('sentences')
    if not sentences or not isinstance(sentences, list):
        return False
    
    return all(isinstance(s, str) and s.strip() for s in sentences)


def validate_support_response(data: Dict) -> bool:
    """Validate support validation response format"""
    if not isinstance(data, dict):
        return False
    
    support = data.get('support', '').lower()
    if support not in ['yes', 'no', 'maybe']:
        return False
    
    # Explanation is optional but should be string if present
    explanation = data.get('explanation')
    if explanation is not None and not isinstance(explanation, str):
        return False
    
    return True


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Parse single response
    parser = SimpleLLMResponseParser()
    
    response1 = '''```json
    {
      "sentences": [
        "Aspirin treats headaches.",
        "Headaches can be relieved by aspirin.",
        "Aspirin is effective for headache treatment."
      ]
    }
    ```'''
    
    result1 = parser.parse_response(response1)
    print("Example 1 - Sentence Generation:")
    print(result1)
    print(f"Valid: {validate_sentences_response(result1)}\n")
    
    # Example 2: Parse validation response
    response2 = '''{
      "support": "yes",
      "explanation": "The abstract explicitly states that aspirin reduces headache severity."
    }'''
    
    result2 = parser.parse_response(response2)
    print("Example 2 - Validation:")
    print(result2)
    print(f"Valid: {validate_support_response(result2)}\n")
    
    # Example 3: Batch processing
    batch_parser = BatchResponseParser()
    
    responses = [
        {'index': 0, 'response': response1},
        {'index': 1, 'response': response2},
        {'index': 2, 'response': 'invalid json{{{'},
    ]
    
    results = batch_parser.parse_batch(responses)
    batch_parser.print_summary()
    
    print("\nBatch Results:")
    for r in results:
        print(f"  Index {r['index']}: {r['status']}")
