import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

def analyze_json_responses(data):
    """
    Analyze the JSON file to understand the structure and patterns of LLM responses
    """
    print(f"Total records: {len(data)}")
    print("="*50)

    # Pattern analysis
    patterns = defaultdict(int)
    errors = []
    sample_responses = defaultdict(list)

    for i, record in enumerate(data):

        response = record.get('response', '')

        # Check if response starts with markdown
        if response is None:
            patterns['None'] += 1
            sample_responses['None'].append((i, response))
        elif response.startswith('```json'):
            patterns['markdown_wrapped'] += 1
            sample_responses['markdown_wrapped'].append((i, response))
        elif response.startswith('```'):
            patterns['other_markdown'] += 1
            sample_responses['other_markdown'].append((i, response))
        elif response.startswith('{'):
            patterns['plain_json'] += 1
            sample_responses['plain_json'].append((i, response))
        elif response.startswith('['):
            patterns['json_array'] += 1
            sample_responses['json_array'].append((i, response))
        else:
            patterns['other_format'] += 1
            sample_responses['other_format'].append((i, response))

    # Display patterns
    print("RESPONSE PATTERNS:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} ({count/len(data)*100:.1f}%)")

    print("\n" + "="*50)
    print("SAMPLE RESPONSES BY PATTERN:")

    for pattern, samples in sample_responses.items():
        print(f"\n{pattern.upper()}:")
        for idx, (record_idx, sample) in enumerate(samples[:3]):  # Show first 3 samples
            print(f"  Sample {idx+1} (record {record_idx}):")
            print(f"    {repr(sample)}")

    return patterns, sample_responses, errors

class LLMResponseParser:
    """
    Parser for extracting structured data from LLM-generated JSON responses
    """

    def __init__(self):
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.errors = []
        self.failed_records = []  # Store failed records separately

    def parse_file(self, data):
        """
        Parse the entire JSON file and extract structured data
        """
        results = []
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.errors = []
        self.failed_records = []  # Reset failed records

        for i, record in enumerate(data):
            try:
                response_text = record['response']
                extracted_data = self.extract_json_from_response(response_text)
                result = {
                    'index': record['index'],
                    'pmid': record['pmid'],
                    'extracted_data': extracted_data,
                    'extraction_status': 'success'
                }
                results.append(result)
                self.successful_extractions += 1

            except Exception as e:
                # Store the original failed record exactly as-is for manual fixing
                self.failed_records.append(record)
                self.failed_extractions += 1
                self.errors.append((record['index'], str(e), record['response'][:200]))

        return results

    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from various response formats
        """
        if not response_text or not isinstance(response_text, str):
            raise ValueError("Empty or invalid response text")

        response_text = response_text.strip()
        response_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', response_text)

        # Method 1: markdown-wrapped JSON
        if response_text.startswith('```json'):
            return self._extract_from_markdown(response_text)

        # Method 2: plain JSON
        elif response_text.startswith(('{', '[')):
            return self._extract_plain_json(response_text)

    def _extract_from_markdown(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from ```json wrapped responses"""
        # Find the JSON content between ```json and ```
        pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(pattern, response_text, re.DOTALL)

        if match:
            json_content = match.group(1).strip()
        else:
            # Fallback: remove ```json from start and ``` from end
            json_content = response_text[7:]  # Remove ```json
            if json_content.endswith('```'):
                json_content = json_content[:-3]
            json_content = json_content.strip()

        try:
            data = json.loads(json_content)
            self._validate_structure(data)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in markdown block: {str(e)}")

    def _extract_plain_json(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from plain JSON responses with error recovery"""
        try:
            data = json.loads(response_text)
            self._validate_structure(data)
            return data
        except json.JSONDecodeError as e:
            # Try to fix common issues
            fixed_text = self._attempt_json_repair(response_text)
            if fixed_text:
                try:
                    data = json.loads(fixed_text)
                    self._validate_structure(data)
                    return data
                except:
                    pass
            raise ValueError(f"Invalid plain JSON: {str(e)}")

    def _attempt_json_repair(self, text: str) -> Optional[str]:
        """Attempt to repair common JSON issues"""
        text = text.strip()

        # If missing closing brace, add it
        if text.startswith('{') and not text.endswith('}'):
            return text + '}'

        # If missing closing bracket, add it
        if text.startswith('[') and not text.endswith(']'):
            return text + ']'

        return None

    def _validate_structure(self, data: Dict[str, Any]) -> None:
        """Validate that extracted data has expected structure"""
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary structure")

        # Check for sentences field (case-insensitive)
        sentences_key = None
        for key in data.keys():
            if key.lower() == 'sentences':
                sentences_key = key
                break

        if sentences_key is None:
            raise ValueError("Missing 'sentences' key (case-insensitive)")

        sentences = data[sentences_key]
        if not isinstance(sentences, list):
            raise ValueError("'sentences' should be a list")

        # Optional: validate sentence content
        for i, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise ValueError(f"Sentence {i} should be a string")
            if len(sentence.strip()) == 0:
                raise ValueError(f"Sentence {i} is empty")

        # Check for Support field (case-insensitive)
        support_key = None
        for key in data.keys():
            if 'support' in key.lower():
                support_key = key
                break

        if support_key is not None:
            support_value = data[support_key]
            if not isinstance(support_value, str):
                raise ValueError(f"'{support_key}' should be a string")
            if support_value.lower() not in ['yes', 'no', 'maybe']:
                raise ValueError(f"'{support_key}' should be 'yes', 'no', 'maybe'")

    def save_failed_records(self, output_path: str) -> None:
        """Save failed records to a JSON file for manual fixing"""
        if not self.failed_records:
            print("No failed records to save.")
            return

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.failed_records, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(self.failed_records)} failed records to {output_path}")

    def get_all_sentences(self, parsed_results: List[Dict[str, Any]]) -> List[str]:
        """Extract all sentences from successfully parsed results"""
        all_sentences = []

        for result in parsed_results:
            if result['extraction_status'] == 'success' and result['extracted_data']:
                # Find sentences field (case-insensitive)
                sentences = []
                for key, value in result['extracted_data'].items():
                    if key.lower() == 'sentences' and isinstance(value, list):
                        sentences = value
                        break

                all_sentences.extend(sentences)

        return all_sentences

    def get_support_responses(self, parsed_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract support responses from successfully parsed results"""
        support_responses = []

        for result in parsed_results:
            if result['extraction_status'] == 'success' and result['extracted_data']:
                # Find support field (case-insensitive)
                support_value = None
                for key, value in result['extracted_data'].items():
                    if 'support' in key.lower():
                        support_value = value
                        break

                if support_value is not None:
                    support_responses.append({
                        'index': result['index'],
                        'support': support_value
                    })

        return support_responses

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save parsed results to a new JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        """Print parsing summary"""
        total = self.successful_extractions + self.failed_extractions
        success_rate = (self.successful_extractions / total * 100) if total > 0 else 0

        print(f"Parsing Summary:")
        print(f"  Total records: {total}")
        print(f"  Successful: {self.successful_extractions}")
        print(f"  Failed: {self.failed_extractions}")
        print(f"  Success rate: {success_rate:.1f}%")

        if self.errors:
            print(f"\nFirst few errors:")
            for i, (record_idx, error, sample) in enumerate(self.errors[:5]):
                print(f"  Record {record_idx}: {error}")
                print(f"    Sample: {repr(sample[:100])}...")

        print(f"error index: {[x[0] for x in self.errors]}")

# Usage example:
"""
# Initial parsing
parser = LLMResponseParser()
results = parser.parse_file(data)
parser.print_summary()

# Save successful results
parser.save_results(results, 'successful_results.json')

# Save failed records for manual fixing
parser.save_failed_records('failed_records_to_fix.json')

# After manually fixing failed_records_to_fix.json, load and reprocess:
with open('failed_records_to_fix.json', 'r') as f:
    fixed_data = json.load(f)

# Parse the fixed records
parser2 = LLMResponseParser()
fixed_results = parser2.parse_file(fixed_data)
parser2.print_summary()

# Combine with original results if needed
all_results = results + fixed_results
"""
