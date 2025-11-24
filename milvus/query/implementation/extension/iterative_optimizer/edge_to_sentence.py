import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class EdgeTextGenerator:
    """
    Generates concat_sentence and ai_sentences for edges.

    This ensures newly added edges have the required text fields
    for evaluation.
    """

    def __init__(
        self,
        node_dict_path: str = 'dict/rtx-kg2_id_info_dictionary.json',
        predicate_dict_path: str = 'dict/biolink_pred_info_dictionary.json',
        llm_client = None,
        response_parser = None
    ):
        """
        Initialize text generator.

        Args:
            node_dict_path: Path to node dictionary
            predicate_dict_path: Path to predicate dictionary
            llm_client: LLM client for AI sentence generation (optional)
            response_parser: Parser for LLM responses (optional)
        """
        # Load dictionaries
        with open(node_dict_path, 'r') as f:
            self.node_dict = json.load(f)

        with open(predicate_dict_path, 'r') as f:
            self.predicate_dict = json.load(f)

        self.llm_client = llm_client
        self.response_parser = response_parser

        print(f"EdgeTextGenerator initialized:")
        print(f"  Nodes: {len(self.node_dict)}")
        print(f"  Predicates: {len(self.predicate_dict)}")
        print(f"  LLM available: {llm_client is not None}")

    def generate_concat_sentence(self, subject: str, predicate: str, obj: str) -> str:
        """
        Generate concat_sentence by concatenating subject, predicate, object names.

        Args:
            subject: Subject ID (e.g., "MESH:D001234")
            predicate: Predicate (e.g., "biolink:treats")
            obj: Object ID (e.g., "MESH:D005678")

        Returns:
            Concatenated sentence string
        """
        # Get names
        subject_name = self.node_dict.get(subject, {}).get('name', subject)
        object_name = self.node_dict.get(obj, {}).get('name', obj)

        # Remove biolink: prefix from predicate
        predicate_name = predicate.removeprefix('biolink:').replace("_", " ")

        # Concatenate
        concat_sentence = f"{subject_name} {predicate_name} {object_name}"

        return concat_sentence

    def generate_prompt(self, subject: str, predicate: str, obj: str) -> Optional[str]:
        """
        Generate LLM prompt for AI sentence generation.

        Args:
            subject: Subject ID
            predicate: Predicate
            obj: Object ID

        Returns:
            Prompt string or None if missing info
        """
        subj_info = self.node_dict.get(subject)
        if not subj_info:
            return None

        obj_info = self.node_dict.get(obj)
        if not obj_info:
            return None

        pred_info = self.predicate_dict.get(predicate)
        if not pred_info:
            return None

        prompt = f"""Convert the following biochemical edge into natural language sentences that express its meaning. Generate 3 different sentence variations that convey the same relationship using different phrasings or perspectives.
Input Format:

Edge: {subj_info['name']} --{predicate}-> {obj_info['name']}
Subject: {subj_info}
Object: {obj_info}
Predicate: {pred_info}

Instructions:
Generate 3 distinct, grammatically correct sentences
Each sentence should accurately reflect the biochemical relationship
Use the entity descriptions and predicate definition to ensure precision
Vary the sentence structure and vocabulary while maintaining scientific accuracy
Consider different aspects or perspectives of the relationship when creating variations

Output Format: Return your response as a JSON object with the following structure:
{{
  "sentences": [
    "First sentence variation",
    "Second sentence variation",
    "Third sentence variation"
  ]
}}"""

        return prompt

    def generate_ai_sentences(
        self,
        subject: str,
        predicate: str,
        obj: str,
        model: str = "gpt-oss:20b"
    ) -> Optional[List[str]]:
        """
        Generate AI sentences using LLM.

        Args:
            subject: Subject ID
            predicate: Predicate
            obj: Object ID
            model: LLM model to use

        Returns:
            List of AI-generated sentences, or None if generation fails
        """
        if not self.llm_client or not self.response_parser:
            return None

        # Generate prompt
        prompt = self.generate_prompt(subject, predicate, obj)
        if not prompt:
            return None

        try:
            # Call LLM (synchronous call)
            response = self.llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={'num_ctx': 8192},
            )
            response_text = response['message']['content']
            # Parse response
            parsed = self.response_parser.parse_response(response_text)

            if parsed and 'sentences' in parsed:
                sentences = parsed['sentences']
                if isinstance(sentences, list) and len(sentences) > 0:
                    return sentences

            return None

        except Exception as e:
            print(f"  Warning: AI sentence generation failed: {e}")
            return None

    def generate_edge_text(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Generate both concat_sentence and ai_sentences for an edge.

        Args:
            subject: Subject ID
            predicate: Predicate
            obj: Object ID

        Returns:
            Tuple of (concat_sentence, ai_sentences)
            ai_sentences will be None if LLM not available or fails
        """
        # Generate concat_sentence (always succeeds)
        concat_sentence = self.generate_concat_sentence(subject, predicate, obj)

        # Generate AI sentences (synchronous call)
        ai_sentences = self.generate_ai_sentences(subject, predicate, obj)

        return concat_sentence, ai_sentences
