import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from collections import defaultdict
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerNLP:
    def __init__(self):
        """Initialize transformer-based NLP components."""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # NER pipeline
        self.ner_pipeline = pipeline(
            "token-classification",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )

        # Zero-shot classification for topic analysis
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        # Sentence transformer for semantic similarity
        self.sentence_model = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=0 if torch.cuda.is_available() else -1
        )

    def get_entities(self, text: str) -> List[Dict]:
        """Get named entities from text."""
        return self.ner_pipeline(text)

    def get_root_topic(self, sentence: str) -> str:
        """Get the main topic of a sentence."""
        # Extract nouns from the sentence using NLTK
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

        if not nouns:
            return ""

        # Use zero-shot classification to identify the most relevant noun
        if len(nouns) > 1:
            result = self.classifier(sentence, nouns)
            return result['labels'][0]
        return nouns[0]

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get embedding for a sentence."""
        features = self.sentence_model(sentence)
        return np.mean(features[0], axis=0)

    def sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate semantic similarity between two sentences."""
        emb1 = self.get_sentence_embedding(sent1)
        emb2 = self.get_sentence_embedding(sent2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

@dataclass
class ContentConfig:
    subject: str
    grade_level: int
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class ContentGenerator:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the content generator with TinyLlama model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _prepare_prompt(self, prompt: str, config: ContentConfig) -> str:
        """Prepare prompt for TinyLlama's chat format."""
        return f"""<|system|>
You are an expert educational content creator. Create content that is clear, accurate, and engaging.

<|user|>
Create educational content for grade {config.grade_level} students about {config.subject}.

Task: {prompt}

Requirements:
- Make it age-appropriate for grade {config.grade_level}
- Include clear explanations
- Add relevant examples
- Use engaging language

<|assistant|>"""

    def generate_content(self, prompt: str, config: ContentConfig) -> str:
        """Generate educational content using TinyLlama."""
        try:
            prepared_prompt = self._prepare_prompt(prompt, config)
            inputs = self.tokenizer(prepared_prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove any system or user prompts that might be in the output
            cleaned_text = generated_text.split("<|assistant|>")[-1].strip()
            return cleaned_text
        except Exception as e:
            logger.error(f"Error in content generation: {str(e)}")
            raise

class ContentRefiner:
    def __init__(self):
        """Initialize the content refinement module with transformer-based NLP."""
        self.nlp = TransformerNLP()
        self.clarity_model = self._load_clarity_model()
        self.coherence_model = self._load_coherence_model()
        self.grade_level_vocab = self._load_grade_level_vocab()
        self.sentence_complexity_thresholds = {
            'words': 20,
            'syllables': 3,
        }

    def _load_clarity_model(self) -> Dict[str, any]:
        """Load parameters for clarity assessment."""
        return {
            'ideal_sentence_length': 15,
            'max_sentence_length': 25,
            'ideal_paragraph_length': 5,
            'transition_words': set([
                'first', 'second', 'third', 'next', 'then', 'finally',
                'therefore', 'thus', 'consequently', 'for example',
                'specifically', 'in other words', 'notably',
                'additionally', 'moreover', 'furthermore', 'however',
                'nevertheless', 'although', 'in conclusion'
            ]),
            'complexity_weights': {
                'sentence_length': 0.4,
                'transition_usage': 0.3,
                'readability': 0.3
            }
        }

    def _load_coherence_model(self) -> Dict[str, Set[str]]:
        """Load coherence assessment parameters."""
        return {
            'topic_transitions': {
                'addition': {
                    'additionally', 'moreover', 'furthermore', 'also',
                    'in addition', 'besides'
                },
                'contrast': {
                    'however', 'nevertheless', 'although', 'on the other hand',
                    'conversely', 'in contrast', 'despite this'
                },
                'example': {
                    'for instance', 'specifically', 'such as', 'for example',
                    'to illustrate', 'namely', 'in particular'
                },
                'conclusion': {
                    'therefore', 'thus', 'consequently', 'as a result',
                    'hence', 'in conclusion', 'to summarize'
                },
                'sequence': {
                    'first', 'second', 'third', 'next', 'then', 'finally',
                    'meanwhile', 'subsequently', 'afterward'
                }
            },
            'coherence_weights': {
                'topic_consistency': 0.4,
                'transition_usage': 0.3,
                'entity_tracking': 0.3
            }
        }

    def _load_grade_level_vocab(self) -> Dict[int, Set[str]]:
        """Load grade-appropriate vocabulary lists with expanded academic words."""
        return {
            5: set([
                'analyze', 'compare', 'describe', 'explain', 'identify',
                'illustrate', 'observe', 'organize', 'recognize', 'sequence',
                'classify', 'define', 'estimate', 'group', 'measure',
                'outline', 'predict', 'record', 'select', 'solve'
            ]),
            6: set([
                'evaluate', 'interpret', 'justify', 'predict', 'summarize',
                'conclude', 'contrast', 'demonstrate', 'distinguish', 'elaborate',
                'examine', 'infer', 'investigate', 'research', 'support',
                'categorize', 'compile', 'discuss', 'explain', 'relate'
            ]),
            7: set([
                'hypothesize', 'synthesize', 'critique', 'formulate',
                'analyze', 'assess', 'compose', 'construct', 'develop',
                'integrate', 'investigate', 'organize', 'propose', 'recommend',
                'validate', 'evaluate', 'generate', 'modify', 'theorize'
            ])
        }

    def _assess_coherence(self, content: str) -> float:
        """
        Assess coherence of content using transformer-based analysis.
        Returns a score between 0 and 1.
        """
        sentences = sent_tokenize(content)

        # Get main topics for each sentence
        topics = [self.nlp.get_root_topic(sent) for sent in sentences]

        # Calculate topic consistency using semantic similarity
        topic_similarities = []
        for i in range(1, len(sentences)):
            similarity = self.nlp.sentence_similarity(sentences[i-1], sentences[i])
            topic_similarities.append(similarity)

        topic_consistency_score = np.mean(topic_similarities) if topic_similarities else 1.0

        # Check transition words usage
        transition_patterns = set.union(*self.coherence_model['topic_transitions'].values())
        transitions_found = sum(1 for sent in sentences
                              if any(pattern in sent.lower()
                                    for pattern in transition_patterns))
        transition_score = min(1.0, transitions_found / (len(sentences) - 1))

        # Check reference consistency using NER
        entities = self.nlp.get_entities(content)
        entity_mentions = defaultdict(list)
        for entity in entities:
            entity_mentions[entity['word']].append(entity['start'])

        entity_consistency = min(1.0, len([e for e in entity_mentions.values()
                                         if len(e) > 1]) / max(1, len(entity_mentions)))

        # Combine scores with weights
        weights = self.coherence_model['coherence_weights']
        final_score = (
            weights['topic_consistency'] * topic_consistency_score +
            weights['transition_usage'] * transition_score +
            weights['entity_tracking'] * entity_consistency
        )

        return round(final_score, 2)

    def _count_syllables(self, word: str) -> int:
        """Count the number of syllables in a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def _assess_clarity(self, content: str) -> float:
        """
        Assess clarity of content using various metrics.
        Returns a score between 0 and 1.
        """
        sentences = sent_tokenize(content)

        # Calculate sentence length score
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
        length_score = max(0, min(1, 1 - abs(avg_sentence_length -
                                           self.clarity_model['ideal_sentence_length']) /
                                           self.clarity_model['max_sentence_length']))

        # Calculate transition words usage
        transitions_found = sum(1 for sent in sentences
                              if any(trans in sent.lower()
                                    for trans in self.clarity_model['transition_words']))
        transition_score = min(1.0, transitions_found / max(1, len(sentences) - 1))

        # Calculate readability using syllable count
        words = word_tokenize(content)
        avg_syllables = np.mean([self._count_syllables(word) for word in words])
        readability_score = max(0, min(1, 1 - (avg_syllables - 1.5) / 2))

        # Combine scores using weights
        weights = self.clarity_model['complexity_weights']
        final_score = (
            weights['sentence_length'] * length_score +
            weights['transition_usage'] * transition_score +
            weights['readability'] * readability_score
        )

        return round(final_score, 2)

    def refine_content(self, content: str) -> Dict[str, any]:
        """Refine the content using various metrics and improvements."""
        coherence_score = self._assess_coherence(content)
        clarity_score = self._assess_clarity(content)
        refined_content = self._apply_refinements(content)

        return {
            "refined": refined_content,
            "metrics": {
                "coherence_score": coherence_score,
                "clarity_score": clarity_score
            }
        }

    def _apply_refinements(self, content: str) -> str:
        """Apply refinements to improve content quality while preserving formatting."""
        paragraphs = content.split("\n")  # Split content into paragraphs based on newlines
        refined_paragraphs = []

        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)  # Split into sentences within each paragraph
            refined_sentences = []

            for i, sent in enumerate(sentences):
                words = word_tokenize(sent)

                # Split very long sentences
                if len(words) > self.sentence_complexity_thresholds['words']:
                    midpoint = len(words) // 2
                    split_point = next((i for i in range(midpoint - 3, midpoint + 4)
                                        if words[i] in {',', 'and', 'but', 'or'}), midpoint)
                    refined_sentences.extend([
                        ' '.join(words[:split_point]),
                        ' '.join(words[split_point:])
                    ])
                else:
                    refined_sentences.append(sent)

                # Add transition words to sentences without transitions
                if i > 0 and not any(
                    transition in sentences[i - 1].lower() for transition in self.clarity_model['transition_words']
                ):
                    transition = "Additionally, "  # Default transition word
                    refined_sentences[-1] = transition + refined_sentences[-1]

            # Recombine sentences into paragraphs
            refined_paragraphs.append(" ".join(refined_sentences))

        return "\n".join(refined_paragraphs)  # Preserve paragraph breaks

class BiasDetector:
    def __init__(self, bias_categories: List[str]):
        """Initialize bias detection module with transformer-based NLP."""
        self.bias_categories = bias_categories
        self.bias_patterns = self._load_bias_patterns()
        self.nlp = TransformerNLP()

    def _load_bias_patterns(self) -> Dict[str, Dict[str, any]]:
        """Load bias detection patterns for different categories."""
        return {
            'gender': {
                'words': [
                    'mankind', 'chairman', 'policeman', 'fireman', 'mailman',
                    'stewardess', 'waitress', 'actress', 'businessman',
                    'congresswoman', 'spokesman', 'housewife', 'manpower',
                    'saleswoman', 'foreman', 'weatherman'
                ],
                'patterns': [
                    r'he/she',
                    r'his/her',
                    r'\b(all|every|most)\s+(men|women)\b',
                    r'\b(businessman|businesswoman)\b',
                    r'\b(male|female)\s+(nurse|doctor|teacher|lawyer)\b',
                    r'\bmanmade\b',
                    r'\b(mankind|humankind)\b'
                ],
                'replacements': {
                    'mankind': 'humanity',
                    'chairman': 'chairperson',
                    'policeman': 'police officer',
                    'fireman': 'firefighter',
                    'mailman': 'mail carrier',
                    'stewardess': 'flight attendant',
                    'waitress': 'server',
                    'actress': 'actor',
                    'businessman': 'business person',
                    'congresswoman': 'member of congress',
                    'spokesman': 'spokesperson',
                    'housewife': 'homemaker',
                    'manpower': 'workforce',
                    'saleswoman': 'salesperson',
                    'foreman': 'supervisor',
                    'weatherman': 'meteorologist',
                    'manmade': 'artificial',
                    'he/she': 'they',
                    'his/her': 'their'
                }
            },
            'cultural': {
                'words': [
                    'exotic', 'primitive', 'tribal', 'third-world',
                    'developing world', 'oriental', 'colored', 'backward',
                    'savage', 'uncivilized', 'ghetto', 'ethnic',
                    'alien', 'illegal alien'
                ],
                'patterns': [
                    r'\b(all|every|most)\s+([A-Z][a-z]+\s+people)\b',
                    r'\b(these|those)\s+people\b',
                    r'\byour\s+kind\b',
                    r'\bthey\s+all\b.*\b(culture|religion|country)\b',
                    r'\b(civilized|uncivilized)\s+culture\b'
                ],
                'replacements': {
                    'exotic': 'unique',
                    'primitive': 'traditional',
                    'tribal': 'indigenous',
                    'third-world': 'developing',
                    'developing world': 'developing countries',
                    'oriental': 'Asian',
                    'colored': 'person of color',
                    'backward': 'developing',
                    'savage': 'indigenous',
                    'uncivilized': 'traditional',
                    'ghetto': 'under-resourced area',
                    'alien': 'immigrant',
                    'illegal alien': 'undocumented immigrant'
                }
            },
            'socioeconomic': {
                'words': [
                    'poor people', 'the poor', 'lower class',
                    'welfare queens', 'ghetto', 'trailer trash',
                    'white trash', 'redneck', 'hillbilly',
                    'disadvantaged', 'privileged'
                ],
                'patterns': [
                    r'\b(poor|rich)\s+people\b',
                    r'\bthe\s+(poor|rich)\b',
                    r'\b(lower|upper)\s+class\b',
                    r'\bon\s+welfare\b',
                    r'\b(government|public)\s+assistance\b',
                    r'\b(privileged|underprivileged)\s+background\b'
                ],
                'replacements': {
                    'poor people': 'people experiencing poverty',
                    'the poor': 'people with limited resources',
                    'lower class': 'lower-income',
                    'welfare queens': 'welfare recipients',
                    'ghetto': 'under-resourced area',
                    'trailer trash': 'mobile home residents',
                    'white trash': 'low-income people',
                    'redneck': 'rural resident',
                    'hillbilly': 'rural resident',
                    'disadvantaged': 'under-resourced',
                    'privileged': 'economically advantaged'
                }
            },
            'ability': {
                'words': [
                    'handicapped', 'crippled', 'retarded', 'disabled',
                    'crazy', 'insane', 'psycho', 'mentally ill',
                    'special needs', 'challenged'
                ],
                'patterns': [
                    r'\b(suffers?|suffering)\s+from\b',
                    r'\bconfined\s+to\s+a\s+wheelchair\b',
                    r'\b(normal|abnormal)\s+person\b',
                    r'\bmental\s+problems\b'
                ],
                'replacements': {
                    'handicapped': 'person with a disability',
                    'crippled': 'person with a physical disability',
                    'retarded': 'person with an intellectual disability',
                    'disabled': 'person with a disability',
                    'crazy': 'person with a mental health condition',
                    'insane': 'person with a mental health condition',
                    'psycho': 'person with a mental health condition',
                    'mentally ill': 'person with a mental health condition',
                    'special needs': 'person with additional needs',
                    'challenged': 'person with a disability'
                }
            }
        }

    def detect_bias(self, content: str) -> Dict[str, List[str]]:
        """
        Detect potential biases using transformer-based analysis.
        """
        bias_report = {category: [] for category in self.bias_categories}
        sentences = sent_tokenize(content)

        # Get named entities
        entities = self.nlp.get_entities(content)

        for category in self.bias_categories:
            patterns = self.bias_patterns[category]

            # Check for biased words
            for word in patterns['words']:
                if word.lower() in content.lower():
                    bias_report[category].append(f"Potentially biased term: '{word}'")

            # Check for biased patterns
            for pattern in patterns['patterns']:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    bias_report[category].append(
                        f"Potentially biased pattern: '{match.group(0)}'"
                    )

            # Check for demographic generalizations
            for entity in entities:
                if entity['entity_group'] in {'ORG', 'MISC'}:
                    prev_words = content[max(0, entity['start']-20):entity['start']].lower()
                    if any(word in prev_words for word in {'all', 'every', 'most'}):
                        bias_report[category].append(
                            f"Potential generalization about: '{entity['word']}'"
                        )

        return bias_report

    def suggest_corrections(self, content: str, bias_report: Dict[str, List[str]]) -> str:
        """Suggest corrections for detected biases."""
        corrected_content = content

        for category, biases in bias_report.items():
            if not biases:
                continue

            replacements = self.bias_patterns[category]['replacements']

            # Apply word replacements
            for biased_term, replacement in replacements.items():
                corrected_content = re.sub(
                    rf'\b{biased_term}\b',
                    replacement,
                    corrected_content,
                    flags=re.IGNORECASE
                )

        return corrected_content

class ContentPipeline:
    def __init__(self, config: ContentConfig):
        """Initialize the content pipeline."""
        self.config = config
        self.generator = ContentGenerator()
        self.refiner = ContentRefiner()
        self.bias_detector = BiasDetector(['gender', 'cultural', 'socioeconomic'])

    def process_content(self, prompt: str) -> Dict[str, any]:
        """Process content through the pipeline."""
        try:
            # Generate initial content
            content = self.generator.generate_content(prompt, self.config)

            # Refine content
            refined_content = self.refiner.refine_content(content)

            # Detect and mitigate bias
            bias_report = self.bias_detector.detect_bias(refined_content["refined"])
            final_content = self.bias_detector.suggest_corrections(
                refined_content["refined"],
                bias_report
            )

            # Prepare result
            result = {
                "original_prompt": prompt,
                "generated_content": content,
                "refined_content": refined_content,
                "final_content": final_content,
                "bias_report": bias_report,
                "metadata": {
                    "subject": self.config.subject,
                    "grade_level": self.config.grade_level,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Save results to a JSON file
            with open("content_output.json", "w") as f:
                json.dump(result, f, indent=2)

            return result

        except Exception as e:
            logger.error(f"Error in content pipeline: {str(e)}")
            raise

def main():
    # Configure the pipeline
    config = ContentConfig(
        subject="Mathematics",
        grade_level=5
    )

    # Initialize pipeline
    pipeline = ContentPipeline(config)

    # Process content
    prompt = "Create a lesson about fractions and decimals"
    result = pipeline.process_content(prompt)

    # Save results
    with open("content_output.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
