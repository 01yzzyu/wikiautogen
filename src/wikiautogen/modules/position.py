import logging
import json
import os
from typing import List, Dict, Union
from openai import OpenAI
from tqdm import tqdm
import dspy
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO)

COMMON_PROMPT_TEMPLATE = """You are an expert in evaluating image insertion suggestions for an article. Use the provided additional data and context to evaluate and improve these suggestions.

Follow these rules for a clear response:
1. Leverage the context to re-evaluate image insertion details for a smooth final response.
2. Don't repeat information already in the response.
3. Format your answer neatly, following these key aspects, making it easy for people to read and understand.

Key aspects:
- Information Supplementation: Images offer abundant crucial info hard to present in text, expanding understanding.
- Visual Reinforcement: They visualize abstract text, deepening comprehension.
- Theme Relevance: Images focus on the theme, showing key points intuitively for grasping the main idea.
- Emotional Resonance: Images evoke strong emotions, more so than text. """

class SuperviseDialogue(dspy.Signature):
    """You are a helpful monitor, used to supervise the work of other agents."""
    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    proposal = dspy.InputField(prefix="Proposal for research: ", format=str)
    section_content = dspy.InputField(prefix="Generated section content: ", format=str)
    feedback = dspy.OutputField(format=str)

class ConversationMonitor(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.monitor_dialogue = dspy.Predict(SuperviseDialogue)
        self.engine = engine

    def forward(self, topic: str, proposal: str, section_content: str):
        with dspy.settings.context(lm=self.engine):
            feedback = self.monitor_dialogue(
                topic=topic,
                proposal=proposal,
                section_content=section_content
            ).feedback
        return feedback

class MMAnalyzer:
    """A class to analyze article content and suggest optimal image insertions with rate limit handling."""
    
    def __init__(self, api_key: str, engine: str = "gpt-4o", temperature: float = 1.0, top_p: float = 0.9):
        self.client = OpenAI(api_key=api_key)
        self.engine = engine
        self.default_params = {
            "model": engine,
            "temperature": temperature,
            "top_p": top_p
        }
        self.dspy_engine = dspy.OpenAI(api_key=api_key, model=engine)
        self.monitor = ConversationMonitor(self.dspy_engine)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
    def _call_openai(self, params):
        """Helper method with retry logic for OpenAI API calls."""
        return self.client.chat.completions.create(**params)

    def analyze_section(self, section_content: str, max_image_num: int = 1) -> List[Dict[str, str]]:
        prompt = (
            f"Thoroughly analyze the following section of an article and propose at most {max_image_num} segments where inserting images would significantly enhance the understanding and engagement of the content. Limit in 15 the suggestions of images inserting in full paper totally.  These sentences must be guaranteed to be exactly the same as the original text. Avoid any instance where the target sentence for image insertion is within a new section starting with '#'. All insertions should be confined to the same subsection (marked by '##').\n\n"
            f"{section_content}\n\n"
            "For each suggestion, please return it in JSON format, which must include the following fields:\n"
            "1. Section: The name of the section where the image should be inserted. Format as 'Section number - Section name' (if section name is available, otherwise just 'Section number').\n"
            "2. Subsection: The name of the subsection where the image should be inserted. Format as 'Subsection number - Subsection name' (if subsection name is available, otherwise just 'Subsection number'). If not applicable, set it to 'Not applicable'.\n"
            "3. Position: The specific starting point of the sentence where the image should be inserted. Indicate the exact sentence text, with the sentence before and after.\n"
            "4. Description: Limit in 60 words. A detailed description of the content that the image should represent.\n"
            "5. Title: A suitable title or filename for the image, concise and descriptive.\n\n"
            "For example:\n"
            "{\n"
            "  \"Section\": \"Section 1 - Introduction\",\n"
            "  \"Subsection\": \"Not applicable\",\n"
            "  \"Position\": \"Before sentence: The hurricane had a significant impact on the region. Target sentence: After the hurricane, the area was severely damaged. After sentence: Rescue teams were quickly dispatched to the area.\",\n"
            "  \"Description\": \"A satellite image of the area after a hurricane, depicting destroyed buildings, flooded roads, and debris-filled landscapes.\",\n"
            "  \"Title\": \"HurricaneSatelliteImage.jpg\"\n"
            "}"
        )

        params = {
            **self.default_params,
            "messages": [
                {"role": "system", "content": "You are a professional image suggestion generator."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = self._call_openai(params)
            content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            suggestions = []
            try:
                suggestions.append(json.loads(content))
            except json.JSONDecodeError:
                for json_str in content.split("\n\n"):
                    if json_str.strip():
                        suggestions.append(json.loads(json_str))
            return suggestions[:max_image_num]
        except Exception as e:
            logging.error(f"Failed to parse GPT response: {e}")
            return []

    def generate_reflection(self, suggestions: List[Dict[str, str]], section_content: str, supervisor_feedback: str) -> str:
        specific_instructions = (
            f"The article section content: {section_content}\n"
            f"The image insertion suggestions: {json.dumps(suggestions)}\n"
            f"Supervisor feedback: {supervisor_feedback}\n"
            "Evaluate these suggestions based on Information Supplementation, Visual Reinforcement, Theme Relevance, "
            "and Emotional Resonance, considering the supervisor feedback. Provide detailed feedback on quality, "
            "relevance, and improvements. End with '[IMPROVE]' if issues exist, or '[NO_ISSUE]' if none."
        )
        prompt = COMMON_PROMPT_TEMPLATE.format(specific_instructions=specific_instructions)
        
        params = {
            **self.default_params,
            "messages": [
                {"role": "system", "content": "You are an expert at evaluating image insertion suggestions."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = self._call_openai(params)
            reflection = response.choices[0].message.content.strip()
            if "[IMPROVE]" not in reflection and "[NO_ISSUE]" not in reflection:
                logging.warning(f"Reflection missing keywords: {reflection}")
                return f"{reflection} [IMPROVE]"
            return reflection
        except Exception as e:
            logging.error(f"Error in reflection generation: {e}")
            return "[IMPROVE]"

    def process_article(self, input_file: str, output_file: str, topic: str, proposal: str, max_image_num: int = 2, max_iterations: int = 3) -> List[Dict[str, str]]:
        sections = self._read_sections(input_file)
        all_suggestions = []
        total_subsections = sum(len(section.split('## ')) for section in sections if section.strip())

        with tqdm(total=total_subsections, desc="Processing Subsections") as pbar:
            for idx, section in enumerate(sections, 1):
                section_name = section.split(" - ")[1] if " - " in section else ""
                subsections = [s.strip() for s in section.split('## ') if s.strip()]
                
                for sub_idx, subsection in enumerate(subsections, 1):
                    subsection_name = subsection.split(" - ")[1] if " - " in subsection else ""
                    suggestions = []
                    
                    for _ in range(max_iterations):
                        current_suggestions = self.analyze_section(subsection, max_image_num)
                        if not current_suggestions:
                            break
                            
                        for suggestion in current_suggestions:
                            suggestion.update({
                                "Section": f"Section {idx} - {section_name}",
                                "Subsection": f"Subsection {sub_idx} - {subsection_name}"
                            })
                        suggestions = current_suggestions
                        
                        supervisor_feedback = self.monitor.forward(
                            topic=topic,
                            proposal=proposal,
                            section_content=json.dumps(suggestions)
                        )
                        logging.info(f"Supervisor feedback for Section {idx} Subsection {sub_idx}: {supervisor_feedback}")
                        
                        reflection = self.generate_reflection(suggestions, subsection, supervisor_feedback)
                        logging.info(f"Reflection for Section {idx} Subsection {sub_idx}: {reflection}")
                        
                        if "[NO_ISSUE]" in reflection:
                            break
                        elif "[IMPROVE]" in reflection:
                            time.sleep(1)  # 添加短暂延迟，避免立即重试
                            continue
                        else:
                            logging.error(f"Unexpected reflection format: {reflection}")
                            break
                    
                    all_suggestions.extend(suggestions)
                    pbar.update(1)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_suggestions[:15], f, ensure_ascii=False, indent=4)
        
        return all_suggestions[:15]

    @staticmethod
    def _read_sections(file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [section.strip() for section in f.read().split('# ') if section.strip()]
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return []

