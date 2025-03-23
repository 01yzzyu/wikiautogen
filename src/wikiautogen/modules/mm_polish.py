import copy
import argparse
import copy
import re
import json
import dspy
import os
import openai
from .wikiautogen_dataclass import WikiArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing
from typing import Union, List
from ...lm import OpenAIModel
import concurrent.futures
import logging
from concurrent.futures import as_completed
from openai import OpenAI, OpenAIError

COMMON_PROMPT_TEMPLATE = """You are an expert at interweaving original answers with additional data presented in text - format. 
Here is the potential additional data and some surrounding context that can be revised and integrated into the original answer. 

To create a coherent and informative response, please follow these guidelines:
1. Based on the context in the additional data, re - phrase the details of the original answer to ensure a smooth flow of the final response.
2. Do not repeat information that has already been mentioned in the final response, including images.
3. To maintain coherence, use introductory phrases such as "as shown in the image/video/table below", "see the image/video/table below", "refer to the following image/video/table", or "as illustrated in the image/video/table below". Vary these phrases to ensure a natural flow and avoid repetition.
4. Feel free to re - position the additional data if there is a more suitable location based on the context in the original answer.
5. Preserve the original format of content in HTML tags or Markdown style.
6. Avoid including hallucinated content such as <Placeholder>.
7. Your response must be well - formatted, easily readable, and understandable by humans.

{specific_instructions}"""

class SuperviseDialogue(dspy.Signature):
    """You are a helpful Retrieve AugmentedGeneration model, used to supervise the work of other agents."""
    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    proposal = dspy.InputField(prefix="Proposal for research: ", format=str)
    section_content = dspy.InputField(prefix="Generated section content: ", format=str)
    feedback = dspy.OutputField(format=str)

class WriteLeadSection(dspy.Signature):
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    proposal = dspy.InputField(prefix="The proposal for the article: ", format=str)
    draft_page = dspy.InputField(prefix="The draft page:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)

class PolishPage(dspy.Signature):
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    proposal = dspy.InputField(prefix="The proposal for the article: ", format=str)
    draft_page = dspy.InputField(prefix="The draft article:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article:\n", format=str)

class MMPolishModule(ArticlePolishingModule):
    """A module for polishing Markdown articles with multimodal support."""
    
    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        monitor_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        openai_api_key: str = None,
        max_iterations: int = 3,
        num_intro_phrases: int = 4
    ):
        super().__init__()
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm
        self.monitor_engine = monitor_engine
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.max_iterations = max_iterations
        self.num_intro_phrases = num_intro_phrases
        
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)
        self.monitor_dialogue = dspy.Predict(SuperviseDialogue)
        self.client = openai.OpenAI(api_key=self.openai_api_key)

    def _generate_intro_phrases(self, temperature: float = 1.0, top_p: float = 0.9) -> List[str]:
        """Generate introductory phrases for multimodal elements."""
        specific_instructions = f"Please generate {self.num_intro_phrases} different introductory phrases for integrating images into a text."
        prompt = COMMON_PROMPT_TEMPLATE.format(specific_instructions=specific_instructions)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p
            )
            return [phrase.strip() for phrase in response.choices[0].message.content.split('\n') if phrase.strip()]
        except Exception as e:
            logging.error(f"Error generating intro phrases: {e}")
            return ["as shown in the image below", "see the image below", "refer to the following image", "as illustrated in the image below"]

    def _generate_reflection(self, topic: str, proposal: str, section_content: str, feedback: str, model: str = 'o3-mini') -> List[str]:
        """Generate reflection on the article content."""
        reflection_prompt = f"""From a reader's perspective, reflect on the multimodal-generation process:
        Topic: {topic}
        Proposal: {proposal}
        Article Content: {section_content}
        Monitoring Feedback: {feedback}
        Analyze comprehension, visual-textual integration, readability, and suggest improvements."""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": reflection_prompt}],
                n=1
            )
            return [str(response.choices[0].message.content)]
        except Exception as e:
            logging.error(f"Error in generate_reflection: {e}")
            return []

    def polish_article(
        self, topic: str, proposal: str, draft_article: Union[str, WikiArticle], remove_duplicate: bool = False
    ) -> WikiArticle:
        """Polish the article with iterative refinement."""
        article_text = draft_article.to_string() if isinstance(draft_article, WikiArticle) else draft_article
        iteration = 0

        while iteration < self.max_iterations:
            # Generate lead section
            lead_prompt = COMMON_PROMPT_TEMPLATE.format(specific_instructions=f"Topic: {topic}\nProposal: {proposal}\nDraft: {article_text}\nWrite the lead section:")
            with dspy.settings.context(lm=self.article_gen_lm, show_guidelines=False):
                lead_section = self.write_lead(topic=topic, proposal=proposal, draft_page=article_text).lead_section
                if "The lead section:" in lead_section:
                    lead_section = lead_section.split("The lead section:")[1].strip()

            # Polish the article
            polish_prompt = COMMON_PROMPT_TEMPLATE.format(specific_instructions=f"Topic: {topic}\nProposal: {proposal}\nDraft: {article_text}\nYour revised article:")
            with dspy.settings.context(lm=self.article_polish_lm, show_guidelines=False):
                polished_text = self.polish_page(topic=topic, proposal=proposal, draft_page=article_text).page if remove_duplicate else article_text

            polished_article = f"# summary\n{lead_section}\n\n{polished_text}"

            # Monitor and reflect
            with dspy.settings.context(lm=self.monitor_engine):
                feedback = self.monitor_dialogue(topic=topic, proposal=proposal, section_content=polished_article).feedback
            logging.info(f"Article monitor feedback: {feedback}")

            reflections = self._generate_reflection(topic, proposal, polished_article, feedback)
            need_improvement = any("improve" in r.lower() or "issue" in r.lower() for r in reflections)
            
            if not need_improvement:
                break
                
            for reflection in reflections:
                logging.info(f"Article reflection: {reflection}")
                if "Uncommon information" in reflection:
                    proposal += " Look for uncommon information"
                # Add other reflection-based proposal updates as needed (simplified here for brevity)

            iteration += 1

        polished_dict = ArticleTextProcessing.parse_article_into_dict(polished_article)
        result = copy.deepcopy(draft_article) if isinstance(draft_article, WikiArticle) else WikiArticle.from_string(topic, polished_article)
        result.insert_or_create_section(article_dict=polished_dict)
        result.post_processing()
        return result

    def polish_md_file(
        self, md_file_path: str, topic: str, proposal: str, output_file_path: str, 
        remove_duplicate: bool = False, temperature: float = 1.0, top_p: float = 0.9
    ) -> str:
        """Polish a Markdown file and save the result."""
        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        multimodal_data = list(set(re.findall(r'!\[.*?\]\(.*?\)', md_content)))
        references_file = os.path.join(os.path.dirname(md_file_path), 'url_to_info.json')
        references = json.load(open(references_file, 'r', encoding='utf-8')) if os.path.exists(references_file) else {"url_to_unified_index": {}, "url_to_info": {}}

        draft_article = WikiArticle.from_string(topic_name=topic, article_text=md_content, references=references)
        polished_article = self.polish_article(topic, proposal, draft_article, remove_duplicate)

        intro_phrases = self._generate_intro_phrases(temperature, top_p)
        polished_text = polished_article.to_string()
        new_text = ""
        phrase_index = 0

        for line in polished_text.splitlines():
            for data in multimodal_data:
                if data in line:
                    new_text += f"{intro_phrases[phrase_index % len(intro_phrases)]}: {data}\n"
                    phrase_index += 1
                    break
            else:
                new_text += line + "\n"

        new_text = re.sub(r'\d+\. ', '', new_text)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(new_text)

        return output_file_path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Polish a Markdown file.')
    parser.add_argument('--original_text_file_path', type=str, default="./results/gpt/218_selfdoubt_monitor_o3_gen/2020_United_States_presidential_election/final_article.md", help='Path to the original text file.')
    parser.add_argument('--output_file_path', type=str, default="final_article_mmpolish.md", help='Path to the output file.')
    parser.add_argument('--topic', type=str, default='2020_United_States_presidential_election', help='The topic of the article.')
    parser.add_argument('--proposal', type=str, default=None, help='The proposal for the article.')
    parser.add_argument('--openai_api_key', type=str, default="YOUR_OPENAI_KEY", help='OpenAI API key.')
    parser.add_argument('--remove_duplicate', action='store_true', help='Whether to use one additional LM call to remove duplicates from the article.')
    args = parser.parse_args()

    openai_kwargs = {
        'api_key': args.openai_api_key,
        'temperature': 1.0,
        'top_p': 0.9,
    }

    gpt_4 = OpenAIModel(model='gpt-4o', max_tokens=3500, **openai_kwargs)

    output_path = polish_md_file_multi(
        md_file_path=args.original_text_file_path,
        topic=args.topic,
        proposal=args.proposal,
        article_gen_lm=gpt_4,
        article_polish_lm=gpt_4,
        monitor_engine=gpt_4,
        output_file_path=args.output_file_path,
        remove_duplicate=args.remove_duplicate,
        num_intro_phrases=4,
        openai_kwargs=openai_kwargs
    )

    print(f"Polished Markdown saved to {output_path}")

                