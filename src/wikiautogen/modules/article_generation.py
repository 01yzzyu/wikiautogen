import concurrent.futures
import copy
import logging
import os
from concurrent.futures import as_completed
from typing import List, Union

import dspy
from openai import OpenAI, OpenAIError

from .callback import BaseCallbackHandler
from .wikiautogen_dataclass import InformationTableGen, WikiArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


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


class SuperviseDialogue(dspy.Signature):
    """You are a helpful monitor, used to supervise the work of other agents. 
    Your task is to answer questions by logically decomposing them into clear sub-questions and iteratively addressing each one. 
    Judge whether the problem has been solved, whether the depth and breadth of the discussion are sufficient, and the role of the wrong supervisor agent, 
    mainly strictly adjust the discussion direction of multiple agents according to the topic and proposal we input. 
    Use "Follow up:" to introduce each sub-question and "Intermediate answer:" to provide answers.
    For each sub-question, decide whether you can provide a direct answer or if additional information is required. 
    If additional information is needed, state, "Let’s search the question in Wikipedia." and then use the retrieved information to respond comprehensively. 
    If a direct answer is possible, provide it immediately without searching."""

    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    proposal = dspy.InputField(prefix="Proposal for research: ", format=str)
    section_content = dspy.InputField(prefix="Generated section content: ", format=str)
    feedback = dspy.OutputField(format=str)


class ArticleGenerationModule(ArticleGenerationModule):
    """
    Interface for article generation stage. Given a topic, collected information from the knowledge curation stage,
    and a generated outline from the outline generation stage, this module generates an article.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
        monitor_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel] = None
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)
        self.conversation_monitor = ConversationMonitor(engine=monitor_engine)
        self.openai_client = self._init_openai_client()

    def _init_openai_client(self) -> OpenAI:
        """Initialize OpenAI client with environment variables."""
        try:
            return OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_ENDPOINT")
            )
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate_section(
        self,
        topic: str,
        proposal: str,
        section_name: str,
        information_table: InformationTableGen,
        section_outline: str,
        section_query: str,
    ):
        collected_info: List[Information] = (
            information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            ) if information_table else []
        )
        output = self.section_gen(
            topic=topic,
            proposal=proposal,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        section_content = output.section

        for _ in range(3):
            feedback = self.conversation_monitor(topic=topic, proposal=proposal, section_content=section_content)
            logging.info(f"Section {section_name} monitor feedback: {feedback}")

            reflections = self.generate_reflection(topic, proposal, section_content, feedback)
            should_continue = False
            for reflection in reflections:
                logging.info(f"Section {section_name} reflection: {reflection}")
                if reflection and ("improve" in reflection.lower() or "issue" in reflection.lower()):
                    proposal = self._update_proposal(reflection, proposal)
                    should_continue = True
            if not should_continue:
                break

            output = self.section_gen(
                topic=topic,
                proposal=proposal,
                outline=section_outline,
                section=section_name,
                collected_info=collected_info,
            )
            section_content = output.section

        return {
            "section_name": section_name,
            "section_content": section_content,
            "collected_info": collected_info,
        }

    def _update_proposal(self, reflection: str, proposal: str) -> str:
        """Update proposal based on reflection feedback."""
        updates = {
            "Uncommon information": "Look for uncommon information",
            "Initial goal" "not tied back": "Re-align with initial goal",
            "Off - topic indicators": "Remove off - topic content",
            "Audience appeal" "too technical": "Use more accessible language",
            "Storytelling elements": "Incorporate storytelling elements",
            "Logical flow" "disjointed": "Improve logical flow",
            "Transitional phrases": "Use transitional phrases",
            "Citation sources": "Emphasize citation sources",
            "Fact - checking process": "Mention fact - checking process",
            "Term usage": "Ensure consistent term usage",
            "Idea consistency": "Check idea consistency",
            "User's goals" "not contributing": "Refocus on user's goals",
            "Feedback loop": "Establish feedback loop",
            "Redundant statements": "Avoid redundant statements",
            "Summarization": "Summarize effectively"
        }
        for key, value in updates.items():
            if all(k in reflection for k in key.split()):
                proposal += f" {value}"
        return proposal.strip()

    def generate_reflection(self, topic: str, proposal: str, section_content: str, feedback: str, pair_num: int = 1, model: str = 'o3-mini') -> List[str]:
        reflection_prompt = f"""From a writer's perspective, self-reflect on the article-generation process based on:
            Topic: {topic}
            Proposal: {proposal}
            Section Content: {section_content}
            Monitor's Feedback: {feedback}
            Problem Analysis:
            - Coherence: Ensure logical connection and smooth flow.
            - Audience-Appropriateness: Check if content suits the audience without being overly complex.
            - Topic Exploration: Evaluate if topic exploration fits audience and intent.
            Improvement:
            - Coherence: Rearrange sentences, add transitions.
            - Audience: Simplify concepts, use relatable language.
            - Topic: Refine based on audience needs.
            Info Reliability:
            - Sources: Review credibility.
            - Accuracy: Verify facts."""
        try:
            response = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": reflection_prompt}],
                model=model,
                n=pair_num
            )
            return [str(c.message.content) for c in response.choices]
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error in reflection: {e}")
            return []

    def generate_article(
        self,
        topic: str,
        proposal: str,
        information_table: InformationTableGen,
        article_with_outline: WikiArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> WikiArticle:
        information_table.prepare_table_for_retrieval()
        article_with_outline = article_with_outline or WikiArticle(topic_name=topic)
        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if not sections_to_write:
            logging.error(f"No outline for {topic}. Using topic as query.")
            section_output_dict_collection.append(
                self.generate_section(topic, proposal, topic, information_table, "", [topic])
            )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    if section_title.lower().strip() in ("introduction",) or section_title.lower().strip().startswith(("conclusion", "summary")):
                        continue
                    section_query = article_with_outline.get_outline_as_list(root_section_name=section_title, add_hashtags=False)
                    section_outline = "\n".join(article_with_outline.get_outline_as_list(root_section_name=section_title, add_hashtags=True))
                    future_to_sec_title[executor.submit(
                        self.generate_section, topic, proposal, section_title, information_table, section_outline, section_query
                    )] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                parent_section_name=topic,
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        article.post_processing()
        return article


class ConvToSection(dspy.Module):
    """Use collected information to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(self, topic: str, proposal: str, outline: str, section: str, collected_info: List[Information]):
        info = "\n\n".join(f"[{idx + 1}]\n" + "\n".join(info.snippets) for idx, info in enumerate(collected_info))
        info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)

        with dspy.settings.context(lm=self.engine):
            section_output = self.write_section(topic=topic, proposal=proposal, info=info, section=section).output
            section = ArticleTextProcessing.clean_up_section(section_output)

        return dspy.Prediction(section=section)


class WriteSection(dspy.Signature):
    """Write a Wikipedia section based on collected information and proposal.
    Format: Use "#" for section titles, "##" for subsections, etc., and [1], [2], etc., for inline citations."""

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    proposal = dspy.InputField(prefix="The proposal for the article: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start with # section title):\n",
        format=str,
    )