import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional
import dspy

from .modules.article_generation import ArticleGenerationModule
from .modules.article_polish import PolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_exploration import KnowledgeExplorationModule
from .modules.outline_generation import OutlineGenerationModule
from .modules.persona_generator import PersonaGenerator
from .modules.wikiautogen_dataclass import InformationTableGen, WikiArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import LitellmModel
from ..utils import FileIOHelper, makeStringRed, truncate_filename, process_files, generate_html_and_markdown, merge_text_files
from .modules.position import MMAnalyzer
from .modules.image_retrieval import ImageRetriever
from .modules.mm_polish import MMPolishModule

class WikiLMConfigs(LMConfigs):
    def __init__(self):
        self.conv_simulator_lm = None
        self.question_asker_lm = None
        self.outline_gen_lm = None
        self.article_gen_lm = None
        self.article_polish_lm = None

    def init_openai_model(
        self,
        openai_api_key: str,
        azure_api_key: str,
        openai_type: Literal["openai", "azure"],
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }

        openai_kwargs = {
            "api_key": openai_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }
        if openai_type and openai_type == "openai":
            self.conv_simulator_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.outline_gen_lm = LitellmModel(
                model="gpt-4-0125-preview", max_tokens=400, **openai_kwargs
            )
            self.article_gen_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **openai_kwargs
            )
            self.article_polish_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **openai_kwargs
            )
        elif openai_type and openai_type == "azure":
            self.conv_simulator_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **azure_kwargs,
                model_type="chat",
            )
            self.outline_gen_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=400, **azure_kwargs, model_type="chat"
            )
            self.article_gen_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **azure_kwargs,
                model_type="chat",
            )
            self.article_polish_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **azure_kwargs,
                model_type="chat",
            )
        else:
            logging.warning(
                "No valid OpenAI API provider is provided. Cannot use default LLM configurations."
            )

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model

@dataclass
class RunnerArguments:
    """Arguments for controlling WikiAutoGen pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={
            "help": "Maximum number of questions in conversational question asking."
        },
    )
    max_perspective: int = field(
        default=3,
        metadata={
            "help": "Maximum number of perspectives to consider in perspective-guided question asking."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )

class Runner(Engine):
    def __init__(
        self, args: RunnerArguments, lm_configs: WikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        persona_generator = PersonaGenerator(self.lm_configs.question_asker_lm)
        self.knowledge_curation_module = KnowledgeExplorationModule(
            retriever=self.retriever,
            persona_generator=persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
            monitor_engine=self.lm_configs.question_asker_lm
        )
        self.outline_generation_module = OutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.article_generation = ArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
            monitor_engine=self.lm_configs.article_gen_lm
        )
        self.article_polishing_module = PolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> InformationTableGen:
        information_table, conversation_log = (
            self.knowledge_curation_module.research(
            topic=self.topic,
            proposal=self.proposal,
            ground_truth_url=ground_truth_url,
            callback_handler=callback_handler,
            max_perspective=self.args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True,
        )
        )

        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )
        return information_table

    def run_outline_generation_module(
        self,
        information_table: InformationTableGen,
        callback_handler: BaseCallbackHandler = None,
    ) -> WikiArticle:
        outline, draft_outline = self.outline_generation_module.generate_outline(
            topic=self.topic,
            proposal=self.proposal,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "gen_outline.txt")
        )
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        )
        return outline

    def run_article_generation_module(
        self,
        outline: WikiArticle,
        information_table: InformationTableGen,
        callback_handler: BaseCallbackHandler = None,
    ) -> WikiArticle:
        draft_article = self.article_generation.generate_article(
            topic=self.topic,
            proposal=self.proposal,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )

        draft_article.dump_article_as_plain_text(
            os.path.join(self.article_output_dir, "textual_article_unpolish.txt")
        )
        
        draft_article.dump_reference_to_file(
            os.path.join(self.article_output_dir, "url_to_info.json")
        )
        return draft_article

    def run_article_polishing_module(
        self, draft_article: WikiArticle, remove_duplicate: bool = False
    ) -> WikiArticle:
        polished_article = self.article_polishing_module.polish_article(
            topic=self.topic,
            proposal=self.proposal,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )
        FileIOHelper.write_str(
            polished_article.to_string(),
            os.path.join(self.article_output_dir, "textual_article.txt"),
        )
        return polished_article

    def post_run(self):
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(
            os.path.join(self.article_output_dir, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in llm_call_history:
                if "kwargs" in call:
                    call.pop("kwargs")
                f.write(json.dumps(call) + "\n")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic."
        )
        return InformationTableGen.from_conversation_log_file(
            information_table_local_path
        )

    def _load_outline_from_local_fs(self, topic, outline_local_path):
        assert os.path.exists(outline_local_path), makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the gen_outline.txt for this topic."
        )
        return WikiArticle.from_outline_file(topic=topic, file_path=outline_local_path)

    def _load_draft_article_from_local_fs(
        self, topic, draft_article_path, url_to_info_path
    ):
        assert os.path.exists(draft_article_path), makeStringRed(
            f"{draft_article_path} not exists. Please set --do-generate-article argument to prepare the textual_article_unpolish.txt for this topic."
        )
        assert os.path.exists(url_to_info_path), makeStringRed(
            f"{url_to_info_path} not exists. Please set --do-generate-article argument to prepare the url_to_info.json for this topic."
        )
        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)
        return WikiArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    def run(
        self,
        og_topic: str,
        topic: str,
        proposal: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        self.topic = topic
        self.proposal = proposal
        self.og_topic = og_topic
        self.article_dir_name = truncate_filename(
            og_topic.replace(" ", "_").replace("/", "_")
        )
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )
        os.makedirs(self.article_output_dir, exist_ok=True)

        information_table: InformationTableGen = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )

        outline: WikiArticle = None
        if do_generate_outline:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        draft_article: WikiArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(
                        self.article_output_dir, "gen_outline.txt"
                    ),
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )
        
        if do_polish_article:
            if draft_article is None:
                draft_article_path = os.path.join(
                    self.article_output_dir, "textual_article_unpolish.txt"
                )
                url_to_info_path = os.path.join(
                    self.article_output_dir, "url_to_info.json"
                )
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path,
                )
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )

    def mmrun(
        self,
        topic: str,
        proposal: str,
        do_positing: bool = True,
        do_Retrieve_images: bool = True,
        do_mmpolish: bool = True,
        remove_duplicate: bool = False,
        openai_api_key: str = None,
    ):

        article_file_path = os.path.join(self.article_output_dir, "textual_article.txt")
        output_file = os.path.join(self.article_output_dir, "image_suggestions.json")
        input_json = os.path.join(self.article_output_dir, "url_to_info.json")
        output_txt = os.path.join(self.article_output_dir, "article_with_image_placeholders.txt")
        reference_file_path = os.path.join(self.article_output_dir, "references.txt")
        output_file_path = os.path.join(self.article_output_dir, "images")
        output_final_path = os.path.join(self.article_output_dir, "final_article.txt")
        md_file_path = os.path.join(self.article_output_dir, "final_article.md")
        output_file_path_md_multi = os.path.join(self.article_output_dir, "final_article_multimodal.md")

        if do_positing:
            analyzer = MMAnalyzer(api_key=openai_api_key)
            analyzer.process_article(
                topic=topic,
                proposal=proposal,
                input_file=article_file_path,
                output_file=output_file,
                max_image_num=1,
                max_iterations=3
            )

            # Process files to incorporate image suggestions
            process_files(
                input_json_file=input_json,
                image_suggestions_json=output_file,
                original_text_file=article_file_path,
                output_txt_file=output_txt,
                references_txt_file=reference_file_path
            )

        if do_Retrieve_images:
        # Retrieve and insert images
            processor = ImageRetriever(openai_api_key=openai_api_key)
            processor.process_article(
                input_file_path=output_txt,
                article_file_path=article_file_path,
                output_folder=output_file_path
            )

            # Merge article and references
            merge_text_files(
                file1_path=article_file_path,
                file2_path=reference_file_path,
                output_path=output_final_path
            )

            # Generate HTML and Markdown outputs
            generate_html_and_markdown(output_final_path, output_txt, output_file_path)

        # Step 3: Multimedia Polishing
        if do_mmpolish:
            MMPolish = MMPolishModule(
                article_gen_lm=self.lm_configs.article_gen_lm,
                article_polish_lm=self.lm_configs.article_polish_lm,
                monitor_engine=self.lm_configs.article_polish_lm,
                openai_api_key=openai_api_key
            )
            output_path = MMPolish.polish_md_file(
                md_file_path=md_file_path,
                topic=topic,
                proposal=proposal,
                output_file_path=output_file_path_md_multi,
                remove_duplicate=remove_duplicate
            )

        # Finalize by saving logs
        self.post_run()
