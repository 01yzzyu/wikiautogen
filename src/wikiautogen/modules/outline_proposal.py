import os
import time
import json
import threading
import requests
import collections
from typing import Set, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI, OpenAIError


class WikipediaProposalGenerator:
    def __init__(self, openai_api_key: str, serper_api_key: str, ner_model: str = "dslim/bert-large-NER"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_ENDPOINT"] = "https://api.openai.com/v1"
        self.serper_api_key = serper_api_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model).to(self.device)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device)
        self.id2query = {}
        self.id2img = set()
        self.thread_local = threading.local()
        self.thread_local.lock = threading.Lock()
        self.serper_api_url = "https://google.serper.dev/lens"
        # print(f"Using OpenAI API Key: {openai_api_key[:10]}...")
        # print(f"Using Serper API Key: {serper_api_key[:10]}...")

    def _gpt_text_only(self, text: str, pair_num: int = 1, temperature: float = 0.4, model: str = "gpt-4o") -> list:
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_ENDPOINT"])
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": text}],
                model=model,
                temperature=temperature,
                n=pair_num if pair_num > 1 else 1
            )
            return [str(choice.message.content) for choice in response.choices]
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def _parse_ner_res(self, query: str, ner_res: list) -> list:
        ret_list = []
        save_start_idx, save_end_idx, last_type = 0, 0, ""
        for info in ner_res:
            ent_type, start_idx, end_idx = info["entity"], info["start"], info["end"]
            prefix, suffix = ent_type.split("-") if "-" in ent_type else ("O", "O")
            if prefix in ["B", "O"] and last_type:
                ret_list.append({"entity": query[save_start_idx:save_end_idx], "type": last_type})
                save_start_idx = start_idx
            save_end_idx = end_idx if prefix == "I" else save_end_idx
            last_type = suffix
        if last_type:
            ret_list.append({"entity": query[save_start_idx:save_end_idx], "type": last_type})
        return ret_list

    def _gen_query_by_gpt(self, query: str) -> list:
        prompt = (
        "You are an expert in Wikipedia article curation and knowledge organization. Given the user query: \"{query}\". "
        "Your task is to: "
        "1. Identify the main subject(s) in the query. Do not provide a direct answer to the query itself. "
        "2. Recommend several closely related topics or Wikipedia pages that share significant relevance with the query subject. These topics should offer: "
        "   - Illustrative examples; "
        "   - Notable aspects commonly associated with the main subject; "
        "   - Insights into how Wikipedia typically structures articles within this domain. "
        "3. Format your response: Put all recommended topics on a single line, separate each topic by a comma, and do not include additional explanation or context—only the topic names or concise references. "
        "Constraints: Do not reveal the direct answer to the main query. Avoid repeating the user's query verbatim, except when briefly referencing it in your reasoning. "
        "Present only the final list of recommended topics as your output."
        ).format(query=query)
        res_list = self._gpt_text_only(prompt, temperature=0.3, pair_num=1, model="gpt-4o")
        return res_list[0].split(', ') if res_list else []

    def _retry_operation(self, operation, max_attempts: int = 3, delay: int = 1) -> Set[str]:
        for attempt in range(max_attempts):
            try:
                return operation()
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error occurred: {e}")
                if attempt == max_attempts - 1:
                    print("Max attempts reached.")
                else:
                    time.sleep(delay)
        return set()

    def extract_q_query(self, query: str) -> Set[str]:
        def extract():
            ner_set = {x['entity'].lower() for x in self._parse_ner_res(query, self.nlp(query))}
            gpt_set = {x.lower() for x in self._gen_query_by_gpt(query)}
            return ner_set.union(gpt_set)
        return self._retry_operation(extract)

    def _google_visual_search(self, img_url: str) -> dict:
        payload = json.dumps({"url": img_url})
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
        response = requests.post(self.serper_api_url, headers=headers, data=payload)
        if response.status_code != 200:
            return {"error": f"Request failed with status code {response.status_code}"}
        result = response.json()
        search_str = ', '.join(item.get('title', '') for item in result.get('organic', []))
        return {'search_str': search_str, 'brq_list': [], 'entity_list': [], 'related_list': [], 'url_title_list': []}

    def extract_v_query(self, img_url: str) -> Set[str]:
        def extract():
            res = self._google_visual_search(img_url)
            search_str = res['search_str']
            entities = [x['entity'].lower() for x in self._parse_ner_res(search_str, self.nlp(search_str))]
            top_20_entities = [entity for entity, _ in collections.Counter(entities).most_common(20)]
            return set(top_20_entities)
        return self._retry_operation(extract)

    def extract_mm_query(self, img_url: str, text_queries: Set[str]) -> Set[str]:
        prompt = (
        "You are an expert in Wikipedia article curation and knowledge organization. You are provided with an image and a set of textual queries extracted from a user query. "
        "Image URL: {img_url}. "
        "Textual queries: {text_queries}. "
        "Analyze the given image and the provided queries, then define a set of query keywords that describe the content conveyed by both the image and the queries. "
        "Provide your answer as a single line of comma - separated keywords without additional commentary."
        ).format(img_url=img_url, text_queries=', '.join(text_queries))
        res_list = self._gpt_text_only(prompt, temperature=0.9, pair_num=1, model="gpt-4o")
        return {kw.strip() for kw in res_list[0].split(',') if kw.strip()} if res_list else set()

    def _get_first_title_entity(self, img_url: str) -> Optional[str]:
        def extract():
            search_result = self._google_visual_search(img_url)
            ner_res = self._parse_ner_res(search_result['search_str'], self.nlp(search_result['search_str']))
            return ner_res[0]['entity'] if ner_res else None
        return self._retry_operation(extract)

    def rewrite_with_gpt4o(self, combined_queries: Set[str], query: str) -> Optional[str]:
        prompt = (
        "You are a seasoned Wikipedia editorial expert, specializing in content curation and knowledge organization. Based on the following combined queries: {combined_queries}. "
        "Given the specific topic \"{query}\", generate a detailed, Wikipedia-like article proposal, each section needs to have a clearly explored query or direction "
        "Your proposal should be organized into clearly defined sections with concise, descriptive paragraphs—avoid using bullet points entirely. "
        "Ensure the response strictly adheres to this format: "
        "Wikipedia Article Proposal: [your complete proposal here]"
        ).format(combined_queries=', '.join(combined_queries), query=query)
        res_list = self._gpt_text_only(prompt, temperature=0.9, pair_num=1, model="gpt-4o")
        print(f"GPT-4o response for proposal: {res_list}")
        return res_list[0].split("Wikipedia Article Proposal: ")[1].strip() if res_list and "Wikipedia Article Proposal: " in res_list[0] else None

    def generate_proposal(self, img_url: Optional[str] = None, query: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        print(f"Starting with img_url: {img_url}, query: {query}")
        q_queries, v_queries, mm_queries, combined_queries, proposal = set(), set(), set(), set(), None

        if query:
            query_id = "01"
            with self.thread_local.lock:
                q_queries = self.extract_q_query(query) if query_id not in self.id2query else set(self.id2query[query_id])
                self.id2query[query_id] = q_queries

        if img_url:
            v_queries = self.extract_v_query(img_url)
            first_title_entity = self._get_first_title_entity(img_url)
            if first_title_entity:
                print(f"Topic from image: {first_title_entity}")
                query = query or first_title_entity

        if img_url and query:
            mm_queries = self.extract_mm_query(img_url, q_queries)
            combined_queries = v_queries.union(mm_queries)
        elif img_url:
            combined_queries = v_queries
        elif query:
            combined_queries = q_queries

        if query:
            proposal = self.rewrite_with_gpt4o(combined_queries, query)

        if query:
            print(f"Text Extracted Queries: {q_queries}")
        print(f"Image Extracted Queries: {v_queries}")
        if query and img_url:
            print(f"Multimodal Extracted Queries: {mm_queries}")
        print(f"All Extracted Queries: {combined_queries}")
        if query:
            print(f"Topic: {query}")
        if proposal:
            print(f"Wikipedia Article Proposal: {proposal}")
        else:
            print("No proposal generated.")

        return query, proposal
