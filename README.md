# WikiAutoGen - Automated Wikipedia Content Generation System
<a href='https://wikiautogen.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://wikiautogen.github.io/WikiAutoGen/'><img src='https://img.shields.io/badge/Paper-Example-blue'></a> 

Zhongyu Yang<sup>1, 2*</sup>, Jun Chen<sup>1*</sup>, Dannong Xu<sup>1,3</sup>, Junjie Fei<sup>1</sup>, Xiaoqian Shen<sup>1</sup>, Liangbing Zhao<sup>1</sup>, Chun-Mei Feng<sup>4</sup>, Mohamed Elhoseiny<sup>1</sup>

<sup>1</sup>King Abdullah University of Science and Technology, 
<sup>2</sup>Lanzhou University, 
<sup>3</sup>The University of Sydney, 
<sup>4</sup>IHPC, A*STAR

# 👀 The code is coming soon...

## Installation

```bash
git clone https://github.com/wikiautogen/wikiautogen_code.git
cd wikiautogen
conda create -n wikiautogen python=3.11
conda activate wikiautogen
pip install -r requirements.txt
```

## Key Features
- 🖼️ **Multimodal Content Generation** with image-aware topic proposal
- 🤖 **Automated Research** using search engines (Serper/You.com)
- 📝 **Structured Writing** with outline generation and article polishing
- 🔍 **Fact Verification** through multi-perspective conversation simulation

## Quick Start

### 1. Environment Setup
```bash
export OPENAI_API_KEY="your_openai_key"
export SERPER_API_KEY="your_serper_key"
```

### 2. Process Topics with Images
```python
from src import RunnerArguments, Runner, WikiLMConfigs
from src.lm import OpenAIModel
from src.rm import SerperRM
from src.wikiautogen.modules.outline_proposal import WikipediaProposalGenerator

generator = WikipediaProposalGenerator(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    serper_api_key=os.getenv("SERPER_API_KEY")
)

lm_configs = WikiLMConfigs()
gpt_4o_mini = OpenAIModel(
    model='gpt-4o-mini',
    temperature=0.9,
    api_key=os.getenv("OPENAI_API_KEY")
)
lm_configs.set_all_models(gpt_4o_mini)

runner = Runner(
    RunnerArguments(
        output_dir="./output",
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=10
    ),
    lm_configs,
    SerperRM(
        serper_search_api_key=os.getenv('SERPER_API_KEY'),
        query_params={"num": 10}
    )
)

topic = input('Topic: ')
img_url = input('Image: ')

new_topic, proposal = generator.generate_proposal(img_url, topic)

runner.run(
    og_topic=topic,
    topic=new_topic,
    proposal=proposal,
    do_research=True,
    do_generate_outline=True,
    do_generate_article=True,
    do_polish_article=True
)


runner.mmrun(
    og_topic=topic,
    topic=new_topic,
    proposal=proposal,
    do_positing=True,
    do_Retrieve_images=True,
    do_mmpolish=True
)


```

- `do_research`: if True, simulate conversations with difference perspectives to collect information about the topic; otherwise, load the results.
- `do_generate_outline`: if True, generate an outline for the topic; otherwise, load the results.
- `do_generate_article`: if True, generate an article for the topic based on the outline and the collected information; otherwise, load the results.
- `do_polish_article`: if True, polish the article by adding a summarization section and (optionally) removing duplicate content; otherwise, load the results.
- `do_positing`: if True, generate a positioning proposal for the article; otherwise, load the results.
- `do_Retrieve_images`: if True, generate a multimodal article for the topic based on the positioning proposal and the collected information; otherwise, load the results.
- `do_mmpolish`: if True, polish the article by enhancing coherence and consistency across modalities, focusing on potential discrepancies between textual content and visual figures. 



## License
This project is licensed under the MIT License. Content generation based on Wikipedia data follows CC BY-SA guidelines.

## Citation
```bibtex
@misc{yang2025wikiautogenmultimodalwikipediastylearticle,
      title={WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation}, 
      author={Zhongyu Yang and Jun Chen and Dannong Xu and Junjie Fei and Xiaoqian Shen and Liangbing Zhao and Chun-Mei Feng and Mohamed Elhoseiny},
      year={2025},
      eprint={2503.19065},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19065}, 
}
```

## 👍 Acknowledgement
WikiAutoGen is built with reference to the following outstanding works: [Storm](https://github.com/stanford-oval/storm), [Co-storm](https://github.com/stanford-oval/storm), [Dspy](https://github.com/stanfordnlp/dspy).
Thanks！
