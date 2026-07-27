# WikiAutoGen - 自动化维基百科内容生成系统
<a href='https://wikiautogen.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://wikiautogen.github.io/WikiAutoGen/'><img src='https://img.shields.io/badge/Paper-Example-blue'></a> 

Zhongyu Yang<sup>1, 2*</sup>, Jun Chen<sup>1*</sup>, Dannong Xu<sup>1,3</sup>, Junjie Fei<sup>1</sup>, Xiaoqian Shen<sup>1</sup>, Liangbing Zhao<sup>1</sup>, Chun-Mei Feng<sup>4</sup>, Mohamed Elhoseiny<sup>1</sup>

<sup>1</sup>King Abdullah University of Science and Technology, 
<sup>2</sup>兰州大学, 
<sup>3</sup>悉尼大学, 
<sup>4</sup>IHPC, A*STAR

# 👀 WikiAutoGen 已被 ICCV2025 接收...

## WIKISEEK 基准测试

WIKISEEK 基准测试公开可用，下载地址：
[Google Drive](https://drive.google.com/drive/folders/1f_KDv9xh-fVXeCz_iUQv-beopfmAJyoO?usp=sharing)

## 安装

```bash
git clone https://github.com/wikiautogen/wikiautogen_code.git
cd wikiautogen
conda create -n wikiautogen python=3.11
conda activate wikiautogen
pip install -r requirements.txt
```

## 核心特性
- 🖼️ **多模态内容生成**：具备图像感知的主题提案能力
- 🤖 **自动化研究**：利用搜索引擎（Serper/You.com）进行信息采集
- 📝 **结构化写作**：包含大纲生成与文章润色
- 🔍 **事实验证**：通过多视角对话模拟确保内容准确性

## 快速开始

### 1. 环境配置
```bash
export OPENAI_API_KEY="your_openai_key"
export SERPER_API_KEY="your_serper_key"
```

### 2. 处理带图像的主题
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

- `do_research`: 若为 True，则模拟不同视角的对话以收集关于主题的信息；否则，加载已有结果。
- `do_generate_outline`: 若为 True，则为主题生成大纲；否则，加载已有结果。
- `do_generate_article`: 若为 True，则根据大纲和收集的信息为主题生成文章；否则，加载已有结果。
- `do_polish_article`: 若为 True，通过添加摘要部分（以及可选地删除重复内容）来润色文章；否则，加载已有结果。
- `do_positing`: 若为 True，为文章生成定位方案；否则，加载已有结果。
- `do_Retrieve_images`: 若为 True，根据定位方案和收集的信息为主题生成多模态文章；否则，加载已有结果。
- `do_mmpolish`: 若为 True，通过增强模态间的连贯性和一致性来润色文章，重点关注文本内容与视觉图表之间潜在的差异。 



## 许可证
本项目基于 MIT 许可证开源。基于维基百科数据的内容生成遵循 CC BY-SA 指南。

## 引用
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

## 👍 致谢
WikiAutoGen 参考了以下优秀工作：[Storm](https://github.com/stanford-oval/storm), [Co-storm](https://github.com/stanford-oval/storm), [Dspy](https://github.com/stanfordnlp/dspy)。
感谢！
