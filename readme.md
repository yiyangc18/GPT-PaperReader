# 基于GPT的“鲸吞式”文献阅读工具

## 项目介绍

[English Version of README](readme_EN.md)

每年都有成千上万篇新论文发表，而时间和精力是有限的。传统的方式，也就是挑几个关键词，找几篇看起来可能相关的论文，然后快速浏览摘要，关键文献边读边记笔记。这样的方法既费时又可能漏掉宝贵的信息。更别提，很多时候读完了一个月后可能就忘了内容，更不用说那些隐藏的错误或是文章质量参差不齐的问题了。

## 功能特点

1. **利用llm给文献打标签**：使用LLM自动给文献打标签，快速了解文章的方法和细分领域。
2. **基于文献知识库的chat**：基于RAG（Retrievable Augmented Generation）技术，构建高质量文献的知识库，使用户能够通过GPT的问答（QA）功能直接找到需要的信息。
3. **自然语言sql查询**：通过连接SQL数据库，允许用户用自然语言查询特定的文献数据，例如方法使用、发表时间和期刊等级等。

## 快速开始

### 安装

首先把项目git clone到本地。


在开始之前，请确保已经安装了Python和Jupyter Notebook。然后，你可以通过以下步骤安装所需的依赖。

```bash
pip install -r requirements.txt
```

### 使用指南

叠加: 本项目说明还不够完善，没有配备gradio之类的UI，不推荐没有编程基础的同学使用。

![this is not a bug](\document\not_bug.png)


1. 打开Jupyter Notebook:

```bash
jupyter notebook
```

或者在vscode里面打开`paper_reader.ipynb`文件。

2. 按照notebook中的指示进行操作。


## 进阶使用

我们的工具提供了多种自定义选项和高级功能，详见 `paper_reader.ipynb` 中的进阶部分。

## 计划中的功能

- [ ] 支持多种LLM - 通过langchain实现应该不难，但我没有需求就没做
- [ ] 通过embedding而不是LLM实现文献标签。 - 节省tokens，之前尝试的方法效果不太行
- [ ] 实现文献数据的多线程处理。 - 加快数据处理速度

## 如何贡献

我们欢迎任何形式的贡献，包括功能建议、代码改进、文档更新等。请通过GitHub提交Pull Request或Issue与我们联系。

## 鸣谢

感谢钟老师的“寻找idea”方法，启发了本项目的开发。更多详情请参考 [哔哩哔哩教程](https://www.bilibili.com/read/cv26369099/)。
本项目的config、toolbox、utils等工具源自项目[gpt_academic](https://github.com/binary-husky/gpt_academic)
