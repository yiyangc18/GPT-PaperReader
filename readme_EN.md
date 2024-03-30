

# GPT-Based Voracious Literature Reading Tool

## Introduction

Annually, thousands of new research papers are published, challenging scholars to stay current within their fields given their limited time and resources. Traditional literature review methods—identifying a few keywords, selecting seemingly relevant papers, and skimming through abstracts—can be time-consuming and might overlook valuable information. Moreover, the details of a paper can easily be forgotten over time, not to mention the potential for overlooking errors or varying quality across publications.

## Features

1. **Automated Tagging with LLM**: Utilize LLMs to automatically tag literature, swiftly categorizing articles by method and subfield.
2. **Literature Knowledge Base Chat**: Leveraging RAG (Retrievable Augmented Generation) technology, we have built a high-quality literature knowledge base. This enables users to find the information they need through GPT's QA (Question Answering) functionality.
3. **Natural Language SQL Queries**: By connecting to an SQL database, users can query specific literature data using natural language. This includes searching by methodology used, publication date, and journal ranking, among other criteria.

## Getting Started

### Installation

First, clone the project to your local machine.

Ensure Python and Jupyter Notebook are installed on your system before proceeding. Then, follow these steps to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage Guide

Note: This project is not fully equipped with a user interface like gradio, and thus is not recommended for individuals without a programming background.

![this is not a bug](document/not_bug.png)

1. copy the `config.py` file and rename it to `config_private.py`. Fill in the configuration information, such as the API key and proxy.

2. Open Jupyter Notebook:

```bash
jupyter notebook
```

Or, open the `paper_reader.ipynb` file in VSCode.

3. Follow the instructions provided within the notebook.

## Advanced Usage

Our tool offers various customization options and advanced features, detailed within the `paper_reader.ipynb` file.

## Planned Features

- [ ] Support for multiple LLMs - Implementing with langchain should be straightforward, but I have not had the need to do so yet.
- [ ] Tagging literature through embeddings rather than LLMs - This would save tokens, although previous attempts have not been very successful.
- [ ] Multithreading for literature data processing - To speed up data handling.

## How to Contribute

We welcome contributions of all kinds, including feature suggestions, code improvements, and documentation updates. Please contact us through GitHub by submitting Pull Requests or Issues.

## Acknowledgments

Thanks to Professor Zhong's "Finding Ideas" method, which inspired the development of this project. For more details, please refer to [Bilibili Tutorial](https://www.bilibili.com/read/cv26369099/). The configuration, toolbox, and utilities of this project were derived from [gpt_academic](https://github.com/binary-husky/gpt_academic).
