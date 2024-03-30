from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma, FAISS


class llmConfig:
    def __init__(self, temperature=0.2, max_tokens=2000, model_name='gpt-3.5-turbo', openai_api_key=' ', openai_proxy=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        from toolbox import is_any_api_key
        if not is_any_api_key(openai_api_key):
            raise ValueError("OpenAI API key is not set. Please set it in the configuration file or as an environment variable.")
        self.openai_api_key = openai_api_key
        self.openai_proxy = openai_proxy

def get_rag_chain(llmconfig):

    llm = ChatOpenAI(
            temperature=llmconfig.temperature,
            max_tokens=llmconfig.max_tokens,
            model_name=llmconfig.model_name,
            openai_api_key=llmconfig.openai_api_key,
            openai_proxy=llmconfig.openai_proxy
    )

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain

def get_chain_prompt_classifier(llmconfig, methods, domains):
    # Convert methods and domains lists to strings for inclusion in the prompt, including an option for "Not Specified"
    methods_str = ', '.join([f"'{method}'" for method in methods]) + ", 'Not Specified'"
    domains_str = ', '.join([f"'{d}'" for d in domains])

    # Update system message for better clarity and instruction for the model
    system_message = f"Based on the title and abstract provided, please analyze and identify the specific methods used. indicate 'Not Specified' if the methods are not explicitly stated. " \
                    f"Additionally, determine the primary domain of study the paper contributes to from the provided list or specify the most relevant area if not listed. " \
                    f"Available methods: {methods_str}. " \
                    f"Possible domains of study: {domains_str}."\
                    "Format your response is Json format."
    # print(system_message)

    # Create a new classifier prompt with refined instructions
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}")
    ])

    # Create the llm using llmconfig
    llm = ChatOpenAI(
        temperature=llmconfig.temperature,
        max_tokens=llmconfig.max_tokens,
        model_name=llmconfig.model_name,
        openai_api_key=llmconfig.openai_api_key,
        openai_proxy=llmconfig.openai_proxy
    )

    # Switch to JsonOutputParser for JSON-formatted output
    from langchain_core.output_parsers import JsonOutputParser
    output_parser = StrOutputParser()

    # Assemble the new chain with the refined prompt and JSON output parser
    new_chain = classifier_prompt | llm | output_parser

    return new_chain

def get_chain_paper_tagging(llmconfig, schema):
    from langchain.chains import create_tagging_chain
    llm = ChatOpenAI(
        temperature=llmconfig.temperature,
        max_tokens=llmconfig.max_tokens,
        model_name=llmconfig.model_name,
        openai_api_key=llmconfig.openai_api_key,
        openai_proxy=llmconfig.openai_proxy
    )
    chain = create_tagging_chain(schema, llm)
    return chain

def save_papers_tiab(papers,output_file):
    import re
    import nltk
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            # 使用 nltk.sent_tokenize 将摘要切分成句子
            abstract_sentences = nltk.sent_tokenize(paper.abstract)
            # 检查每个句子的长度，如果超过200字符，就在句号后面添加换行符
            formatted_sentences = [re.sub(r'\.', '.\n', sentence) if len(sentence) > 200 else sentence for sentence in abstract_sentences]
            # 将处理后的句子连接成一个字符串
            formatted_abstract = '\n'.join(formatted_sentences)
            formatted_paper = f"title: {paper.title}\nabstract: {formatted_abstract}\n\n"
            f.write(formatted_paper)


def get_paper_RAG_chain_pdf(llmconfig, file_paths = None):
    # 这个方法返回的pdf信息总是特别差，很耗费token  所以另外构建了一个通过自己定义prompt的方法
    if file_paths is None:
        print("Please provide the file path of the PDF papers.")
        return

    embedding=OpenAIEmbeddings(openai_api_key=llmconfig.openai_api_key, openai_proxy=llmconfig.openai_proxy)
    pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load_and_split())

    vector = FAISS.from_documents(pages, embedding)
    llm = ChatOpenAI(
        temperature=llmconfig.temperature,
        max_tokens=llmconfig.max_tokens,
        model_name=llmconfig.model_name,
        openai_api_key=llmconfig.openai_api_key,
        openai_proxy=llmconfig.openai_proxy
    )
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.documents import Document

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    from langchain.chains import create_retrieval_chain
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def generate_vectors_pdf(llmconfig, file_paths=None):
    if file_paths is None:
        print("Please provide the file path of the PDF papers.")
        return
    embedding = OpenAIEmbeddings(openai_api_key=llmconfig.openai_api_key, openai_proxy=llmconfig.openai_proxy)
    pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load_and_split())
    vectors = FAISS.from_documents(pages, embedding)
    return vectors

def get_llm_rag_answer(llmchain, vectors, question, topk=3):
    # 自己做RAG检索内容的生成、prompt  不用langchain的方法
    docs = vectors.similarity_search(question, k=topk)
    context = "\n".join([x.page_content for x in docs])

    input_text = "Answer the following question based only on the provided context:\nContext:\n" + context + "\nUser question:\n" + question
    
    answer = llmchain.invoke({"context": context, "input": question})

    return {"answer": answer, "context": context}

def get_paper_RAG_chain_txt(llmconfig, file_paths = None):
    if file_paths is None:
        print("Please provide the file path of the text papers.")
        return
    all_documents = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
                documents = TextLoader(file_path, encoding='utf-8').load()
                all_documents.extend(documents)

    embedding=OpenAIEmbeddings(openai_api_key=llmconfig.openai_api_key, openai_proxy=llmconfig.openai_proxy)
    text_splitter = SemanticChunker(embedding)
    documents = text_splitter.split_documents(all_documents)
    db = Chroma.from_documents(documents, embedding)

    retriever = db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    
    llm = ChatOpenAI(
        temperature=llmconfig.temperature,
        max_tokens=llmconfig.max_tokens,
        model_name=llmconfig.model_name,
        openai_api_key=llmconfig.openai_api_key,
        openai_proxy=llmconfig.openai_proxy
    )
        
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_with_source

def get_chain_paper_sql(llmconfig, file_path):
    import pandas as pd
    from sqlalchemy import create_engine
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    # 读取文件
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8', dtype=str)
    # 用空字符串替换NaN值
    df.fillna('', inplace=True)
    papers_dict_list = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 创建一个Paper对象，并将其添加到列表中
        paper = {
            'publication_type': row.get('PT', ''),
            'authors': row.get('AU', ''),
            'title': row.get('TI', ''),
            'publication_year': row.get('PY', ''),
            'publication_name': row.get('SO', ''),
            'conference_name': row.get('SE', ''),
            'start_page': row.get('BP', ''),
            'end_page': row.get('EP', ''),
            'abstract': row.get('AB', ''),
            'document_type': row.get('DT', ''),
            'publication_date': row.get('PD', ''),
            'doi': row.get('DI', ''),
            'methods': row.get('methods', ''),
            'domains': row.get('domains', ''),
        }
        papers_dict_list.append(paper)
    
    df = pd.DataFrame(papers_dict_list)

    # 创建数据库引擎
    engine = create_engine("sqlite:///output/papers.db")

    # 将DataFrame保存到SQLite数据库中
    df.to_sql("papers", engine, index=False, if_exists='replace')

    # 创建SQL数据库实例
    db = SQLDatabase(engine=engine)

    # 初始化LLM
    llm = ChatOpenAI(
        temperature=llmconfig.temperature,
        max_tokens=llmconfig.max_tokens,
        model_name=llmconfig.model_name,
        openai_api_key=llmconfig.openai_api_key,
        openai_proxy=llmconfig.openai_proxy
    )

    # 创建SQL代理
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    
    return agent_executor



if __name__ == '__main__':
    from toolbox import get_conf
    proxies, API_KEY,LLM_MODEL = get_conf('proxies', 'API_KEY','LLM_MODEL')
    llmconfig = llmConfig(openai_api_key=API_KEY, openai_proxy=proxies['http'], model_name=LLM_MODEL)
    
    # chain = get_chain(llmconfig)
    # result = chain.invoke({"input": "hello, world!"})
    # print(result)

    llmconfig.temperature = 0
    methods = [
    "Deep Q-Learning",  # 深度Q学习
    "Dueling DQN",  # 决斗DQN
    "Double DQN",  # 双重DQN
    "Actor Critic",  # 演员批判家方法
    "Q-Learning",  # Q学习
    "Policy Gradient",  # 策略梯度
    "A3C (Asynchronous Advantage Actor-Critic)",  # 异步优势演员批判家
    "DDPG (Deep Deterministic Policy Gradient, depth deterministic strategy gradient)",  # 深度确定性策略梯度
    "TRPO (Trust Region Policy Optimization)",  # 信任区域策略优化
    "PPO (Proximal Policy Optimization)",  # 近端策略优化
    "SAC (Soft Actor-Critic)",  # 软演员批判家
    "Twin Delayed DDPG (TD3)",  # 双延迟深度确定性策略梯度
    "Rainbow DQN",  # 彩虹DQN
    "HER (Hindsight Experience Replay)"  # 追忆经验回放
    ]

    domains = [
    "Motion Planning",  # 运动规划
    "Behavior Decision",  # 行动选择（包括变道、转向等）
    "Local Path Planning",  # 局部路径规划
    "Velocity Control",  # 纵向控制（速度保持、加速等）
    "Lateral Control",  # 横向控制（轨迹跟踪、稳定性控制等）
    "Attitude Control",  # 姿态控制（主要涉及稳定性）
    "Trajectory Optimization",  # 轨迹优化
    "Dynamic Obstacle Avoidance",  # 动态障碍物避让
    "Risk Assessment",  # 风险评估
    "Map Integration",  # 地图集成（将车辆感知与地图数据相结合以支持规划和导航）
    "Vehicle-to-Everything (V2X) Communication",  # 车辆至一切(V2X)通信（用于提高预测准确性和决策的合理性）
    ]

    schema = {
        "properties": {
            "method": {
                "type": "string",
                "enum": methods,
                "description": "The specific method used in the paper from a predefined list. indicate 'Not Specified' if the methods are not explicitly stated.",
            },
            "domain": {
                "type": "string",
                "enum": domains,
                "description": "The primary domain or field of study the paper contributes to.",
            },
        },
        "required": ["method", "domain"],
    }

    # chain = get_chain_paper_tagging(llmconfig, schema)
    # chain = get_chain_prompt_classifier(llmconfig, methods, domains)

    # from paper_preprocess import preprocess_main
    # 使用PaperLoader函数加载papers
    # file_paths = [
    #     'test_files\\savedrecs_UD_RL_1-1000.txt',
    #     'test_files\\savedrecs_UD_RL_1001-1346.txt'
    # ]

    # papers = preprocess_main(file_paths)
    knowledge_files = ['literature_pdf\\Advanced planning for autonomous vehicles using reinforcement.pdf']
    vector = generate_vectors_pdf(llmconfig, knowledge_files)
    llmchain = get_rag_chain(llmconfig)

    question = "Why the autor use DRL for automotive motion planning?"
    result = get_llm_rag_answer(llmchain, vector, question)

    print(result)
    

