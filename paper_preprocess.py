from paper_loader import PaperLoader

def remove_duplicates(papers):
    """
    移除重复的文献。假设DOI可以唯一标识一篇文献。
    如果文献没有DOI，将使用标题和出版年份作为唯一标识。
    """
    unique_papers = []
    seen = set()
    for paper in papers:
        identifier = paper.dio if paper.dio else f"{paper.title}_{paper.publication_year}"
        if identifier not in seen:
            unique_papers.append(paper)
            seen.add(identifier)
    return unique_papers

def remove_papers_before_year(papers, year):
    """
    剔除发表年份在指定年份之前的文献。
    """
    filtered_papers = [paper for paper in papers if paper.publication_year.isdigit() and int(paper.publication_year) >= year]
    return filtered_papers

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

def generate_wordcloud(papers, domain_names):
    # 检查停用词列表是否已经下载
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')  # 如果没有下载，就下载停用词列表

    # 检查 'punkt' 是否已经下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')  # 如果没有下载，就下载 'punkt'

    # 获取英文停用词列表
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.update(['using', 'based', 'method', 'algorithm', 'function','system','model','data','approach','problem','analysis','design','performance','application','systems','evaluation','optimization','involves','strategy','development','implementation','solution','environment','simulation','architecture','arxiv'])

    # 将所有文献的标题合并成一个长字符串，并转换为小写
    text = ' '.join(paper.title.lower() for paper in papers)
    # text = ' '.join(paper.abstract.lower() for paper in papers)

    # 将领域名称添加到stopwords集合中
    for domain_name in domain_names:
        domain_name = domain_name.replace('*', '')  # 去除通配符
        # 使用正则表达式匹配以domain_name开头的所有词，并将它们添加到stopwords集合中
        stopwords_set.update(re.findall(r'\b' + domain_name + r'\w*\b', text))

    
    # 创建词云对象
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords_set, 
                min_font_size = 10).generate(text)

    # 显示生成的词云图像
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    # 保存图像到文件
    plt.savefig('output\\wordcloud.png')
    plt.close()

    # 获取词频
    filtered_words = [word for word in text.split() if word not in stopwords_set]
    word_frequencies =  nltk.FreqDist(filtered_words)    
    # 按词频排序
    sorted_words = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)
    # 打印出最高频的词
    for word, freq in sorted_words[:20]:  # 打印前20个
        print(f'Word: {word}, Frequency: {freq}')
    return sorted_words[:20]

from collections import Counter
def count_papers_per_year_and_plot(papers):
    # 获取每篇论文的发表年份
    publication_years = [paper.publication_year for paper in papers]

    # 计算每一年发表的数量
    count_per_year = Counter(publication_years)

    # 获取年份和对应的论文数量
    years = sorted(count_per_year.keys())
    counts = [count_per_year[year] for year in years]

    # 创建柱状图
    plt.bar(years, counts)

    # 设置图表标题和轴标签
    plt.title('Number of Papers Published Each Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')

    plt.savefig('output\\year-publishes.png')
    plt.close()

    return count_per_year


def preprocess_main(file_paths, latest_year=2010):
    # 从多个文件中加载papers
    paper_loader = PaperLoader(file_paths)
    papers = paper_loader.load_papers_from_multiple_files()
    print(f"加载的文献总数量: {len(papers)}")

    # 移除重复的文献
    papers_no_duplicates = remove_duplicates(papers)
    print(f"移除重复后的文献数量: {len(papers_no_duplicates)}")

    papers = remove_papers_before_year(papers, latest_year)
    print(f"删除{latest_year}年前文献后数量: {len(papers_no_duplicates)}")

    # 生成词云
    domain_names = ['auto','vehicle', 'reinforcement','learning','driving','deep','unmanned']
    generate_wordcloud(papers, domain_names)

    # 获取年份和对应的论文数量
    count_papers_per_year_and_plot(papers)
    
    PaperLoader.save_papers_to_file(papers_no_duplicates, 'output\\papers_no_duplicates.txt')

    return papers


if __name__ == '__main__':
    file_paths = [
    'test_files\\savedrecs_UD_RL_1-1000.txt',
    'test_files\\savedrecs_UD_RL_1001-1346.txt'
    ]   
    preprocess_main(file_paths)
