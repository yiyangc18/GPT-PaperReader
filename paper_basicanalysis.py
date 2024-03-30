import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# 确保已下载NLTK的停用词集
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词和单字符
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

# 读取文件内容，只保留标题和摘要
df = pd.read_csv('papers_no_duplicates.txt', sep='\t', usecols=['TI', 'AB'])

# 将标题和摘要合并成一个长字符串
text = ' '.join(df['TI'].tolist() + df['AB'].tolist())

tokens = preprocess_text(text)

# 计算词频
word_freq = Counter(tokens)

# 选择最频繁的20个词
most_common_words = word_freq.most_common(50)
print(most_common_words)


