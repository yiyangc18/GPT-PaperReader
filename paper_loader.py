import pandas as pd

class PaperInfo:
    def __init__(self, publication_type, authors, title, publication_year, publication_name, conference_name, start_page, end_page, abstract, document_type, publication_date, dio):
        self.publication_type = publication_type
        self.authors = authors
        self.title = title
        self.publication_year = publication_year
        self.publication_name = publication_name
        self.conference_name = conference_name
        self.start_page = start_page
        self.end_page = end_page
        self.abstract = abstract
        self.document_type = document_type
        self.publication_date = publication_date
        self.dio = dio
        self.methods = []
        self.domains = []
        self.IF = -1
        self.Cite = -1
        self.partition = -1

    def __repr__(self):
        return f"Paper(title='{self.title}', authors='{self.authors}', year={self.publication_year}, doi='{self.dio}')"

class PaperLoader:
    def __init__(self, file_paths = ''):
        self.file_paths = file_paths
        self.papers = []

    def load_paper(self, file_path = ''):
        if file_path == '':
            file_path = self.file_paths[0]
        # 读取Tab Delimited File
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', dtype=str)
        df.fillna('', inplace=True)  # 用空字符串替换NaN值

        # 创建Paper对象列表
        for index, row in df.iterrows():
            paper = PaperInfo(
                publication_type=row.get('PT', ''),
                authors=row.get('AU', ''),
                title=row.get('TI', ''),
                publication_year=row.get('PY', ''),
                publication_name=row.get('SO', ''),
                conference_name=row.get('SE', ''),
                start_page=row.get('BP', ''),
                end_page=row.get('EP', ''),
                abstract=row.get('AB', ''),
                document_type=row.get('DT', ''),
                publication_date=row.get('PD', ''),
                dio=row.get('DI', '')
            )
            # 添加其他属性
            for attr in ['methods', 'domains', 'IF', 'Cite', 'partition']:
                if attr in row:
                    setattr(paper, attr, row[attr])
            self.papers.append(paper)
        return self.papers
            


    def load_papers_from_multiple_files(self,file_paths = ''):
        """
        从多个文件中加载文献。
        """
        if file_paths == '':
            file_paths = self.file_paths
        all_papers = []
        for file_path in file_paths:
            papers = self.load_paper(file_path)
            all_papers.extend(papers)
        return all_papers
    
    @staticmethod
    def save_papers_to_file(papers, file_path):
        # 创建一个空的 DataFrame
        df = pd.DataFrame()
        # 将每篇论文的信息添加到 DataFrame 中
        for paper in papers:
            paper_dict = {
                'PT': paper.publication_type,
                'AU': paper.authors,
                'TI': paper.title,
                'PY': paper.publication_year,
                'SO': paper.publication_name,
                'SE': paper.conference_name,
                'BP': paper.start_page,
                'EP': paper.end_page,
                'AB': paper.abstract,
                'DT': paper.document_type,
                'PD': paper.publication_date,
                'DI': paper.dio
            }
            # 添加其他属性
            for attr in ['methods', 'domains', 'IF', 'Cite', 'partition']:
                if hasattr(paper, attr) and getattr(paper, attr) is not None:
                    paper_dict[attr] = getattr(paper, attr)
            df = pd.concat([df, pd.DataFrame([paper_dict])], ignore_index=True)

        # 将 DataFrame 保存到文件
        df.to_csv(file_path, sep='\t', encoding='utf-8', index=False)

def papers_load_test():
    # 示例：使用类加载papers
    file_path = 'output\\savedrecs_UD_RL_tagged.txt'
    paper_loader = PaperLoader()
    papers = paper_loader.load_paper(file_path)

    # 显示加载的papers的前几项
    # for paper in papers[:2]:
    #     print(paper)

    # 保存papers到文件
    PaperLoader.save_papers_to_file(papers, 'test_files\\papers_test.txt')
    

if __name__ == '__main__':
    papers_load_test()
