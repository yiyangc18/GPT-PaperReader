from scholarly import scholarly

class PaperInfoFetch:
    def __init__(self, doi, journal_name):
        self.doi = doi
        self.journal_name = journal_name
        self.IF = -1
        self.partition = -1
        self.cites = -1

    def fetch_cites_by_doi(self):
        search_query = scholarly.search_pbs(self.doi)
        paper = next(search_query)
        cites = paper['num_citations']
        print(paper)
        self.cites = cites
        return cites


    def fetch_if_by_journal_name(self):
        # 根据期刊名称从指定数据库获取影响因子
        # 这里需要实现API调用逻辑
        # 示例返回值
        self.IF = 5.2  # 假设的影响因子
        return self.IF

    def fetch_partition_by_journal_name(self):
        # 根据期刊名称从指定数据库获取分区信息
        # 这里需要实现API调用逻辑
        # 示例返回值
        self.partition = 'Q1'  # 假设的分区信息
        return self.partition

    def update_paper_info(self):
        # 更新文献信息
        self.fetch_cites_by_doi()
        self.fetch_if_by_journal_name()
        self.fetch_partition_by_journal_name()

        return {
            'doi': self.doi,
            'journal_name': self.journal_name,
            'IF': self.IF,
            'partition': self.partition,
            'cites': self.cites
        }

if __name__ == '__main__':
    import requests
    import urllib.parse
    def fetch_cites_by_doi(doi):
        url = f"https://api.semanticscholar.org/v1/paper/{doi}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            cites = data.get("citationCount", 0)
            return cites
        else:
            print("Error fetching data")
            return 0

    def fetch_cites_by_title(title):
        # 编码论文标题以用于URL
        encoded_title = urllib.parse.quote_plus(title)
        search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_title}&limit=1"
        search_response = requests.get(search_url)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            papers = search_data.get('data', [])
            
            if papers:
                # 假设我们取第一个搜索结果
                paper_id = papers[0].get('paperId')
                paper_url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
                paper_response = requests.get(paper_url)
                
                if paper_response.status_code == 200:
                    paper_data = paper_response.json()
                    cites = paper_data.get("citationCount", 0)
                    return cites
                else:
                    print("Error fetching paper data")
                    return 0
            else:
                print("No papers found with the given title")
                return 0
        else:
            print("Error searching for paper")
            return 0

    # 示例使用
    cites = fetch_cites_by_doi('10.1126/scirobotics.adg1462')
    print(f"Cites: {cites}")




