
from openai import OpenAI
import requests
from xml.etree import ElementTree
from datetime import datetime
import json
import asyncio
from pywebio.input import input, TEXT
from pywebio.output import put_text, put_table, put_buttons, popup, clear, put_html
from pywebio import start_server
from pywebio.output import put_markdown, put_link

client = OpenAI(
  api_key="sk-your api",
  base_url='your url',
)


def get_completion(prompt, model="gpt-4o-2024-05-13",conversation_history=[]):
    messages = [
                   {"role": "system", "content": "You are a helpful assistant."},
               ] + conversation_history + [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        if response.choices:
            content = response.choices[0].message.content.strip()  # 正确访问属性
            return content
    except Exception as e:
        print(f"API调用出错: {e}")
        return None


def search_pubmed(query, max_results=5, year=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    query += f" AND {year}[PDAT]" if year else ""
    query_url = f"{base_url}db=pubmed&term={query}&retmax={max_results}&retmode=json"
    try:
        response = requests.get(query_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求错误：{e}")
        return None



def fetch_pubmed_details(paper_ids, full_text=False):
    """ 获取文献详细信息 """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(paper_ids),
        'retmode': 'xml',
        'rettype': 'full' if full_text else 'abstract'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"获取文献详细信息错误：{e}")
        return None



def parse_pubmed_response(xml_response):
    """对文献解析，提取更多相关信息"""
    tree = ElementTree.fromstring(xml_response)
    papers = []
    for article in tree.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle")
        abstract = article.findtext(".//AbstractText")
        introduction = article.findtext(".//Introduction")
        conclusion = article.findtext(".//Conclusion")
        pub_date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate")
        doi = article.findtext(".//ELocationID[@EIdType='doi']")
        authors = [author.findtext(".//LastName") + " " + author.findtext(".//ForeName") for author in article.findall(".//Author")]
        journal = article.findtext(".//Journal/Title")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
        papers.append({
            'title': title, 'abstract': abstract, 'introduction': introduction,
            'conclusion': conclusion, 'published': pub_date, 'link': link,
            'authors': authors, 'journal': journal, 'doi': doi or "No DOI available"
        })
    return papers



def extract_paper_details(article):
    """ 从PubMed文章中提取详细信息 """
    title = article.findtext(".//ArticleTitle")
    abstract = article.findtext(".//AbstractText")
    # 获取日期信息
    pub_date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate")
    if not pub_date:
        pub_date = article.findtext(".//DateRevised/Year")
    doi = article.findtext(".//ELocationID[@EIdType='doi']")
    authors = [author.findtext(".//LastName") + " " + author.findtext(".//ForeName") for author in article.findall(".//Author")]
    if not authors:
        authors = ["Unknown Authors"]
    journal = article.findtext(".//Journal/Title")
    link = f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}"
    return {
        'title': title, 'abstract': abstract, 'published': pub_date,
        'link': link or "No link available", 'authors': authors, 'journal': journal, 'doi': doi or "No DOI available"
    }


def generate_apa_citation(paper):
    authors = paper.get('authors', ['No authors listed'])
    if not authors:
        authors_str = "No authors listed"
    else:
        authors_str = ', '.join([f"{name.split(' ')[-1]}, {' '.join(name.split(' ')[:-1])}" for name in authors[:-1]])
        if len(authors) > 1:
            authors_str += ", & " + f"{authors[-1].split(' ')[-1]}, {' '.join(authors[-1].split(' ')[:-1])}"
        else:
            authors_str = f"{authors[0].split(' ')[-1]}, {' '.join(authors[0].split(' ')[:-1])}"

    year = paper.get('published', "n.d.")
    title = paper['title']
    journal = paper.get('journal', 'No journal available')
    doi = paper.get('doi', 'No DOI available')
    citation = f"{authors_str} ({year}). {title}. *{journal}*. doi:{doi}"
    return citation.replace("doi:No DOI available", "")  # 如果没有DOI，从字符串中去掉DOI部分



from transformers import pipeline
summarizers = {
    "default": pipeline("summarization"),
    "pegasus": pipeline("summarization", model="google/pegasus-xsum"),
    "bart": pipeline("summarization", model="facebook/bart-large-cnn")
}

def summarize_text(text, model_name="default", max_length=200, min_length=30):
    try:
        summarizer = summarizers.get(model_name, summarizers["default"])
        summaries = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        if summaries:
            return summaries[0]['summary_text']
        else:
            return "无法生成摘要。"
    except Exception as e:
        return f"生成摘要时出错：{str(e)}"

def split_text(text, max_length=1024):
        """将长文本按最大长度切分为多个段落。"""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]
def summarize_long_text(text, model_name="default", max_length=200, min_length=30):
        try:
            segments = split_text(text)
            summaries = [summarize_text(segment, model_name=model_name, max_length=max_length, min_length=min_length)
                         for segment in segments]
            return "\n".join(summaries)
        except Exception as e:
            return f"生成长文本摘要时出错：{str(e)}"
def summarize_text_with_options(text, model_name="default", length_option="short"):
    if length_option == "short":
        max_length = 100
        min_length = 30
    elif length_option == "medium":
        max_length = 200
        min_length = 50
    elif length_option == "long":
        max_length = 300
        min_length = 100
    else:
        max_length = 200
        min_length = 30
    return summarize_long_text(text, model_name=model_name, max_length=max_length, min_length=min_length)




current_papers = []
conversation_history = []

def chat_with_user(conversation_history, user_input):
    global current_papers
    if user_input.startswith("第") and "篇文献" in user_input:
        try:
            parts = user_input.split('篇文献')[0]
            index = int(parts[1:]) - 1
            if index < 0 or index >= len(current_papers):
                raise IndexError("文献编号超出范围。")
            current_paper_details = current_papers[index]
            paper_content = f"Title: {current_paper_details['title']}\n" \
                            f"Abstract: {current_paper_details['abstract']}\n" \
                            f"Introduction: {current_paper_details['introduction']}\n" \
                            f"Conclusion: {current_paper_details['conclusion']}\n"
            summary_length = "medium"
            summary_model = "pegasus"
            abstract_summary = summarize_text_with_options(
                current_paper_details['abstract'] if current_paper_details['abstract'] else "No abstract available",
                model_name=summary_model, length_option=summary_length)
            introduction_summary = summarize_text_with_options(
                current_paper_details['introduction'] if current_paper_details[
                    'introduction'] else "No introduction available", model_name=summary_model,
                length_option=summary_length)
            conclusion_summary = summarize_text_with_options(
                current_paper_details['conclusion'] if current_paper_details[
                    'conclusion'] else "No conclusion available", model_name=summary_model,
                length_option=summary_length)

            conversation_history.append({"role": "user", "content": paper_content})
            prompt = (
                f"你是一个智能文献检索助手，专门帮助用户查找和总结科学文献。回答时请使用中文，并以友好、专业的语气进行交流。\n\n"
                f"请总结以下文献的主要内容：\n"
                f"标题：{current_paper_details['title']}\n"
                f"摘要：{abstract_summary}\n"
                f"介绍：{introduction_summary}\n"
                f"结论：{conclusion_summary}\n"
            )
            response = get_completion(prompt, model="gpt-4o-2024-05-13", conversation_history=conversation_history)
        except (IndexError, ValueError):
            response = "请输入正确的文献编号。"
        except Exception as e:
            response = f"生成摘要时出错：{str(e)}"
    elif user_input.lower() in ['结束', 'exit', 'quit']:
        response = "感谢你使用文献助手！再见！"
    else:
        conversation_history.append({"role": "user", "content": user_input})
        prompt = (
            f"你是一个智能文献检索助手，专门帮助用户查找和总结科学文献。回答时请使用中文，并以友好、专业的语气进行交流。\n\n"
            f"用户输入：{user_input}\n\n"
            f"请根据以上用户输入生成适当的回答。"
        )
        response = get_completion(prompt, model="gpt-4o-2024-05-13", conversation_history=conversation_history)

    conversation_history.append({"role": "assistant", "content": response})
    return response, conversation_history





def initialize_conversation(paper_summary):
    return [{"role": "system", "content": "你是一个文献助手。"},
            {"role": "assistant", "content": f"以下是文献的摘要：{paper_summary['abstract']}\n引言：{paper_summary['introduction']}\n结论：{paper_summary['conclusion']}"}]


def literature_assistant(query, pubmed_max_results=5, year=None):
    """ 文献助手主函数 """
    pubmed_results = search_pubmed(query, max_results=pubmed_max_results, year=year)
    if pubmed_results and 'esearchresult' in pubmed_results and 'idlist' in pubmed_results['esearchresult']:
        pubmed_ids = pubmed_results['esearchresult']['idlist']
        pubmed_xml = fetch_pubmed_details(pubmed_ids)
        pubmed_papers = parse_pubmed_response(pubmed_xml) if pubmed_xml else []
    else:
        pubmed_papers = []
    return pubmed_papers





#test
def start_literature_assistant():
    put_html("<h1 style='color: #4CAF50; text-align: center;'>欢迎使用文献助手</h1>")
    query = input("请输入你的查询主题:", type=TEXT, placeholder="例如：人工智能在医疗中的应用", required=True)
    year = input("请输入查询年份(可选):", type=TEXT, placeholder="例如：2020")

    papers = literature_assistant(query, pubmed_max_results=5, year=year)
    if papers:
        global current_papers
        current_papers = papers
        put_markdown("### 我找到了以下文献：")
        for i, paper in enumerate(papers):
            put_markdown(f"#### 文献 {i + 1}")
            put_table([
                ['标题', put_html(f"<b>{paper['title']}</b>")],
                ['摘要', put_html(f"{paper['abstract'][:200]}..." if paper['abstract'] else '无摘要提供。')],
                ['发布时间', paper['published']],
                ['链接', put_link("点击查看全文", paper.get('link', 'No link available'))],
                ['APA引用', paper['doi'] or "无 DOI 提供"]
            ])
            put_markdown(f"**APA 引用**: {generate_apa_citation(paper)}")
            put_html("<hr>")

        put_markdown("你可以问我关于这些文献的更多问题，或者输入 '结束' 来结束对话。")

        conversation_history = []
        while True:
            user_input = input("请输入你的问题或命令:", type=TEXT, placeholder="例如：总结第1篇文献的内容")
            if user_input.lower() in ['结束', 'exit', 'quit']:
                put_text("感谢你使用文献助手！再见！")
                break
            response, conversation_history = chat_with_user(conversation_history, user_input)
            put_markdown(f"**助手回答**: {response}")
    else:
        put_text("没有找到相关的文献，请尝试其他查询。")


if __name__ == '__main__':
    start_server(start_literature_assistant, port=8081, debug=True)




