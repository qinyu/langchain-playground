from typing import List, Any
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.prompts import PromptTemplate

prompt_template = """
Write a summary of the following:
{text}
SUMMARY IN"""


def _build_metadata(soup: Any, url: str) -> dict:
    """Build metadata from BeautifulSoup output."""
    metadata = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", None)
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", None)
    return metadata


class HugoPostLoader(WebBaseLoader):
    """Loader that loads Hugo blog posts."""
    def __init__(self, web_path: str):
        # Only support one page for now
        super().__init__(web_path=web_path)

    def load(self) -> List[Document]:
        soup = self.scrape()
        hugo_post = soup.find_all("div", {"class": "post__content"})
        if len(hugo_post) > 0:
            post_content = hugo_post[0].text
            metadata = _build_metadata(soup, self.web_path)
            return [Document(page_content=post_content, metadata=metadata)]
        else:
            return super().load()


def _get_map_prompt(prompt_template, lang="Chinese"):
    return PromptTemplate(template=prompt_template+f" {lang}:",
                          input_variables=["text"])


def _get_blog_documents(blog_url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
    return HugoPostLoader(blog_url).load_and_split(text_splitter)


def summarize_blog(url, lang):
    """Summarize a blog post."""
    map_prompt = _get_map_prompt(prompt_template, lang)
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=map_prompt, verbose=True)
    data = _get_blog_documents(url)
    return chain.run(data)
