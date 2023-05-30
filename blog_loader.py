from typing import Any, List
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHINESE = "Chinese"
ENGLISH = "English"
CHINESE_SEPARATORS = ["。", "？", "！", "\n", "\n\n"]
ENGLISH_SEPARATORS = [".", "?", "!", "\n", "\n\n"]


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


def get_blog_documents(blog_url, lang):
    separators = CHINESE_SEPARATORS if lang == CHINESE else ENGLISH_SEPARATORS
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                   separators=separators)
    return HugoPostLoader(blog_url).load_and_split(text_splitter)
