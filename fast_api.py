from typing import Union
from fastapi import FastAPI
from blog_summarize import summarize_blog
from blog_loader import CHINESE

app = FastAPI()


@app.get("/summary")
def read_item(url: str,
              lang: Union[str, None] = CHINESE,
              concise: Union[bool, None] = False):
    return summarize_blog(url, lang, concise)
