from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os
from blog_loader import get_blog_documents  # noqa: E402

prompt_template = """Write a summary of the following:
{text}
{concise} SUMMARY IN {lang}: """


def _get_prompt(prompt_template, lang, concise):
    concise = " " if concise else "CONCISE "
    return PromptTemplate(template=prompt_template,
                          partial_variables={"lang": lang,
                                             "concise": concise},
                          input_variables=["text"])


def set_openai_api_key(key):
    os.environ["OPENAI_API_KEY"] = key


def summarize_blog(url, lang, concise):
    """Summarize a blog post."""
    map_prompt = _get_prompt(prompt_template, lang, concise)
    combine_prompt = _get_prompt(prompt_template, lang, concise)
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True)
    data = get_blog_documents(url, lang)
    return chain.run(data)
