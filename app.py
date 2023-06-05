from prompt_engineering import api, task
from streamlit import (
    cache_data,
    form,
    form_submit_button,
    multiselect,
    secrets,
    set_page_config,
    text_area,
)

set_page_config(
    page_title="Classify Search Terms", page_icon="🤖", initial_sidebar_state="expanded"
)

API_KEY = secrets["OPENAI_KEY"]
API_URL = "https://api.openai.com/v1/engines/text-davinci-002/completions"


@api(
    endpoint=API_URL,
    key=API_KEY,
    hyperparameters=dict(temperature=0.6, n=5),
    cache=False,
)
def gpt3(data: dict) -> None:
    return list(map(lambda result: result["text"], data["choices"]))


@task(task_name="topic_classification")
def topic_classification(prompt: str) -> str:
    return gpt3(data=dict(prompt=prompt))


@cache_data
def query(search_term: str) -> str:
    results = topic_classification(prompt=search_term)
    return set(
        map(
            lambda result: result.split("The topic of this article is:")[-1].strip(),
            results,
        )
    )


with form(key="zeroshot_classifier"):
    search_term = text_area("Search Term:", "Portable blender")
    if form_submit_button("Suggest Topics"):
        topics = query(search_term=search_term)
        multiselect("Select:", topics)
