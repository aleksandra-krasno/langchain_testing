import requests
import wikipedia
from langchain.agents import tool
from langchain_core.pydantic_v1 import BaseModel, Field

wikipedia.set_lang("pl")


class CurrencyInput(BaseModel):
    currency_code: str = Field(
        description="Three letters currency code in ISO 4217 format"
    )


@tool(args_schema=CurrencyInput)
def get_exchange_rate(currency_code: str) -> dict:
    """Get current exchange rate for given currency in NBP"""

    url = f"https://api.nbp.pl/api/exchangerates/rates/A/{currency_code}/"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        rate = data["rates"][0]["mid"]
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    return f"The current exchange rate is {rate}."


@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)
