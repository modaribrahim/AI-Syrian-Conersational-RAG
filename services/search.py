import wikipedia  
from langchain_core.messages import SystemMessage  
import os
from dotenv import load_dotenv  
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchRun
from ..utils.utils import bcolors
from ..models.schemas import Search
from ..config.settings import settings
from ..config.prompts import prompts

###################### Wikipedia Search Class ######################

class WikiSearch(Search):
    def __init__(self, model):
        super().__init__()
        self.model = model
        

    def search_for_page(self, query: str) -> list:
        wikipedia.set_lang("en")
        print(bcolors.OKCYAN + f"Searching for pages..." + bcolors.ENDC)
        results = wikipedia.search(query)[:2]
        print(bcolors.OKCYAN + f"Found pages: {results}" + bcolors.ENDC)
        return results

    def get_wiki_content(self, queries: list) -> list:
        res = []
        for query in queries:
            try:
                res.append(wikipedia.page(query))
            except wikipedia.exceptions.PageError:
                print(bcolors.FAIL + f"Page not found for query: {query}" + bcolors.ENDC)
            except wikipedia.exceptions.DisambiguationError as e:
                print(bcolors.FAIL + f"Disambiguation error for query: {query}. Options: {e.options}" + bcolors.ENDC)
        return res

    def search(self, state):
        query = state["question"]
        print(bcolors.WARNING + f"Using the Wikipedia tool..." + bcolors.ENDC)

        search_query = self.model.invoke([
            SystemMessage(content=prompts.search_keywords_generation_prompt.format(query=query))
        ]).content

        pages = self.search_for_page(search_query)
        if not pages:
            return "No Wikipedia pages found."

        content = self.get_wiki_content(pages)
        if not content:
            return "No results found."

        results = "\n\n".join(
            f"Title: {page.title}\nSummary: {page.summary[:800]}...\nURL: {page.url}\n"
            for page in content
        )
        print(bcolors.WARNING + "Returned results from Wikipedia." + bcolors.ENDC)
        return results

###################### Tavily Search Class ######################

class TavilySearch(Search):
    def __init__(self):
        super().__init__()

    def get_tavily_search(self, query):
        print(bcolors.WARNING + "Using the Tavily search tool..." + bcolors.ENDC)
        try:
            load_dotenv()
            api_key = settings.tavily_api_key

            if not api_key:
                print(bcolors.WARNING + "Warning: TAVILY_API_KEY is not set. Cannot perform Tavily search.\n" + bcolors.ENDC)
                return "No API key found for Tavily search."

            os.environ["TAVILY_API_KEY"] = api_key

            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=False,
            )

            res = tool.invoke({"query": query})
            content = "\n".join(f"Source: {r.get('url', 'N/A')} - {r.get('content', 'No content')}" for r in res)

            print(bcolors.WARNING + "Returned results from Tavily" + bcolors.ENDC)
            return content or "No relevant results found."

        except Exception as e:
            print(bcolors.WARNING + f"Error in Tavily search: {e}. Returning empty response.\n" + bcolors.ENDC)
            return "Error in Tavily search."

    def search(self, state):
        query = state["question"]
        print(bcolors.OKBLUE + "Searching using Tavily..." + bcolors.ENDC)
        web_results = self.get_tavily_search(query)
        return {"tavily_docs": web_results, "question": query}

###################### DuckDuckGo Search Class ######################

class DuckDuckGoSearch(Search):
    def __init__(self):
        super().__init__()

    def get_duckduckgo_search(self, query):
        try:
            search = DuckDuckGoSearchRun()
            return search.invoke(query.strip())  
        except Exception as e:
            print(bcolors.WARNING + f"Error in DuckDuckGo search: {e}. Returning empty response.\n" + bcolors.ENDC)
            return "Error in DuckDuckGo search."

    def search(self, state):
        print(bcolors.OKBLUE + "Searching using DuckDuckGo..." + bcolors.ENDC)
        question = state["question"]
        web_results = self.get_duckduckgo_search(question)
        return {"duckduck_docs": web_results}
