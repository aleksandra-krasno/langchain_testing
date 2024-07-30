# Function calling with Langchain

Chatbot with two supportive functions: `search_wikipedia` (to search texts over Wikipedia) and `get_exchange_rate` (to get the current exchange rate from the NBP API).

1. First, set up environment variable **OPENAI_API_KEY** to connect with OPEN AI API. 

2. Run application by calling:
```commandline
python app.py
```

Example of running:
```commandline
co potrafisz?
Jestem pomocnym asystentem AI, który może odpowiedzieć na pytania na podstawie wyników z Wikipedii oraz podać najnowsze kursy walut NBP. W czym mogę Ci pomóc?
You: kurs wymiany euro
Aktualny kurs wymiany euro wynosi 4.2847.
You: kiedy urodził się Piłsudski?
Józef Piłsudski urodził się 5 grudnia 1867 roku w majątku Zułów.
You: a kiedy zmarł?
Józef Piłsudski zmarł 12 maja 1935 roku w Warszawie.
```

