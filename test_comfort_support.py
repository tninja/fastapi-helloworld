from comfort_search import DuckDuckGoSearchProvider


class FakeSearchProvider:
    def __init__(self, context: str):
        self.context = context
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.context


class FakeDuckDuckGoSearchProvider(DuckDuckGoSearchProvider):
    def __init__(self, context: str):
        super().__init__(
            server_cmd="uvx",
            server_args=("duckduckgo-mcp-server",),
            server_dir=None,
            max_results=4,
        )
        self.context = context
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.context


class FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("Response", (), {"output_text": self.output_text})()


class FakeCompletions:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": self.response_content}
                            )()
                        },
                    )()
                ]
            },
        )()


class FakeOpenAIClient:
    def __init__(self, response_content: str, research_output_text: str = ""):
        self.completions = FakeCompletions(response_content)
        self.chat = type("Chat", (), {"completions": self.completions})()
        self.responses = FakeResponses(research_output_text)
