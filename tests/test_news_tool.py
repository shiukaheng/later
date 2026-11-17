import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from lars.bandit import ThompsonBandit
from lars.evolve import Level, propose_level
from lars.llm import NewsTool, ask


class Answer(BaseModel):
    value: str


def response(parsed=None, tool_calls=None, content=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        parsed=parsed, tool_calls=tool_calls, content=content,
    ))])


class NewsToolTests(unittest.TestCase):
    def setUp(self):
        self.stories = [{"title": "Unique headline", "description": "Details", "source": "source", "published_at": "today"}]

    def test_no_tool_call_does_not_send_stories(self):
        news = NewsTool(self.stories)
        with patch("lars.llm.client.chat.completions.create", return_value=response(content='{"value":"ok"}')) as create, \
             patch("lars.llm.client.chat.completions.parse") as parse:
            result = ask("Write an evergreen post", Answer, news=news)
        self.assertEqual(result.value, "ok")
        self.assertFalse(news.used)
        self.assertNotIn("Unique headline", str(create.call_args.kwargs))
        parse.assert_not_called()

    def test_tool_call_delivers_news_only_in_tool_response(self):
        call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="read_news_cache", arguments="{}"))
        news = NewsTool(self.stories)
        with patch("lars.llm.client.chat.completions.create", return_value=response(tool_calls=[call])) as create, \
             patch("lars.llm.client.chat.completions.parse", return_value=response(Answer(value="ok"))) as parse:
            result = ask("Write a post", Answer, news=news)
        self.assertEqual(result.value, "ok")
        self.assertTrue(news.used)
        self.assertNotIn("Unique headline", str(create.call_args.kwargs["messages"][0]))
        continuation = parse.call_args.kwargs
        self.assertNotIn("tools", continuation)
        self.assertEqual(json.loads(continuation["messages"][-1]["content"])[0]["title"], "Unique headline")

    def test_policy_proposal_has_optional_tool_without_headlines_in_prompt(self):
        bandit = ThompsonBandit(["sports"])
        def propose(prompt, schema, *, news):
            self.assertNotIn("Unique headline", prompt)
            self.assertEqual(news.stories, self.stories)
            return Level(level="science")
        with patch("lars.evolve.ask", side_effect=propose):
            self.assertEqual(propose_level(bandit, "topic", [], [], "feed", stories=self.stories), "science")


if __name__ == "__main__":
    unittest.main()
