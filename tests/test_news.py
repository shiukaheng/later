import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from lars.factored import FactoredBandit
from lars.generator import Tweet, generate_tweet
from lars.news import News
from lars.prefetch import Prefetcher
from lars.session import SEEDED, Session


class NewsTests(unittest.TestCase):
    def test_first_refresh_prefills_then_subsequent_refresh_uses_one_page(self):
        with tempfile.TemporaryDirectory() as tmp, patch("lars.news.threading.Thread.start"):
            path = Path(tmp) / "news.json"
            service = News("secret", path)

            def response(request, timeout):
                page = int(parse_qs(urlsplit(request.full_url).query)["page"][0])
                stories = [{"title": f"story {page}-{i}", "url": f"https://example.com/{page}/{i}",
                            "published_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()} for i in range(3)]
                return io.BytesIO(json.dumps({"data": stories}).encode())

            with patch("lars.news.urlopen", side_effect=response) as fetch:
                service.refresh()
                self.assertEqual(fetch.call_count, 10)
                self.assertEqual(len(service.stories()), 30)
                state = json.loads(path.read_text())
                state["last_attempt"] = 0
                path.write_text(json.dumps(state))
                service.refresh()
                self.assertEqual(fetch.call_count, 11)
            service.close()

    def test_refresh_cache_rate_limit_and_expiry(self):
        with tempfile.TemporaryDirectory() as tmp, patch("lars.news.threading.Thread.start"):
            path = Path(tmp) / "news.json"
            service = News("secret", path)
            story = {"title": "A new discovery", "description": "Researchers report a result",
                     "url": "https://example.com/story", "source": "example.com",
                     "published_at": datetime.now(timezone.utc).isoformat()}
            with patch("lars.news.urlopen", return_value=io.BytesIO(json.dumps({"data": [story]}).encode())) as fetch:
                service.refresh()
                service.refresh()
                self.assertEqual(fetch.call_count, 1)
                self.assertIn("Mozilla/5.0", fetch.call_args.args[0].get_header("User-agent"))
            self.assertEqual(service.stories()[0]["title"], story["title"])
            state = json.loads(path.read_text())
            state["stories"][0]["published_at"] = "2000-01-01T00:00:00Z"
            path.write_text(json.dumps(state))
            self.assertEqual(service.stories(), [])
            service.close()

    def test_failed_refresh_is_rate_limited(self):
        with tempfile.TemporaryDirectory() as tmp, patch("lars.news.threading.Thread.start"):
            service = News("secret", Path(tmp) / "news.json")
            with patch("lars.news.urlopen", side_effect=OSError("secret")) as fetch:
                service.refresh()
                service.refresh()
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(service.stories(), [])
            service.close()

    def test_generator_maps_selected_story_and_includes_context(self):
        story = {"title": "New result", "description": "Details", "source": "example.com",
                 "published_at": "2026-01-01T00:00:00Z", "url": "https://example.com"}
        def use_news(prompt, schema, *, news):
            self.assertNotIn("New result", prompt)
            self.assertIn("New result", news.read())
            return Tweet(text="A post", source_index=0)

        with patch("lars.generator.ask", side_effect=use_news):
            text, source = generate_tweet({"current_events": "use news"}, "feed", [story])
        self.assertEqual((text, source), ("A post", story))
        with patch("lars.generator.ask", return_value=Tweet(text="No news", source_index=0)):
            self.assertIsNone(generate_tweet({}, "feed", [story])[1])

    def test_no_context_means_no_current_events_credit(self):
        bandit = FactoredBandit({"topic": ["tech"], "current_events": ["use news", "evergreen"]})
        with patch("lars.prefetch.generate_tweet", return_value=("post", None)):
            queue = Prefetcher(bandit, "feed", 1)
            levels, future = queue.take()
            self.assertNotIn("current_events", levels)
            self.assertEqual(future.result(), ("post", None))
            queue.close()

    def test_session_records_selected_news_and_rewards_policy(self):
        story = {"title": "New result", "description": "Details", "source": "example.com",
                 "published_at": "2026-01-01T00:00:00Z", "url": "https://example.com"}
        bandit = FactoredBandit({"style": ["plain"], **{name: [levels[0]] for name, levels in SEEDED.items()}})
        retired = {name: [] for name in bandit.dims}
        with patch("lars.session.store.load", return_value=(bandit, [], retired)), \
             patch("lars.session.store.save") as save, \
             patch("lars.session.News") as news, \
             patch("lars.session.AuthorPool") as authors, \
             patch("lars.generator.ask", side_effect=lambda prompt, schema, *, news: (news.read(), Tweet(text="A post", source_index=0))[1]):
            news.return_value.stories.return_value = [story]
            authors.return_value.take.return_value = {"id": "author"}
            session = Session()
            try:
                post = session.next()
                session.rate(post["id"], 1)
                self.assertEqual(post["source"], story)
                self.assertEqual(session.log[0]["source"], story)
                self.assertEqual(session.bandit.dims["current_events"].alpha[0], 2)
                save.assert_called_once()
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
