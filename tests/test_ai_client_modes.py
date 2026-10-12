import os
import json
import unittest
from unittest.mock import Mock, patch

from moduli import ai_client


class OpenRouterJsonModeTests(unittest.TestCase):
    def test_single_attempt_option_never_retries_a_rate_limit(self):
        class RateLimited:
            status_code = 429
            text = "rate limited"

            @staticmethod
            def json():
                return {"error": {"message": "rate limited"}}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}), \
             patch.object(ai_client.requests, "post", return_value=RateLimited()) as request, \
             patch.object(ai_client.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "1 attempt"):
                ai_client.chat_openrouter("single request", max_attempts=1)
        request.assert_called_once()
        sleep.assert_not_called()

    def test_json_mode_is_opt_in(self):
        class Response:
            status_code = 200
            text = "{}"

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "ok"}}]}

        calls = []

        def post(*args, **kwargs):
            calls.append(kwargs["json"])
            return Response()

        messages = [{"role": "user", "content": "plain text"}]
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}), \
             patch.object(ai_client.requests, "post", side_effect=post):
            self.assertEqual(ai_client._openrouter(messages, json_mode=False), "ok")
            self.assertEqual(ai_client._openrouter(messages, json_mode=True), "ok")

        self.assertNotIn("response_format", calls[0])
        self.assertEqual(calls[1]["response_format"], {"type": "json_object"})

    def test_chat_wrapper_forwards_json_mode_to_openrouter(self):
        openrouter = Mock(return_value="ok")
        with patch.dict(os.environ, {"AI_SERVICE": "openrouter", "OPENROUTER_API_KEY": "offline-test-key"}), \
             patch.dict(ai_client._PROVIDERS, {"openrouter": openrouter}):
            self.assertEqual(ai_client.chat_ollama("structured", json_mode=True), "ok")
        openrouter.assert_called_once()
        self.assertEqual(openrouter.call_args.kwargs["json_mode"], True)

    def test_content_generation_uses_one_json_call_and_deterministic_plan_fallback(self):
        from moduli import cervello

        payload = json.dumps({
            "title": "A topic explained",
            "description": "A documentary about a topic.",
            "tags": ["topic"],
            "script": "detail " * 1200,
            "video_keywords": ["company archive records"],
        })
        with patch.object(cervello, "chat_ollama", return_value=payload) as chat:
            content = cervello.genera_contenuto("topic")
        chat.assert_called_once()
        self.assertTrue(chat.call_args.kwargs["json_mode"])
        self.assertTrue(content["visual_beats"])
        self.assertIn("narration_anchor", content["visual_beats"][0])


if __name__ == "__main__":
    unittest.main()