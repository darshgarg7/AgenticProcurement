import unittest
from unittest.mock import patch

from web_demo.server import app


class TestWebDemoApi(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict("os.environ", {"PROCUREMENT_API_KEY": ""})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_run_episode_post_returns_expected_shape(self):
        client = app.test_client()
        response = client.post(
            "/api/run-episode",
            json={
                "persona": "balanced",
                "eps_reg": 0.8,
                "eps_var": 0.8,
                "max_steps": 5,
                "seed": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("outcome", payload)
        self.assertIn("steps", payload)
        self.assertIn("epi_unc_history", payload)
        self.assertIn("action_counts", payload)

    def test_run_episode_stream_emits_complete_event(self):
        client = app.test_client()
        response = client.get(
            "/api/run-episode-stream?persona=balanced&eps_reg=0.8&eps_var=0.8"
            "&max_steps=3&seed=1"
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("event: step", body)
        self.assertIn("event: complete", body)

    def test_invalid_episode_parameter_returns_400(self):
        client = app.test_client()
        response = client.post("/api/run-episode", json={"max_steps": "not-an-int"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_health_and_ready_endpoints_are_public(self):
        client = app.test_client()
        with patch.dict("os.environ", {"PROCUREMENT_API_KEY": "secret"}):
            health = client.get("/healthz")
            ready = client.get("/readyz")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")
        self.assertIn("X-Request-ID", health.headers)

    def test_api_key_required_when_configured(self):
        client = app.test_client()
        with patch.dict("os.environ", {"PROCUREMENT_API_KEY": "secret"}):
            unauthenticated = client.post("/api/run-episode", json={"max_steps": 1})
            authenticated = client.post(
                "/api/run-episode",
                headers={"X-API-Key": "secret", "X-Request-ID": "test-request"},
                json={"max_steps": 1},
            )

        self.assertEqual(unauthenticated.status_code, 401)
        self.assertEqual(authenticated.status_code, 200)
        self.assertEqual(authenticated.headers["X-Request-ID"], "test-request")


if __name__ == "__main__":
    unittest.main()
