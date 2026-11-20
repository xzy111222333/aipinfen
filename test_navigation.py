import unittest

from playwright.sync_api import sync_playwright

from shenlun import NavigationStopped, open_page


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        cls.browser = playwright.chromium.launch(channel="chrome")
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page()
        self.addCleanup(self.page.close)
        self.requests = []

    def serve(self, status, body=""):
        def respond(route):
            self.requests.append(route.request.url)
            route.fulfill(status=status, content_type="text/html; charset=utf-8", body=body)

        self.page.route("**/*", respond)

    def test_regular_page_opens(self):
        self.serve(200, "<h1>试卷目录</h1>")
        open_page(self.page, "https://local-test.invalid/papers")
        self.assertEqual(self.page.get_by_role("heading").inner_text(), "试卷目录")

    def test_denied_and_rate_limited_requests_stop_without_retrying(self):
        for status in (401, 403, 429, 503):
            with self.subTest(status=status):
                self.page.unroute("**/*")
                self.requests.clear()
                self.serve(status, "<h1>请求失败</h1>")
                url = "https://local-test.invalid/papers"
                with self.assertRaisesRegex(NavigationStopped, f"HTTP {status}"):
                    open_page(self.page, url)
                self.assertEqual(self.requests, [url])

    def test_visible_verification_notice_stops_without_retrying(self):
        self.serve(200, '<p role="alert">请完成人机验证</p>')
        url = "https://local-test.invalid/papers"
        with self.assertRaises(NavigationStopped):
            open_page(self.page, url)
        self.assertEqual(self.requests, [url])

    def test_navigation_failure_stops_without_retrying(self):
        self.serve(403)
        url = "https://local-test.invalid/papers"
        with self.assertRaises(NavigationStopped):
            open_page(self.page, url)
        self.assertEqual(self.requests, [url])


if __name__ == "__main__":
    unittest.main()
