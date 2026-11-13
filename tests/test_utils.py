import unittest

from yt_rag import _group_transcript, format_timestamp, parse_video_id


class URLTests(unittest.TestCase):
    def test_supported_youtube_urls(self):
        video_id = "dQw4w9WgXcQ"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}&t=3",
            f"https://youtu.be/{video_id}",
            f"https://youtube.com/shorts/{video_id}",
            f"https://www.youtube.com/embed/{video_id}",
            f"https://youtube.com/live/{video_id}",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(parse_video_id(url), video_id)

    def test_invalid_url_rejected(self):
        for url in ("", "https://example.com/watch?v=dQw4w9WgXcQ", "not a url"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    parse_video_id(url)


class TranscriptTests(unittest.TestCase):
    def test_timestamp_format(self):
        self.assertEqual(format_timestamp(0), "0:00")
        self.assertEqual(format_timestamp(65), "1:05")
        self.assertEqual(format_timestamp(3661), "1:01:01")

    def test_grouping_preserves_start_and_end(self):
        items = [
            {"text": "First caption.", "start": 2.0, "duration": 1.0},
            {"text": "Second caption.", "start": 4.0, "duration": 2.0},
        ]
        docs = _group_transcript(items, target_chars=100)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["start_seconds"], 2.0)
        self.assertEqual(docs[0].metadata["end_seconds"], 6.0)
        self.assertIn("Second caption.", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
