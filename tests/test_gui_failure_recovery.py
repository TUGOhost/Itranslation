import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import desktop
from chunker import Chunk
from consistency import ConsistencyModel
from translator import translate_chapter


class GuiFailureRecoveryTests(unittest.TestCase):
    def test_failure_message_preserves_exception_message(self):
        self.assertEqual(
            desktop._failure_message(RuntimeError("original translation failure")),
            "original translation failure",
        )

        class EmptyMessageError(Exception):
            def __str__(self):
                return ""

        self.assertEqual(desktop._failure_message(EmptyMessageError()), "EmptyMessageError")

    def test_gui_checkpoint_path_is_stable_and_filename_safe(self):
        book_path = (
            "/Users/tg/Downloads/The Intelligent Investor, Rev_ Ed -- "
            "Graham, Benjamin & Jason Zweig.epub"
        )
        title = "Introduction:What This Book Expects to Accomplish"

        first = desktop._gui_checkpoint_path(book_path, 7, title)
        second = desktop._gui_checkpoint_path(book_path, 7, title)

        self.assertEqual(first, second)
        self.assertEqual(first.parent, desktop.PROJECT_ROOT / "cache")
        self.assertEqual(first.suffix, ".json")
        self.assertTrue(first.name.startswith("gui_"))
        self.assertNotRegex(first.name, r"[\s,:/\\\\]")

    def test_translate_chapter_resumes_from_checkpoint_after_partial_failure(self):
        chunks = [
            Chunk(id="chunk_0000", text="Hello.", start_sentence=0, end_sentence=0),
            Chunk(id="chunk_0001", text="World.", start_sentence=1, end_sentence=1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "checkpoint.json")
            first_run_calls = []

            def first_run_llm(system_prompt, user_prompt):
                first_run_calls.append(user_prompt)
                if len(first_run_calls) == 1:
                    return "你好。", {"prompt_tokens": 1, "completion_tokens": 1}
                raise RuntimeError("network dropped")

            first_translations, first_errors = translate_chapter(
                "Chapter",
                chunks,
                None,
                ConsistencyModel(),
                {},
                {},
                first_run_llm,
                {"max_retries": 1},
                checkpoint_path=checkpoint_path,
            )

            self.assertEqual(first_translations[0], "你好。")
            self.assertIn("[翻译失败:", first_translations[1])
            self.assertEqual(len(first_errors), 1)
            self.assertIn("network dropped", first_errors[0]["error"])

            second_run_calls = []

            def second_run_llm(system_prompt, user_prompt):
                second_run_calls.append(user_prompt)
                return "世界。", {"prompt_tokens": 1, "completion_tokens": 1}

            translations, errors = translate_chapter(
                "Chapter",
                chunks,
                None,
                ConsistencyModel(),
                {},
                {},
                second_run_llm,
                {"max_retries": 1},
                checkpoint_path=checkpoint_path,
            )

        self.assertEqual(translations, ["你好。", "世界。"])
        self.assertEqual(errors, [])
        self.assertEqual(len(first_run_calls), 2)
        self.assertEqual(len(second_run_calls), 1)


if __name__ == "__main__":
    unittest.main()
