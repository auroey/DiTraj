"""Dependency-free argument and trajectory checks; no model/GPU/network use."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))

from run_low_vram import negative_prompt_from_source, parse_args, trajectory_for_frames


class LowVramRunnerTests(unittest.TestCase):
    def test_upstream_default_dimensions(self):
        args = parse_args(["--output", "fresh-test-output.mp4"])
        self.assertEqual((args.num_frames, args.height, args.width, args.steps), (81, 480, 832, 50))
        self.assertEqual((args.seed, args.mask_step, args.fix_rope_step), (25234, 30, 5))
        tokens = ((args.num_frames - 1) // 4 + 1) * (args.height // 16) * (args.width // 16)
        self.assertEqual(tokens, 32760)
        self.assertEqual(tokens**2, 1073217600)

    def test_smoke_dimensions(self):
        args = parse_args(["--output", "smoke.mp4", "--num-frames", "9",
                           "--height", "160", "--width", "288", "--steps", "2"])
        self.assertEqual((args.num_frames, args.height, args.width, args.steps), (9, 160, 288, 2))

    def test_non_video_frame_count_rejected(self):
        for count in (0, 1, 8, 80):
            with self.subTest(count=count), self.assertRaises(SystemExit):
                parse_args(["--output", "test.mp4", "--num-frames", str(count)])

    def test_non_patch_aligned_dimension_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--output", "test.mp4", "--height", "150"])

    def test_bad_steps_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--output", "test.mp4", "--steps", "0"])

    def test_bad_prompt_index_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--output", "test.mp4", "--prompt-index", "-1"])

    def test_trajectory_covers_complete_video(self):
        for count in (5, 9, 33, 81):
            boxes = trajectory_for_frames(count)
            self.assertEqual(boxes[0][0], 0)
            self.assertEqual(boxes[-1][0], count - 1)
            self.assertEqual(boxes[0][1:], [0.3, 0.7, 0.1, 0.4])
            self.assertEqual(boxes[-1][1:], [0.3, 0.7, 0.7, 1.0])

    def test_negative_prompt_literal_without_executing_source(self):
        source = 'raise RuntimeError("do not execute")\nnegative_prompt = "bad quality"\n'
        self.assertEqual(negative_prompt_from_source(source), "bad quality")

    def test_negative_prompt_missing_rejected(self):
        with self.assertRaises(ValueError):
            negative_prompt_from_source("unrelated = 1\n")


if __name__ == "__main__":
    unittest.main()
