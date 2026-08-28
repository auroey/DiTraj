"""CPU-only contracts for upstream trajectory helpers, including known failures.

Only the three pure function definitions are compiled from utils.py. Importing
that module normally would also import torch and torchvision. Expected failures
describe upstream defects; an unexpected success calls for revisiting the marker.
"""

import ast
import copy
import unittest
from pathlib import Path


HELPER_NAMES = ("arg_to_bboxs", "bboxs_to_arg", "plan_path")


def load_upstream_helpers():
    source_path = Path(__file__).resolve().parents[1] / "utils.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
    definitions = {
        node.name: node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
    }
    missing = set(HELPER_NAMES) - definitions.keys()
    if missing:
        raise AssertionError(f"Missing upstream helpers: {sorted(missing)}")
    helper_tree = ast.Module(
        body=[definitions[name] for name in HELPER_NAMES], type_ignores=[]
    )
    namespace = {}
    exec(compile(helper_tree, str(source_path), "exec"), namespace)
    return namespace


UPSTREAM = load_upstream_helpers()
arg_to_bboxs = UPSTREAM["arg_to_bboxs"]
bboxs_to_arg = UPSTREAM["bboxs_to_arg"]
plan_path = UPSTREAM["plan_path"]


class TrajectoryAssertions(unittest.TestCase):
    def assert_path_almost_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for frame, (actual_box, expected_box) in enumerate(zip(actual, expected)):
            with self.subTest(frame=frame):
                self.assertIsInstance(actual_box, list)
                self.assertEqual(len(actual_box), 4)
                for coordinate, (actual_value, expected_value) in enumerate(
                    zip(actual_box, expected_box)
                ):
                    with self.subTest(coordinate=coordinate):
                        self.assertAlmostEqual(actual_value, expected_value)


class UpstreamTrajectoryTests(TrajectoryAssertions):
    def test_loader_exposes_only_pure_helpers(self):
        self.assertEqual(set(UPSTREAM), {"__builtins__", *HELPER_NAMES})
        for name in HELPER_NAMES:
            self.assertTrue(callable(UPSTREAM[name]))

    def test_parser_preserves_integer_frames_and_float_coordinates(self):
        boxes = arg_to_bboxs("0,0.1,0.3,0.2,0.4,4,0.3,0.9,0.1,0.9")
        self.assertEqual(
            boxes, [[0, 0.1, 0.3, 0.2, 0.4], [4, 0.3, 0.9, 0.1, 0.9]]
        )
        for box in boxes:
            self.assertIs(type(box[0]), int)
            for coordinate in box[1:]:
                self.assertIs(type(coordinate), float)

    def test_serializer_keeps_keyframe_and_coordinate_order(self):
        boxes = [[0, 0.1, 0.3, 0.2, 0.4], [4, 0.3, 0.9, 0.1, 0.9]]
        self.assertEqual(
            bboxs_to_arg(boxes), "0,0.1,0.3,0.2,0.4,4,0.3,0.9,0.1,0.9"
        )

    def test_serialization_round_trip_for_multiple_keyframes(self):
        boxes = [
            [0, 0.1, 0.3, 0.2, 0.4],
            [2, 0.3, 0.5, 0.4, 0.6],
            [5, 0.0, 0.2, 0.1, 0.3],
        ]
        self.assertEqual(arg_to_bboxs(bboxs_to_arg(boxes)), boxes)

    def test_linear_path_uses_default_49_frame_length(self):
        boxes = [[0, 0.2, 0.6, 0.1, 0.3], [48, 0.2, 0.6, 0.7, 0.9]]
        expected = [
            [0.2, 0.6, 0.1 + frame / 80, 0.3 + frame / 80]
            for frame in range(49)
        ]
        self.assert_path_almost_equal(plan_path(boxes), expected)

    def test_linear_path_supports_explicit_81_frame_length(self):
        boxes = [[0, 0.3, 0.7, 0.1, 0.4], [80, 0.3, 0.7, 0.7, 1.0]]
        expected = [
            [0.3, 0.7, 0.1 + 0.6 * frame / 80, 0.4 + 0.6 * frame / 80]
            for frame in range(81)
        ]
        self.assert_path_almost_equal(plan_path(boxes, video_length=81), expected)

    def test_nonuniform_multiple_keyframes_change_direction_without_duplicates(self):
        boxes = [
            [0, 0.1, 0.3, 0.2, 0.4],
            [2, 0.3, 0.5, 0.4, 0.6],
            [5, 0.0, 0.2, 0.1, 0.3],
        ]
        expected = [
            [0.1, 0.3, 0.2, 0.4],
            [0.2, 0.4, 0.3, 0.5],
            [0.3, 0.5, 0.4, 0.6],
            [0.2, 0.4, 0.3, 0.5],
            [0.1, 0.3, 0.2, 0.4],
            [0.0, 0.2, 0.1, 0.3],
        ]
        self.assert_path_almost_equal(plan_path(boxes, video_length=6), expected)

    def test_varying_size_interpolates_all_four_coordinates_independently(self):
        boxes = [[0, 0.1, 0.3, 0.2, 0.4], [4, 0.3, 0.9, 0.1, 0.9]]
        expected = [
            [0.1, 0.3, 0.2, 0.4],
            [0.15, 0.45, 0.175, 0.525],
            [0.2, 0.6, 0.15, 0.65],
            [0.25, 0.75, 0.125, 0.775],
            [0.3, 0.9, 0.1, 0.9],
        ]
        self.assert_path_almost_equal(plan_path(boxes, video_length=5), expected)

    def test_stationary_path_keeps_the_same_box(self):
        box = [0.2, 0.7, 0.1, 0.8]
        boxes = [[0, *box], [8, *box]]
        self.assert_path_almost_equal(
            plan_path(boxes, video_length=9), [box[:] for _ in range(9)]
        )

    def test_single_frame_path_needs_no_interpolation(self):
        box = [0.2, 0.7, 0.1, 0.8]
        self.assert_path_almost_equal(plan_path([[0, *box]], video_length=1), [box])

    def test_path_planning_does_not_mutate_input_keyframes(self):
        boxes = [[0, 0.1, 0.3, 0.2, 0.4], [4, 0.3, 0.9, 0.1, 0.9]]
        original = copy.deepcopy(boxes)
        plan_path(boxes, video_length=5)
        self.assertEqual(boxes, original)


class KnownUpstreamTrajectoryFailures(TrajectoryAssertions):
    """Desired boundary contracts that the unchanged upstream code violates."""

    def assert_complete_linear_path(self, first_frame, last_frame):
        expected = [
            [coordinate + frame * 0.1 for coordinate in (0.1, 0.3, 0.2, 0.4)]
            for frame in range(5)
        ]
        boxes = [
            [first_frame, *expected[first_frame]],
            [last_frame, *expected[last_frame]],
        ]
        self.assert_path_almost_equal(plan_path(boxes, video_length=5), expected)

    @unittest.expectedFailure
    def test_duplicate_frame_should_raise_a_validation_error(self):
        # Upstream currently divides by zero instead of validating the frames.
        boxes = [[0, 0.1, 0.3, 0.2, 0.4], [0, 0.3, 0.5, 0.4, 0.6]]
        with self.assertRaises(ValueError):
            plan_path(boxes, video_length=1)

    @unittest.expectedFailure
    def test_one_missing_first_frame_should_prepend_a_box(self):
        # Upstream prepends four floats, making eight entries instead of five.
        self.assert_complete_linear_path(first_frame=1, last_frame=4)

    @unittest.expectedFailure
    def test_two_missing_first_frames_should_prepend_boxes(self):
        # A second prepend indexes the first float and raises TypeError.
        self.assert_complete_linear_path(first_frame=2, last_frame=4)

    @unittest.expectedFailure
    def test_one_missing_last_frame_should_append_a_box(self):
        # Upstream appends four floats, making eight entries instead of five.
        self.assert_complete_linear_path(first_frame=0, last_frame=3)

    @unittest.expectedFailure
    def test_two_missing_last_frames_should_append_boxes(self):
        # A second append indexes the last float and raises TypeError.
        self.assert_complete_linear_path(first_frame=0, last_frame=2)


if __name__ == "__main__":
    unittest.main()
