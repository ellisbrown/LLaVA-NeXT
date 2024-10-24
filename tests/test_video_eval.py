# test_video_eval.py

import unittest
import os
import sys

# Adjust the import path to include the parent directory
this_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_dir)
sys.path.insert(0, parent_dir)

from playground.video_eval import clean_answer

class TestVideoEval(unittest.TestCase):
    def test_clean_answer(self):
        """Test the clean_answer function with various inputs."""
        # Test data with choices
        test_data = [
            {"gt_answer": "B", "pred": "(A) yellow vespa", "choices": ["A", "B", "C", "D"], "expected": False},
            {"gt_answer": "D", "pred": "D) Yellow", "choices": ["A", "B", "C", "D"], "expected": True},
            {"gt_answer": "C", "pred": "D", "choices": ["A", "B", "C", "D"], "expected": False},
            {"gt_answer": "B", "pred": "B) no", "choices": ["A", "B"], "expected": True},
            {"gt_answer": "A", "pred": "A", "choices": ["A", "B", "C", "D"], "expected": True},
            {"gt_answer": "A", "pred": "B) no", "choices": ["A", "B"], "expected": False},
            {"gt_answer": "A", "pred": "A) yellow vespa", "choices": ["A", "B", "C", "D"], "expected": True},
            {"gt_answer": "A", "pred": "The correct answer is A.", "choices": ["A", "B", "C", "D"], "expected": True},
            {"gt_answer": "B", "pred": "B: a chicken", "choices": ["A", "B", "C", "D"], "expected": True},
        ]

        for item in test_data:
            pred = item['pred']
            gt_answer = item['gt_answer']
            expected = item['expected']
            choices = item['choices']

            # Use the clean_answer function to extract answers
            cleaned_pred = clean_answer(pred, choices)
            cleaned_gt_answer = clean_answer(gt_answer, choices)

            # Compare the cleaned predicted answer with the ground truth
            result = cleaned_pred.lower() == cleaned_gt_answer.lower()

            # Assert that the result matches the expected outcome
            self.assertEqual(
                result, expected,
                f"Failed for GT: '{gt_answer}', Pred: '{pred}'. "
                f"Cleaned GT: '{cleaned_gt_answer}', Cleaned Pred: '{cleaned_pred}'"
            )

if __name__ == '__main__':
    unittest.main()
