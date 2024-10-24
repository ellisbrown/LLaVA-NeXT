import os

this_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_dir)
os.sys.path.insert(0, parent_dir)

from playground.video_eval import clean_answer, is_correct_answer

# Test data with choices
test_data = [
    {"gt_answer": "B", "pred": "(A) yellow vespa", "choices": ["A", "B", "C", "D"], "is_correct": False},
    {"gt_answer": "D", "pred": "D) Yellow", "choices": ["A", "B", "C", "D"], "is_correct": True},
    {"gt_answer": "C", "pred": "D", "choices": ["A", "B", "C", "D"], "is_correct": False},
    {"gt_answer": "B", "pred": "B) no", "choices": ["A", "B"], "is_correct": True},
    {"gt_answer": "A", "pred": "A", "choices": ["A", "B", "C", "D"], "is_correct": True},
    {"gt_answer": "A", "pred": "B) no", "choices": ["A", "B"], "is_correct": False},
    {"gt_answer": "A", "pred": "A) yellow vespa", "choices": ["A", "B", "C", "D"], "is_correct": True},
    {"gt_answer": "A", "pred": "The correct answer is A.", "choices": ["A", "B", "C", "D"], "is_correct": True},
    {"gt_answer": "B", "pred": "B: a chicken", "choices": ["A", "B", "C", "D"], "is_correct": True},
]

# Run tests
for item in test_data:
    pred = item['pred']
    gt_answer = item['gt_answer']
    expected = item['is_correct']
    choices = item['choices']
    # result = is_correct_answer(pred, gt_answer, choices)
    cleaned_pred = clean_answer(pred, choices)
    cleaned_gt_answer = clean_answer(gt_answer, choices)
    result = cleaned_pred.lower() == cleaned_gt_answer.lower()
    item['fn_correct'] = result
    pass_fail = 'PASS' if result == expected else 'FAIL'
    print(f"[{pass_fail}] GT: {gt_answer}, Pred: {pred}, Cleaned GT: {cleaned_gt_answer}, Cleaned Pred: {cleaned_pred}, Result: {result}, Expected: {expected}")

