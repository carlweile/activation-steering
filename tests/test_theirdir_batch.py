"""Tests for the batched-generation helpers in theirdir_lib (LABBOOK 2026-11-20 late).

Pure Python, no model. Shown FAILING on stubs first, then green. The real-model identity
proof (batched text == one-at-a-time text) is theirdir_batchcheck.py, not a unit test.
"""
import theirdir_lib as L


def test_clean_reply_cuts_at_first_stop_string_and_strips():
    assert L.clean_reply(" Paris.<|im_end|><|endoftext|>junk") == "Paris."
    assert L.clean_reply("A<|endoftext|>B<|im_end|>") == "A"
    assert L.clean_reply("  no stop  ") == "no stop"


def test_plan_batches_covers_every_index_once_and_respects_size():
    lengths = [5, 9, 5, 7, 9, 9, 3]
    batches = L.plan_batches(lengths, batch_size=2)
    assert sorted(i for b in batches for i in b) == list(range(7))
    assert all(1 <= len(b) <= 2 for b in batches)


def test_plan_batches_groups_similar_lengths_together():
    lengths = [3, 50, 3, 50]
    batches = L.plan_batches(lengths, batch_size=2)
    assert sorted(map(sorted, batches)) == [[0, 2], [1, 3]]


def test_plan_batches_equal_only_never_mixes_lengths():
    lengths = [4, 4, 5, 4, 5, 6]
    for b in L.plan_batches(lengths, batch_size=8, equal_only=True):
        assert len({lengths[i] for i in b}) == 1


def test_batched_map_returns_outputs_in_input_order():
    lengths = [9, 1, 5, 1, 9]
    fn = lambda idx: [f"out{i}" for i in idx]
    assert L.batched_map(lengths, fn, batch_size=2) == [f"out{i}" for i in range(5)]


def test_text_mismatches_lists_every_differing_index():
    a = ["x", "same", "y", "z"]
    b = ["x", "same", "Y", "z "]
    assert L.text_mismatches(a, b) == [2, 3]
    assert L.text_mismatches(a, list(a)) == []
