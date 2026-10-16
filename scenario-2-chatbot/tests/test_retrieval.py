import uuid

from app.services.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_items_present_in_both_rankings() -> None:
    shared = uuid.uuid4()
    dense_only = uuid.uuid4()
    lexical_only = uuid.uuid4()

    scores = reciprocal_rank_fusion(
        [[shared, dense_only], [lexical_only, shared]],
        k=60,
    )

    assert scores[shared] > scores[dense_only]
    assert scores[shared] > scores[lexical_only]
    assert 0 < scores[shared] <= 1


def test_rrf_does_not_mix_incompatible_raw_scores() -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    scores = reciprocal_rank_fusion([[first, second], [second, first]], k=60)
    assert scores[first] == scores[second]

