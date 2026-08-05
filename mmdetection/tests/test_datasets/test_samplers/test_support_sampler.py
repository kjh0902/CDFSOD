from mmdet.datasets.samplers import SupportSampler


def test_support_sampler_visits_complete_dataset_in_order():
    dataset = list(range(5))
    sampler = SupportSampler(dataset, shuffle=False, seed=42)

    assert list(sampler) == [0, 1, 2, 3, 4]
    assert len(sampler) == len(dataset)
