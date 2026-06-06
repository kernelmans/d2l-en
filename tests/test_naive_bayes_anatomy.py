"""
Anatomy tests for the Naive Bayes section in:

chapter_appendix-mathematics-for-deep-learning/naive-bayes.md

These tests do not download MNIST and do not depend on torchvision.
They isolate the mathematical core of the chapter:

- estimate class priors P_y by counting labels
- estimate pixel likelihoods P_xy with Laplace smoothing
- score a binary image with the naive Bayes independence assumption
- use log-probabilities to avoid numerical underflow

Run from the repository root with:

    python -m pytest tests/test_naive_bayes_anatomy.py
"""

import math

import torch


def estimate_priors(labels, num_classes):
    """Estimate P_y[y] = p(y) by counting labels."""
    counts = torch.zeros(num_classes)
    for y in range(num_classes):
        counts[y] = (labels == y).sum()
    return counts / counts.sum(), counts


def estimate_pixel_likelihoods(images, labels, num_classes):
    """
    Estimate P_xy[y, i, j] = p(pixel(i,j)=1 | y) with Laplace smoothing.

    images: [num_examples, height, width], binary values 0 or 1
    labels: [num_examples]
    """
    _, height, width = images.shape
    priors, class_counts = estimate_priors(labels, num_classes)

    pixel_counts = torch.zeros((num_classes, height, width))
    for y in range(num_classes):
        pixel_counts[y] = images[labels == y].sum(dim=0)

    # Binary pixel: possible values are 0 and 1, hence +2 in the denominator.
    likelihoods = (pixel_counts + 1) / (class_counts + 2).reshape(num_classes, 1, 1)
    return priors, likelihoods


def bayes_pred_direct(x, priors, likelihoods):
    """Direct product version: p(x|y)p(y). This can underflow for large inputs."""
    x = x.unsqueeze(0)
    p_xy = likelihoods * x + (1 - likelihoods) * (1 - x)
    p_xy = p_xy.reshape(likelihoods.shape[0], -1).prod(dim=1)
    return p_xy * priors


def bayes_pred_stable(x, priors, likelihoods):
    """Stable log version: log p(x|y) + log p(y)."""
    x = x.unsqueeze(0)
    log_likelihoods = torch.log(likelihoods)
    log_likelihoods_neg = torch.log(1 - likelihoods)
    log_priors = torch.log(priors)

    log_p_xy = log_likelihoods * x + log_likelihoods_neg * (1 - x)
    log_p_xy = log_p_xy.reshape(likelihoods.shape[0], -1).sum(dim=1)
    return log_p_xy + log_priors


def test_estimate_priors_by_counting_labels():
    labels = torch.tensor([0, 0, 0, 1, 1])

    priors, counts = estimate_priors(labels, num_classes=2)

    assert torch.equal(counts, torch.tensor([3.0, 2.0]))
    assert torch.allclose(priors, torch.tensor([0.6, 0.4]))


def test_laplace_smoothed_pixel_likelihoods():
    # Two classes, 2x2 binary images.
    # Class 0 appears 3 times; class 1 appears 2 times.
    images = torch.tensor([
        [[1, 0], [1, 0]],
        [[1, 0], [1, 0]],
        [[1, 1], [1, 0]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 1]],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1])

    priors, likelihoods = estimate_pixel_likelihoods(images, labels, num_classes=2)

    assert torch.allclose(priors, torch.tensor([0.6, 0.4]))
    assert likelihoods.shape == (2, 2, 2)

    # For class 0, pixel (0,0) is on in all 3 class-0 examples.
    # Laplace smoothing: (3 + 1) / (3 + 2) = 4/5.
    assert torch.allclose(likelihoods[0, 0, 0], torch.tensor(4 / 5))

    # For class 1, pixel (0,0) is on in 0 of 2 class-1 examples.
    # Laplace smoothing: (0 + 1) / (2 + 2) = 1/4.
    assert torch.allclose(likelihoods[1, 0, 0], torch.tensor(1 / 4))

    # Smoothing keeps probabilities strictly between 0 and 1.
    assert torch.all(likelihoods > 0)
    assert torch.all(likelihoods < 1)


def test_stable_and_direct_prediction_choose_same_class_on_small_input():
    images = torch.tensor([
        [[1, 0], [1, 0]],
        [[1, 0], [1, 0]],
        [[1, 1], [1, 0]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 1]],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1])
    priors, likelihoods = estimate_pixel_likelihoods(images, labels, num_classes=2)

    x_like_class_0 = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)

    direct_scores = bayes_pred_direct(x_like_class_0, priors, likelihoods)
    stable_log_scores = bayes_pred_stable(x_like_class_0, priors, likelihoods)

    assert direct_scores.argmax().item() == 0
    assert stable_log_scores.argmax().item() == 0
    assert direct_scores.argmax().item() == stable_log_scores.argmax().item()


def test_log_version_avoids_underflow_for_many_small_probabilities():
    # This mirrors the chapter's point: multiplying many small probabilities
    # can collapse to zero, while summing logs remains representable.
    a = 0.1

    direct = a ** 784
    log_value = 784 * math.log(a)

    assert direct == 0.0
    assert math.isfinite(log_value)
    assert log_value < 0
