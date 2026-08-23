import unittest

import torch

from integrad import (
    adaptive_integrated_gradients,
    classification_score,
    integrated_gradients,
)


class IntegratedGradientsTest(unittest.TestCase):
    def test_linear_model_is_exact_and_complete(self):
        inputs = torch.tensor([2.0, -1.0, 3.0])
        baseline = torch.tensor([0.5, 1.0, -2.0])
        weights = torch.tensor([2.0, -3.0, 0.5])

        attributions, residual = integrated_gradients(
            lambda batch: batch @ weights + 4,
            inputs,
            baseline,
            steps=8,
        )

        torch.testing.assert_close(attributions, (inputs - baseline) * weights)
        torch.testing.assert_close(residual, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_recovers_change_hidden_by_endpoint_saturation(self):
        inputs = torch.tensor([2.0], requires_grad=True)
        endpoint_score = 1 - torch.relu(1 - inputs)
        endpoint_gradient = torch.autograd.grad(endpoint_score, inputs)[0]

        attributions, residual = integrated_gradients(
            lambda batch: (1 - torch.relu(1 - batch)).squeeze(-1),
            inputs.detach(),
            steps=1024,
        )

        torch.testing.assert_close(endpoint_gradient, torch.tensor([0.0]))
        torch.testing.assert_close(attributions, torch.tensor([1.0]), atol=2e-3, rtol=0)
        self.assertLess(abs(residual.item()), 2e-3)

    def test_functionally_equivalent_models_match(self):
        inputs = torch.tensor([1.5, -2.0, 0.75])
        baseline = torch.zeros_like(inputs)

        left, _ = integrated_gradients(
            lambda batch: (batch[:, 0] * batch[:, 1]) * batch[:, 2],
            inputs,
            baseline,
        )
        right, _ = integrated_gradients(
            lambda batch: batch[:, 0] * (batch[:, 1] * batch[:, 2]),
            inputs,
            baseline,
        )

        torch.testing.assert_close(left, right)

    def test_chunking_does_not_change_result(self):
        inputs = torch.tensor([0.25, -1.5, 2.0])
        score = lambda batch: torch.sigmoid(batch.square().sum(dim=1))

        whole = integrated_gradients(score, inputs, steps=64)
        chunked = integrated_gradients(score, inputs, steps=64, batch_size=7)

        torch.testing.assert_close(whole[0], chunked[0])
        torch.testing.assert_close(whole[1], chunked[1])

    def test_adaptive_steps_stop_after_completeness_converges(self):
        attributions, residual, steps, converged = adaptive_integrated_gradients(
            lambda batch: batch.squeeze(-1).pow(3),
            torch.tensor([2.0]),
            steps=1,
            max_steps=64,
            relative_tolerance=1e-3,
        )

        self.assertTrue(converged)
        self.assertGreater(steps, 1)
        torch.testing.assert_close(attributions, torch.tensor([8.0]), atol=1e-2, rtol=0)
        self.assertLess(abs(residual.item()), 1e-2)

    def test_classification_score_selects_requested_output(self):
        logits = torch.tensor([[1.0, 3.0], [2.0, -1.0]])

        torch.testing.assert_close(
            classification_score(logits, 1, 0, "logit"),
            torch.tensor([3.0, -1.0]),
        )
        torch.testing.assert_close(
            classification_score(logits, 1, 0, "margin"),
            torch.tensor([2.0, -3.0]),
        )
        torch.testing.assert_close(
            classification_score(logits, 1, 0, "probability"),
            logits.softmax(dim=-1)[:, 1],
        )


if __name__ == "__main__":
    unittest.main()
