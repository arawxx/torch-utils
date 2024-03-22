from typing import List, Tuple, Callable

import torch
import torch.nn as nn


__all__ = ['CombinedLoss']


class CombinedLoss(nn.Module):
    """
    A PyTorch Module that combines multiple loss functions for use in a neural network.

    Attributes:
        loss_functions (List[nn.Module] | Tuple[nn.Module]): A list or tuple of PyTorch loss functions to be combined.
        weights (List[float] | Tuple[float]): A list or tuple of weights for each loss function. Defaults to equal weights for all loss functions.
        cosine_similarity_targets (torch.Tensor): The target values for cosine similarity. Defaults to a tensor of ones.
        log_softmax (nn.Module): A PyTorch module for log softmax. Defaults to nn.LogSoftmax(dim=1).
        sigmoid (nn.Module): A PyTorch module for sigmoid. Defaults to nn.Sigmoid().
        loss_fn_dict (dict): A dictionary mapping the names of loss functions to the corresponding methods in this class.

    Methods:
        forward(predictions: torch.Tensor, ground_truths: torch.Tensor, cosine_similarity_targets: torch.Tensor = None) -> torch.Tensor:
            Computes the combined loss for the given predictions and ground truths.
    """
    def __init__(
        self,
        loss_functions: List[nn.Module] | Tuple[nn.Module],
        weights: List[float] | Tuple[float] = None,
        cosine_similarity_targets: torch.Tensor = None,
        use_log_softmax: bool = True,
        use_sigmoid: bool = True,
    ) -> None:
        """
        Initializes the CombinedLoss module.

        Args:
            loss_functions (List[nn.Module] | Tuple[nn.Module]): A list or tuple of PyTorch loss functions to be combined.
            weights (List[float] | Tuple[float], optional): A list or tuple of weights for each loss function. Defaults to equal
                weights for all loss functions.
            cosine_similarity_targets (torch.Tensor, optional): The target values for cosine similarity if `CosineEmbeddingLoss`
                was among the loss functions. Otherwise it's irrelevant. Defaults to a tensor of ones (maximizes simalirity).
            use_log_softmax (bool, optional): Whether to use log softmax on the predictions if `NLLLoss` was among the loss
                functions. Otherwise it's irrelevant. Defaults to True.
            use_sigmoid (bool, optional): Whether to use sigmoid on the predictions if `BCELoss` was among the loss functions.
                Otherwise it's irrelevant. Defaults to True.
        """
        super().__init__()
        self.loss_functions = loss_functions
        self.weights = weights if weights is not None else tuple([1.0 / len(loss_functions)] * len(loss_functions))

        assert len(self.loss_functions) == len(self.weights), \
        'The number of loss functions and weights must be equal.'

        # Cosine Similarity
        self.cosine_similarity_targets = cosine_similarity_targets if cosine_similarity_targets is not None else torch.ones(1)

        # Negative Log-Likelihood
        self.log_softmax = nn.LogSoftmax(dim=1) if use_log_softmax else nn.Identity()

        # Binary Cross Entropy
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

        # Create the loss function dictionary
        self.loss_fn_dict = {
            'CosineEmbeddingLoss': self._cosine_loss,
            'NLLLoss': self._nll_loss,
            'BCELoss': self._bce_loss,
            'default': self._default_loss,
        }

    def forward(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        cosine_similarity_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the combined loss based on the given predictions and ground truths.

        Args:
            predictions (torch.Tensor): The predicted values.
            ground_truths (torch.Tensor): The ground truth values.
            cosine_similarity_targets (torch.Tensor, optional): The cosine similarity targets. Defaults to None.
                Has priority over the default `cosine_similarity_targets` class attribute.

        Returns:
            torch.Tensor: The combined loss.

        """
        device = predictions.device
        total_loss = torch.zeros(1).to(device)

        for idx, loss_fn in enumerate(self.loss_functions):
            loss_fn_callable = self.loss_fn_dict.get(loss_fn.__class__.__name__, self.loss_fn_dict['default'])
            total_loss += loss_fn_callable(loss_fn, predictions, ground_truths, cosine_similarity_targets) * self.weights[idx]

        return total_loss

    def _nll_loss(
        self,
        loss_fn: Callable,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        _: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the negative log-likelihood loss.

        Args:
            loss_fn (Callable): The loss function to be applied.
            predictions (torch.Tensor): The predicted values.
            ground_truths (torch.Tensor): The ground truth values.
            _ (torch.Tensor, optional): This argument is included for consistency with
                other loss functions, but it is not used in this function.

        Returns:
            torch.Tensor: The computed negative log-likelihood loss.
        """
        return loss_fn(self.log_softmax(predictions), ground_truths)

    def _bce_loss(
        self,
        loss_fn: Callable,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        _: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the binary cross-entropy loss.

        Args:
            loss_fn (Callable): The loss function to be applied.
            predictions (torch.Tensor): The predicted values.
            ground_truths (torch.Tensor): The ground truth values.
            _ (torch.Tensor, optional): This argument is included for consistency with
                other loss functions, but it is not used in this function.

        Returns:
            torch.Tensor: The calculated binary cross-entropy loss.
        """
        return loss_fn(self.sigmoid(predictions), ground_truths)

    def _cosine_loss(
        self,
        loss_fn: Callable,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        cosine_similarity_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the cosine loss between the predictions and ground truths.

        Args:
            loss_fn (Callable): The loss function to be used for calculating the loss.
            predictions (torch.Tensor): The predicted values.
            ground_truths (torch.Tensor): The ground truth values.
            cosine_similarity_targets (torch.Tensor): The target values for cosine similarity.

        Returns:
            torch.Tensor: The calculated loss.
        """
        return loss_fn(
            predictions, ground_truths,
            target=cosine_similarity_targets.to(predictions.device) \
                if cosine_similarity_targets is not None else \
                self.cosine_similarity_targets.to(predictions.device),
        )

    def _default_loss(
        self,
        loss_fn: Callable,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        _: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the default loss using the given loss function.

        Args:
            loss_fn (Callable): The loss function to be used.
            predictions (torch.Tensor): The predicted values.
            ground_truths (torch.Tensor): The ground truth values.
            _ (torch.Tensor, optional): This argument is included for consistency with
                other loss functions, but it is not used in this function.

        Returns:
            torch.Tensor: The calculated loss.
        """
        return loss_fn(predictions, ground_truths)
