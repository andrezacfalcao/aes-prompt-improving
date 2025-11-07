"""
Evaluation metrics for automated essay scoring
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from typing import Dict, Optional


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    round_predictions: bool = True
) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK)

    Métrica principal usada no ASAP competition.
    QWK mede a concordância entre os scores preditos e verdadeiros,
    penalizando discordâncias quadraticamente.

    Args:
        y_true: Ground truth scores (inteiros)
        y_pred: Predicted scores (float ou int)
        round_predictions: Se deve arredondar predições para inteiros

    Returns:
        QWK score (0.0 = aleatório, 1.0 = concordância perfeita)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if round_predictions:
        y_pred = np.round(y_pred).astype(int)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    except Exception as e:
        print(f"Error calculating QWK: {e}")
        qwk = 0.0

    return float(qwk)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient

    Mede correlação linear entre scores verdadeiros e preditos.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        Pearson correlation coefficient (-1 a 1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) < 2:
        return 0.0

    try:
        corr, p_value = pearsonr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception as e:
        print(f"Error calculating Pearson: {e}")
        return 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Spearman rank correlation coefficient

    Mede correlação de ranking entre scores verdadeiros e preditos.
    Mais robusto a outliers que Pearson.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        Spearman correlation coefficient (-1 a 1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) < 2:
        return 0.0

    try:
        corr, p_value = spearmanr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception as e:
        print(f"Error calculating Spearman: {e}")
        return 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error

    Mede a magnitude média dos erros de predição.

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        RMSE value (menor é melhor)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    try:
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return float('inf')


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        MAE value (menor é melhor)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    round_for_qwk: bool = True,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores
        round_for_qwk: Se deve arredondar predições para QWK
        prefix: Prefixo para nomes das métricas (ex: "val_")

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        f"{prefix}qwk": quadratic_weighted_kappa(y_true, y_pred, round_for_qwk),
        f"{prefix}pearson": pearson_correlation(y_true, y_pred),
        f"{prefix}spearman": spearman_correlation(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}mae": mae(y_true, y_pred),
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: Optional[str] = None):
    """
    Print metrics em formato formatado

    Args:
        metrics: Dictionary de métricas
        title: Título opcional
    """
    if title:
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")

    print(f"{'Metric':<20} {'Value':>10}")
    print(f"{'-'*60}")

    for metric_name, value in sorted(metrics.items()):
        if 'qwk' in metric_name or 'pearson' in metric_name or 'spearman' in metric_name:
            print(f"{metric_name:<20} {value:>10.4f}")
        else:
            print(f"{metric_name:<20} {value:>10.3f}")

    print(f"{'='*60}\n")


def compute_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None
) -> float:
    """
    Wrapper function for quadratic_weighted_kappa for backward compatibility

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores
        min_score: Minimum score (unused, for API compatibility)
        max_score: Maximum score (unused, for API compatibility)

    Returns:
        QWK score
    """
    return quadratic_weighted_kappa(y_true, y_pred, round_predictions=False)


def compute_correlations(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute both Pearson and Spearman correlations

    Args:
        y_true: Ground truth scores
        y_pred: Predicted scores

    Returns:
        Dictionary with 'pearson' and 'spearman' keys
    """
    return {
        'pearson': pearson_correlation(y_true, y_pred),
        'spearman': spearman_correlation(y_true, y_pred)
    }
