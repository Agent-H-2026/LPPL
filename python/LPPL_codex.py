from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class LPPLFitResult:
    success: bool
    tc: float
    m: float
    omega: float
    A: float
    B: float
    C1: float
    C2: float
    C: float
    phi: float
    rmse: float
    sse: float
    r2: float
    n_obs: int
    message: str


class LPPLModel:
    """
    Stable LPPL calibration using the Filimonov-Sornette reformulation:

        log p(t) = A + B*(tc-t)^m
                     + C1*(tc-t)^m*cos(omega*log(tc-t))
                     + C2*(tc-t)^m*sin(omega*log(tc-t))

    Nonlinear parameters: tc, m, omega
    Linear parameters:    A, B, C1, C2

    Notes:
    - Input y should usually be log-prices.
    - Time t is represented as integers 0, 1, 2, ..., N-1.
    - tc is searched beyond the sample end.
    """

    def __init__(
        self,
        m_bounds: Tuple[float, float] = (0.1, 0.9),
        omega_bounds: Tuple[float, float] = (6.0, 13.0),
        tc_offset_bounds: Tuple[float, float] = (1.0, 0.35),
        max_nfev: int = 300,
        random_state: Optional[int] = None,
    ) -> None:
        """
        tc_offset_bounds:
            (min_days_after_end, max_fraction_of_window_length_after_end)
            Example for a 120-point window:
                min tc offset = 1 day
                max tc offset = 0.35 * 120 = 42 days
        """
        self.m_bounds = m_bounds
        self.omega_bounds = omega_bounds
        self.tc_offset_bounds = tc_offset_bounds
        self.max_nfev = max_nfev
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _design_matrix(t: np.ndarray, tc: float, m: float, omega: float) -> np.ndarray:
        dt = tc - t
        if np.any(dt <= 0):
            raise ValueError("All tc - t values must be positive.")
        f = dt ** m
        g = f * np.cos(omega * np.log(dt))
        h = f * np.sin(omega * np.log(dt))
        X = np.column_stack([np.ones_like(t), f, g, h])
        return X

    @staticmethod
    def _solve_linear_params(
        t: np.ndarray, y: np.ndarray, tc: float, m: float, omega: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve linear parameters [A, B, C1, C2] by OLS.
        Returns:
            beta: shape (4,)
            y_hat: fitted values
        """
        X = LPPLModel._design_matrix(t, tc, m, omega)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        return beta, y_hat

    @staticmethod
    def _residuals_nonlinear(
        params: np.ndarray, t: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        tc, m, omega = params
        try:
            _, y_hat = LPPLModel._solve_linear_params(t, y, tc, m, omega)
            return y - y_hat
        except Exception:
            # Return large residuals for invalid parameter regions
            return np.full_like(y, 1e6, dtype=float)

    def _bounds_for_tc(self, n: int) -> Tuple[float, float]:
        t_end = n - 1
        min_offset = self.tc_offset_bounds[0]
        max_offset = max(self.tc_offset_bounds[0] + 1e-6,
                         self.tc_offset_bounds[1] * n)
        return t_end + min_offset, t_end + max_offset

    def _random_initial_guesses(self, n: int, n_starts: int) -> np.ndarray:
        tc_lo, tc_hi = self._bounds_for_tc(n)
        m_lo, m_hi = self.m_bounds
        w_lo, w_hi = self.omega_bounds

        tc0 = self.rng.uniform(tc_lo, tc_hi, size=n_starts)
        m0 = self.rng.uniform(m_lo, m_hi, size=n_starts)
        w0 = self.rng.uniform(w_lo, w_hi, size=n_starts)
        return np.column_stack([tc0, m0, w0])

    def fit(
        self,
        prices: np.ndarray | pd.Series,
        n_starts: int = 30,
        use_log: bool = True,
        enforce_filters: bool = True,
    ) -> LPPLFitResult:
        """
        Fit LPPL to one window of prices.

        Parameters
        ----------
        prices : array-like
            Positive price series.
        n_starts : int
            Number of multi-start local optimizations.
        use_log : bool
            If True, fit on log(prices). LPPL is usually specified on log-price.
        enforce_filters : bool
            Require practical LPPL filters such as B < 0 and |C| < 1.

        Returns
        -------
        LPPLFitResult
        """
        prices = np.asarray(prices, dtype=float)
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive.")

        y = np.log(prices) if use_log else prices.copy()
        n = len(y)
        t = np.arange(n, dtype=float)

        tc_lo, tc_hi = self._bounds_for_tc(n)
        lower = np.array([tc_lo, self.m_bounds[0], self.omega_bounds[0]], dtype=float)
        upper = np.array([tc_hi, self.m_bounds[1], self.omega_bounds[1]], dtype=float)

        best = None
        best_sse = np.inf

        for x0 in self._random_initial_guesses(n, n_starts):
            res = least_squares(
                self._residuals_nonlinear,
                x0=x0,
                bounds=(lower, upper),
                args=(t, y),
                method="trf",
                max_nfev=self.max_nfev,
            )

            tc, m, omega = res.x
            try:
                beta, y_hat = self._solve_linear_params(t, y, tc, m, omega)
            except Exception:
                continue

            A, B, C1, C2 = beta
            sse = float(np.sum((y - y_hat) ** 2))
            tss = float(np.sum((y - y.mean()) ** 2))
            r2 = np.nan if tss <= 0 else 1.0 - sse / tss

            C = float(np.sqrt(C1 ** 2 + C2 ** 2))
            phi = float(np.arctan2(C2, C1))
            rmse = float(np.sqrt(sse / n))

            passed = True
            if enforce_filters:
                passed = (
                    (B < 0.0) and
                    (abs(C) < 1.0) and
                    (self.m_bounds[0] <= m <= self.m_bounds[1]) and
                    (self.omega_bounds[0] <= omega <= self.omega_bounds[1])
                )

            if passed and sse < best_sse:
                best_sse = sse
                best = LPPLFitResult(
                    success=bool(res.success),
                    tc=float(tc),
                    m=float(m),
                    omega=float(omega),
                    A=float(A),
                    B=float(B),
                    C1=float(C1),
                    C2=float(C2),
                    C=float(C),
                    phi=float(phi),
                    rmse=rmse,
                    sse=sse,
                    r2=float(r2),
                    n_obs=n,
                    message=res.message,
                )

        if best is None:
            return LPPLFitResult(
                success=False,
                tc=np.nan,
                m=np.nan,
                omega=np.nan,
                A=np.nan,
                B=np.nan,
                C1=np.nan,
                C2=np.nan,
                C=np.nan,
                phi=np.nan,
                rmse=np.nan,
                sse=np.inf,
                r2=np.nan,
                n_obs=n,
                message="No fit passed the LPPL filters.",
            )

        return best

    @staticmethod
    def predict(
        t: np.ndarray,
        tc: float,
        m: float,
        omega: float,
        A: float,
        B: float,
        C1: float,
        C2: float,
    ) -> np.ndarray:
        X = LPPLModel._design_matrix(np.asarray(t, dtype=float), tc, m, omega)
        beta = np.array([A, B, C1, C2], dtype=float)
        return X @ beta

    def rolling_scan(
        self,
        series: pd.Series,
        window_size: int = 120,
        step: int = 5,
        n_starts: int = 25,
        use_log: bool = True,
        enforce_filters: bool = True,
    ) -> pd.DataFrame:
        """
        Run rolling LPPL fits across time for bubble diagnostics.

        Parameters
        ----------
        series : pd.Series
            Price series with DatetimeIndex preferred.
        window_size : int
            Number of observations in each fit window.
        step : int
            Move the window by this many observations.
        n_starts : int
            Multi-start count per window.

        Returns
        -------
        pd.DataFrame
            One row per fitted window.
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        out: List[Dict] = []
        n = len(series)

        for end in range(window_size, n + 1, step):
            window = series.iloc[end - window_size:end]
            fit = self.fit(
                window.values,
                n_starts=n_starts,
                use_log=use_log,
                enforce_filters=enforce_filters,
            )

            row = {
                "window_start": window.index[0],
                "window_end": window.index[-1],
                "success": fit.success,
                "tc_index_ahead": np.nan if np.isnan(fit.tc) else fit.tc - (window_size - 1),
                "tc": fit.tc,
                "m": fit.m,
                "omega": fit.omega,
                "A": fit.A,
                "B": fit.B,
                "C1": fit.C1,
                "C2": fit.C2,
                "C": fit.C,
                "phi": fit.phi,
                "rmse": fit.rmse,
                "sse": fit.sse,
                "r2": fit.r2,
                "n_obs": fit.n_obs,
                "message": fit.message,
            }

            # If DatetimeIndex is present, estimate calendar tc date by rounding ahead.
            if isinstance(window.index, pd.DatetimeIndex) and np.isfinite(row["tc_index_ahead"]):
                ahead = int(round(row["tc_index_ahead"]))
                if ahead >= 0:
                    tc_pos = min(end - 1 + ahead, len(series) - 1)
                    row["tc_date_proxy"] = series.index[tc_pos]
                else:
                    row["tc_date_proxy"] = pd.NaT
            else:
                row["tc_date_proxy"] = pd.NaT

            out.append(row)

        return pd.DataFrame(out)


def lppl_confidence_indicator(
    scan_df: pd.DataFrame,
    max_days_to_tc: float = 30.0,
    min_r2: float = 0.85,
) -> pd.DataFrame:
    """
    Create a simple hedge-fund-style LPPL bubble indicator from rolling fits.

    A window counts as a "qualified bubble window" if:
      - fit succeeded
      - tc is near (within max_days_to_tc observations ahead)
      - B < 0
      - R^2 above threshold
      - m and omega already passed in fit filters if enforce_filters=True

    Returns the same DataFrame plus binary and smoothed indicators.
    """
    df = scan_df.copy()

    df["qualified"] = (
        df["success"].fillna(False) &
        df["tc_index_ahead"].between(0, max_days_to_tc, inclusive="both") &
        (df["B"] < 0) &
        (df["r2"] >= min_r2)
    )

    # Rolling fraction of recent windows that are qualified
    df["confidence_5"] = df["qualified"].rolling(5, min_periods=1).mean()
    df["confidence_10"] = df["qualified"].rolling(10, min_periods=1).mean()

    return df