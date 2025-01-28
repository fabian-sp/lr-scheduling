import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize, Bounds
import copy
from typing import Optional


class RateFitter:
    """
    For fitting the unknown parameters in the convergence bound.

    NOTE: In .fit(), we need to specify
            * inputs (id, iteration counter, lr)
            * targets (loss value)
            * id_map

          The id_map should be adictionary that maps each id to a Schedule object.
          This is convenient if we want to fit across different schedules (e.g. wsd with different decays).

    functional form:

    params:
        D, G_1, G_2, A_1, A_2, B, C, M
    """

    def __init__(self, p0=None):

        self.schedule_map = None
        self._params = dict()
        self.p0 = p0  # dict
        return

    def fit(
        self,
        inputs: pd.DataFrame,
        targets: np.ndarray,
        id_map: dict,
        method: str = "least-squares",
        ub: Optional[float] = None,
        **kwargs,
    ):
        """Fit coefficients for rate"""

        # Handle args
        self._fit_inputs = copy.deepcopy(inputs)
        self._fit_targets = copy.deepcopy(targets)
        self.schedule_map = copy.deepcopy(id_map)

        # Set bounds
        ub_C = targets.min() if (ub is None) else ub
        self._p0, self._bounds = self._construct_bounds_and_starting_point(ub_C=ub_C)

        # Fit
        if method == "least-squares":
            params = self._fit_least_squares(inputs=inputs, targets=targets, **kwargs)
        elif method == "huber":
            params = self._fit_huber(inputs=inputs, targets=targets, **kwargs)
        else:
            raise KeyError(f"Unknown method for fitting {method}.")

        # Final steps
        self.set_params(params)

        return

    def predict(self, inputs: pd.DataFrame, id_map: Optional[dict] = None):

        if id_map is not None:
            self.schedule_map = copy.deepcopy(id_map)

        if self.params is None:
            raise KeyError("Cannot predict before params are fitted.")
        else:
            y = self._vec_eval(inputs, *self.params.values())
            return y

    def _bounds(self):
        lb = {
            "D": 0,
            "G_1": 0,
            "G_2": 0,
            "A_1": -np.inf,
            "A_2": -np.inf,
            "B": 0.0,
            "C": 0,
            "M": -np.inf,
        }
        ub = {
            "D": np.inf,
            "G_1": np.inf,
            "G_2": np.inf,
            "A_1": 0.0,
            "A_2": np.inf,
            "B": np.inf,
            "C": np.inf,
            "M": np.inf,
        }
        return lb, ub

    def _construct_bounds_and_starting_point(self, ub_C=None):
        """
        Construct parameter bounds.
        Also use best target value to upper bound C param.
        """
        if self.p0 is None:
            # D, G_1,, A_1, A_2, B, C, M
            p0 = np.array(
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5 * ub_C, 1.0]
            ) 
        else:
            vals = [*self.p0.values()]
            if self.p0["C"] is None:  # C can be None init based on ub_C
                vals[-2] = 0.5 * ub_C
            p0 = np.array(vals)
        lb, ub = self._bounds()
        ub["C"] = ub_C  # upper bound of constant is best target value

        bounds = Bounds(np.array(list(lb.values())), np.array(list(ub.values())))

        return p0, bounds

    def _vec_eval(self, inputs, *params):
        """forward map for batch of inputs to predict rate"""
        D, G_1, G_2, A_1, A_2, B, C, M = params  # params will be a tuple here

        N = len(inputs)
        y = np.zeros(N)

        for j in range(N):
            id, t, gamma = inputs.loc[j, "id"], inputs.loc[j, "t"], inputs.loc[j, "lr"]

            # Get correct schedule object
            S = self.schedule_map[id]
            S.set_base_lr(gamma)
            steps = np.arange(1, S._steps + 1) 
            grad_shape_1 = np.exp(A_1 * steps)
            grad_shape_2 = steps ** A_2
            grad_norms = G_1 * grad_shape_1 + G_2 * grad_shape_2 + B
            y[j] = C + M * (S.compute_rate(grad_norms=grad_norms, D=D, T=t))
        return y

    @property
    def params(self):
        return self._params

    def set_params(self, params):
        (
            self._params["D"],
            self._params["G_1"],
            self._params["G_2"],
            self._params["A_1"],
            self._params["A_2"],
            self._params["B"],
            self._params["C"],
            self._params["M"],
        ) = params

    def _fit_least_squares(self, inputs, targets, maxfev=None):

        res = curve_fit(
            f=self._vec_eval,
            xdata=inputs,
            ydata=targets,
            p0=self._p0,
            full_output=True,
            bounds=self._bounds,
            maxfev=maxfev,
            # method='trf',  # trust region reflective - more stable
            # loss='soft_l1',  # more robust to outliers
            # ftol=1e-6,  # tolerance for termination
        )

        self._std_params = np.sqrt(np.diag(res[1]))
        self._fit_infodict = res[2]

        # print(
        #     f"Residual: {self._fit_infodict['fvec'].min()} (min), {self._fit_infodict['fvec'].max()} (max), {np.abs(self._fit_infodict['fvec']).mean()} (MAD)"
        # )
        
        return res[0]

    def _fit_huber(
        self,
        inputs,
        targets,
        method="L-BFGS-B",
        max_iter=10_000,
        huber_mu=1e-3,
        use_log=True,
    ):
        if use_log:
            assert np.all(
                targets > 0
            ), "Found non-positive targets, not compatible with usage of log in Huber funciton"

        def objective(params):
            y = self._vec_eval(inputs, *params)  # TODO: are y by construction positive?
            if use_log:
                r = np.log(y) - np.log(targets)
            else:
                r = y - targets
            return huber_loss(r, huber_mu).mean()

        solver = minimize(
            objective,
            x0=self._p0,
            bounds=self._bounds,
            method=method,
            options={"disp": False, "maxiter": max_iter},
        )

        final_obj = solver.fun
        # print(f"Final objective function of Huber fit: {final_obj}")

        return solver.x


def huber_loss(r, mu):
    return np.where(np.abs(r) < mu, 0.5 * (r**2), mu * np.abs(r) - 0.5 * (mu**2))
