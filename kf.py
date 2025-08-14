from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseKalmanFilter(ABC):
    def __init__(self):
        self.A = None
        self.B = None
        self.u = None
        self.H = None
        self.d = None
        self.Q = None
        self.R = None
        self.x = None
        self.P = None

    def set_matrices(
        self: "BaseKalmanFilter",
        A: np.ndarray,
        B: np.ndarray,
        u: np.ndarray,
        H: np.ndarray,
        d: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        self.A = A
        self.B = B
        self.u = u
        self.H = H
        self.d = d
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        pass

    def _predict(self: "BaseKalmanFilter"):
        if self.B is not None:
            if self.u is None:
                self.u = np.zeros((self.B.shape[1], 1))
            else:
                self.u = np.atleast_2d(self.u).reshape(-1, 1)
            self.x = self.A @ self.x + self.B @ self.u
        else:
            self.x = self.A @ self.x

        self.P = self.A @ self.P @ self.A.T + self.Q

    def _update(self: "BaseKalmanFilter", z):
        if self.d is None:
            self.d = np.zeros(z.shape)
        y = z - (self.H @ self.x + self.d)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        Id = np.eye(self.P.shape[0])
        self.P = (Id - K @ self.H) @ self.P


class GibsonSchwartz_KF(BaseKalmanFilter):

    def __Kalman_NegLL(self, params, z, dt, x_0, P_0, maturities, R, r=0.04):
        x_hat = x_0.reshape(-1, 1)
        P = P_0.copy()
        log_likelihood = 0.0
        num_f, num_obs = z.shape

        for t in range(num_obs):
            A, Q, B, c = self.__build_state_matrices(params, dt)
            H, d = self.__build_H_and_d(params, maturities, r)

            x_pred = A @ x_hat + B @ c
            P_pred = A @ P @ A.T + Q
            z_t = z[:, t].reshape(-1, 1)
            y = z_t - (H @ x_pred + d)
            S = H @ P_pred @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return 1e3

            K = P_pred @ H.T @ S_inv
            x_hat = x_pred + K @ y
            P = (np.eye(P.shape[0]) - K @ H) @ P_pred
            det_S = max(np.linalg.det(S), 1e-5)
            log_likelihood -= 0.5 * (
                num_f * np.log(2 * np.pi) + np.log(det_S) + y.T @ S_inv @ y
            )

        return float(log_likelihood)

    def Fit_Run(
        self,
        z_df,
        dt,
        x_0,
        P_0,
        maturities,
        R,
        bounds,
        n_trials=1000,
        r=0.04,
        verbose=True,
        manual_params=None,
    ):
        """
        Through manual_params, specific parameters
        can be provided to the filter in the order
        [k, alpha, mu, lambda, Sigma_s, Sigma_c, rho].
        Otherwise, the method will perform a grid search.
        The default mode is manual_params=None
        """

        if manual_params is not None:
            best_params = np.array(manual_params, dtype=float)
            best_loglike = None
            if verbose:
                print("\n Manually set parameters")
                param_names = ["k", "a", "mu", "l", "s_s", "s_c", "rho"]
                for name, value in zip(param_names, best_params):
                    print(f"{name}: {value:.6f}")
        else:
            z = z_df.to_numpy().T
            best_loglike = np.inf
            best_params = None
            bounds_array = np.array(bounds)
            lower_bounds = bounds_array[:, 0]
            upper_bounds = bounds_array[:, 1]

            for i in range(n_trials):
                trial_params = np.random.uniform(lower_bounds, upper_bounds)
                try:
                    loglike = self.__Kalman_NegLL(
                        trial_params, z, dt, x_0, P_0, maturities, R, r
                    )
                except Exception:
                    continue

                if loglike < best_loglike:
                    best_loglike = loglike
                    best_params = trial_params.copy()
            if verbose:
                print("\nBest Parameters (Random Grid Search):")
                param_names = ["k", "a", "mu", "l", "s_s", "s_c", "rho"]
                for name, value in zip(param_names, best_params):
                    print(f"{name}: {value:.6f}")
                print(f"Best log-likelihood: {best_loglike:.6f}")

        self.fitted_params = best_params
        x_hat = x_0.reshape(-1, 1)
        P = P_0.copy()
        x_hat_series = []
        A, Q, B, c = self.__build_state_matrices(best_params, dt)

        for i in range(z_df.shape[0]):
            H, d = self.__build_H_and_d(best_params, maturities, r)
            self.set_matrices(A, B, c, H, d, Q, R, x_hat, P)
            z_t = z_df.iloc[i].to_numpy().reshape(-1, 1)
            self._predict()
            self._update(z_t)
            x_hat = self.x
            P = self.P
            x_hat_series.append(x_hat.flatten())

        return pd.DataFrame(x_hat_series, index=z_df.index, columns=["S_t", "C_t"])

    def __build_state_matrices(self, params, dt):
        k, a, mu, _, s_s, s_c, rho = params
        B = np.eye(2)
        A = np.array([[1, -dt], [0, 1 - k * dt]])
        c = np.array([(mu - 0.5 * s_s**2) * dt, k * a * dt]).reshape(-1, 1)
        Q = np.array(
            [
                [(s_s**2) * dt, s_s * s_c * rho * dt],
                [s_s * s_c * rho * dt, (s_c**2) * dt],
            ]
        )
        return A, Q, B, c

    def __build_H_and_d(self, params, maturities, r):
        k, a, _, l, s_s, s_c, rho = params
        num_f = len(maturities)
        H = np.zeros((num_f, 2))
        d = np.zeros((num_f, 1))

        a1 = r - (a - l / k) + (s_c**2) / (2 * k**2) - s_s * s_c * rho / k
        a3 = (a - l / k) * k + s_s * s_c * rho - (s_c**2) / k

        def A_fun(Ti):
            return (
                a1 * Ti
                + ((s_c**2) / (4 * k**3)) * (1 - np.exp(-2 * k * Ti))
                + a3 * (1 - np.exp(-k * Ti)) / (k**2)
            )

        for i in range(num_f):
            H[i, :] = [1, -(1 - np.exp(-k * maturities[i])) / k]
            d[i, 0] = A_fun(maturities[i])

        return H, d
