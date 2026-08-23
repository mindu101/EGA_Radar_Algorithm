import numpy as np
from typing import List

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    class KalmanFilter:
        """filterpy 不可用时的最小线性 KF，实现同一组 predict/update 语义。"""

        def __init__(self, dim_x: int, dim_z: int):
            self.dim_x = int(dim_x)
            self.dim_z = int(dim_z)
            self.x = np.zeros(self.dim_x, dtype=float)
            self.F = np.eye(self.dim_x, dtype=float)
            self.H = np.zeros((self.dim_z, self.dim_x), dtype=float)
            self.P = np.eye(self.dim_x, dtype=float)
            self.Q = np.eye(self.dim_x, dtype=float)
            self.R = np.eye(self.dim_z, dtype=float)

        def predict(self):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x

        def update(self, measurement):
            z = np.asarray(measurement, dtype=float).reshape(self.dim_z)
            innovation = z - self.H @ self.x
            covariance = self.H @ self.P @ self.H.T + self.R
            gain = self.P @ self.H.T @ np.linalg.pinv(covariance)
            self.x = self.x + gain @ innovation
            identity = np.eye(self.dim_x, dtype=float)
            # Joseph 形式在 R=0 的精确 0x02BB 更新下更稳定。
            residual = identity - gain @ self.H
            self.P = residual @ self.P @ residual.T + gain @ self.R @ gain.T
            return self.x


class KalmanFilter3D:

    def __init__(self, initial_pos: List[float] = None, dt: float = 1.0):

        # (x, y, z, vx, vy, vz)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
        )
        self.reset(initial_pos)

    def predict(self) -> tuple[List[float], List[float]]:
        """
        Predict the next state of the Kalman filter
        Returns:
            A tuple containing the predicted position (x, y, z) and velocity (vx, vy, vz).
        """
        self.kf.predict()
        return self.kf.x[:3].tolist(), self.kf.x[3:].tolist()

    def update(self, pos: List[float]) -> tuple[List[float], List[float]]:
        """
        Update the Kalman filter with a new position measurement.
        Args:
            pos: A list containing the new position measurement [x, y, z].
        """
        self.kf.update(np.array(pos))
        return self.kf.x[:3].tolist(), self.kf.x[3:].tolist()

    def reset(self, initial_pos: List[float] = None) -> None:
        """
        Reset the Kalman filter with a new initial position.
        Args:
            initial_pos: A list containing the new initial position [x, y, z].
        """
        self.kf.x[:3] = np.array(initial_pos or [0.0, 0.0, 0.0]).reshape(3, -1)
        self.kf.x[3:] = np.array([0.0, 0.0, 0.0]).reshape(3, -1)
        self.kf.P *= 1.0
        self.kf.R *= 0.1
        self.kf.Q *= 0.01


class KalmanFilter2d:
    """
    2D Kalman Filter for position and velocity tracking.
    State vector: [x, y, vx, vy]
    """

    def __init__(self, initial_pos: List[float] = None, q_std:float = 2.0, r_std: float = 1.0, dt: float = 0.1):
        """
        Initialize 2D Kalman filter.
        Args:
            initial_pos: Initial position [x, y]
            dt: Time step
        """
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array(
            [
                [1, 0, dt, 0],   # x
                [0, 1, 0, dt],   # y  
                [0, 0, 1, 0],    # vx
                [0, 0, 0, 1],    # vy
            ]
        )
        
        # Measurement matrix (we observe x, y positions)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]
        )
        
        # Process noise covariance matrix
        # Higher values = more process noise, more responsive to changes
        self.dt = dt
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * q_std**2
        self.q_std = q_std
        self.r_std = r_std
        
        # Measurement noise covariance matrix
        # Higher values = less trust in measurements
        self.kf.R = np.eye(2) * r_std**2
        
        # Error covariance matrix
        self.kf.P = np.eye(4) * 100.0  # Initial uncertainty
        
        self.reset(initial_pos)

    def predict(self, dt = None) -> tuple[List[float], List[float]]:
        """
        Predict the next state of the Kalman filter.
        Returns:
            A tuple containing the predicted position (x, y) and velocity (vx, vy).
        """
        if dt == None:
            dt = self.dt
        self.dt = float(dt)
        self.kf.F = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * self.q_std**2
            
        self.kf.predict()
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def update(
        self, pos: List[float], r_std: float | None = None
    ) -> tuple[List[float], List[float]]:
        """
        Update the Kalman filter with a new position measurement.
        Args:
            pos: A list containing the new position measurement [x, y].
            r_std: Optional measurement noise std-dev for this update.
        Returns:
            A tuple containing the updated position (x, y) and velocity (vx, vy).
        """        

        if r_std is not None:
            prev_r = self.kf.R.copy()
            self.kf.R = np.eye(2) * float(r_std) ** 2
            self.kf.update(np.array(pos))
            self.kf.R = prev_r
            return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

        self.kf.update(np.array(pos))
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def reset(
        self,
        initial_pos: List[float] = None,
        initial_position_std_m: float | None = None,
        initial_velocity_std_mps: float | None = None,
    ) -> None:
        """
        Reset the Kalman filter with a new initial position.
        Args:
            initial_pos: A list containing the new initial position [x, y].
        """
        if initial_pos is None:
            initial_pos = [0.0, 0.0]
            
        # Initialize state vector
        self.kf.x = np.zeros(4)
        self.kf.x[:2] = np.array(initial_pos)
        self.kf.x[2:] = 0.0  # Initial velocities are zero
        
        # Reset error covariance matrix
        if initial_position_std_m is None or initial_velocity_std_mps is None:
            self.kf.P = np.eye(4) * 100.0
        else:
            self.kf.P = np.diag(
                [
                    float(initial_position_std_m) ** 2,
                    float(initial_position_std_m) ** 2,
                    float(initial_velocity_std_mps) ** 2,
                    float(initial_velocity_std_mps) ** 2,
                ]
            )

    def hard_reset(
        self,
        position: List[float],
        initial_position_std_m: float,
        initial_velocity_std_mps: float,
    ) -> None:
        """在可信新测量处重建位置，速度清零且交叉协方差为零。"""
        self.reset(
            position,
            initial_position_std_m=initial_position_std_m,
            initial_velocity_std_mps=initial_velocity_std_mps,
        )

    def get_state(self) -> tuple[List[float], List[float]]:
        """
        Get the current state of the filter.
        Returns:
            A tuple containing (position, velocity) where:
            - position: [x, y]
            - velocity: [vx, vy]
        """
        return self.kf.x[:2].tolist(), self.kf.x[2:].tolist()

    def get_position(self) -> List[float]:
        """
        Get the current position estimate.
        Returns:
            Current position [x, y]
        """
        return self.kf.x[:2].tolist()

    def get_velocity(self) -> List[float]:
        """
        Get the current velocity estimate.
        Returns:
            Current velocity [vx, vy]
        """
        return self.kf.x[2:].tolist()

    def set_process_noise(self, q_std: float) -> None:
        """
        Set the process noise covariance.
        Args:
            q_std: Standard deviation of process noise
        """
        dt = self.dt  # Assuming dt=1.0, could be made configurable
        self.kf.Q = np.array(
            [
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ]
        ) * q_std**2

    def set_measurement_noise(self, r_std: float) -> None:
        """
        Set the measurement noise covariance.
        Args:
            r_std: Standard deviation of measurement noise
        """
        self.r_std = r_std
        self.kf.R = np.eye(2) * r_std**2

    def predict_future(self, steps: int) -> List[List[float]]:
        """
        Predict future positions for multiple time steps ahead.
        Args:
            steps: Number of time steps to predict
        Returns:
            List of predicted positions [[x1, y1], [x2, y2], ...]
        """
        # Save current state
        current_x = self.kf.x.copy()
        current_P = self.kf.P.copy()
        
        predictions = []
        
        # Predict for each step
        for _ in range(steps):
            self.kf.predict()
            predictions.append(self.kf.x[:2].tolist())
        
        # Restore original state
        self.kf.x = current_x
        self.kf.P = current_P
        
        return predictions
