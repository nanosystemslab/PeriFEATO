"""
Theta Band Thickness Parameterization
======================================

Represents shell thickness as piecewise-constant values in theta (latitude) bands.
Includes both protected shell thickness and fracture zone thickness factor per band.

Theta angle convention:
- theta = 0° at poles (|z| ≈ c_core, top/bottom)
- theta = 90° at equator (z ≈ 0, middle)

For a 3-band parameterization with edges [0, 30, 60, 90]:
- Band 0 (pole region): 0° ≤ theta < 30°
- Band 1 (mid-side):    30° ≤ theta < 60°
- Band 2 (equator):     60° ≤ theta ≤ 90°

Design variables per band:
- thickness_mm: Protected shell thickness
- fracture_factor: Fracture zone thickness = factor × protected thickness
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


class ThetaBandParams:
    """
    Thickness parameterization using theta bands.

    Attributes:
        theta_edges_deg: Band edge angles in degrees [0, ..., 90]
        thickness_mm: Protected shell thickness for each band (mm)
        fracture_factor: Fracture zone thickness factor per band (0-1)
        min_thickness_mm: Lower bound for protected thickness
        max_thickness_mm: Upper bound for protected thickness
        min_fracture_factor: Lower bound for fracture factor
        max_fracture_factor: Upper bound for fracture factor
    """

    def __init__(
        self,
        theta_edges_deg: List[float],
        thickness_mm: List[float],
        min_thickness_mm: float = 0.5,
        max_thickness_mm: float = 10.0,
        fracture_factor: List[float] = None,
        min_fracture_factor: float = 0.1,
        max_fracture_factor: float = 0.9,
        use_spline: bool = False,
    ):
        """
        Initialize theta band thickness parameters.

        Args:
            theta_edges_deg: Band edges in degrees, must start at 0 and end at 90
            thickness_mm: Protected thickness for each band (length = len(edges) - 1)
            min_thickness_mm: Minimum allowed protected thickness
            max_thickness_mm: Maximum allowed protected thickness
            fracture_factor: Fracture zone factor per band (default: 0.6 for all)
            min_fracture_factor: Minimum fracture factor (default: 0.1)
            max_fracture_factor: Maximum fracture factor (default: 0.9)
            use_spline: If True, treat per-band values as B-spline control points
                for C²-smooth t(θ) evaluation instead of piecewise-constant bands

        Example:
            >>> params = ThetaBandParams(
            ...     theta_edges_deg=[0, 30, 60, 90],
            ...     thickness_mm=[0.8, 1.0, 1.2],  # 3 bands
            ...     fracture_factor=[0.5, 0.6, 0.7]  # 3 bands
            ... )
        """
        self.theta_edges_deg = np.array(theta_edges_deg, dtype=float)
        self.thickness_mm = np.array(thickness_mm, dtype=float)
        self.min_thickness_mm = float(min_thickness_mm)
        self.max_thickness_mm = float(max_thickness_mm)

        # Fracture factor per band (default 0.6 if not specified)
        n_bands = len(self.theta_edges_deg) - 1
        if fracture_factor is None:
            self.fracture_factor = np.full(n_bands, 0.6)
        else:
            self.fracture_factor = np.array(fracture_factor, dtype=float)
        self.min_fracture_factor = float(min_fracture_factor)
        self.max_fracture_factor = float(max_fracture_factor)
        self.use_spline = bool(use_spline)

        # Validate
        self._validate()

    def _validate(self):
        """Validate parameterization constraints."""
        # Check edges
        if len(self.theta_edges_deg) < 2:
            raise ValueError("theta_edges_deg must have at least 2 values")

        if self.theta_edges_deg[0] != 0.0:
            raise ValueError("theta_edges_deg must start at 0")

        if self.theta_edges_deg[-1] != 90.0:
            raise ValueError("theta_edges_deg must end at 90")

        if not np.all(np.diff(self.theta_edges_deg) > 0):
            raise ValueError("theta_edges_deg must be strictly increasing")

        # Check thickness array length
        n_bands = len(self.theta_edges_deg) - 1
        if len(self.thickness_mm) != n_bands:
            raise ValueError(
                f"thickness_mm length ({len(self.thickness_mm)}) must equal "
                f"number of bands ({n_bands})"
            )

        # Check thickness bounds
        if np.any(self.thickness_mm < self.min_thickness_mm):
            raise ValueError(
                f"All thickness values must be >= {self.min_thickness_mm} mm"
            )

        if np.any(self.thickness_mm > self.max_thickness_mm):
            raise ValueError(
                f"All thickness values must be <= {self.max_thickness_mm} mm"
            )

        # Check fracture factor array length
        if len(self.fracture_factor) != n_bands:
            raise ValueError(
                f"fracture_factor length ({len(self.fracture_factor)}) must equal "
                f"number of bands ({n_bands})"
            )

        # Check fracture factor bounds
        if np.any(self.fracture_factor < self.min_fracture_factor):
            raise ValueError(
                f"All fracture factors must be >= {self.min_fracture_factor}"
            )

        if np.any(self.fracture_factor > self.max_fracture_factor):
            raise ValueError(
                f"All fracture factors must be <= {self.max_fracture_factor}"
            )

    @property
    def n_bands(self) -> int:
        """Number of theta bands."""
        return len(self.thickness_mm)

    @property
    def n_params(self) -> int:
        """Number of optimization parameters (thickness + fracture factor per band)."""
        return self.n_bands * 2

    @property
    def fracture_thickness_mm(self) -> np.ndarray:
        """Computed fracture zone thickness per band (factor × protected thickness)."""
        return self.thickness_mm * self.fracture_factor

    @property
    def theta_midpoints_rad(self) -> np.ndarray:
        """Midpoint angle (radians) for each theta band."""
        edges_rad = np.deg2rad(self.theta_edges_deg)
        return 0.5 * (edges_rad[:-1] + edges_rad[1:])

    def compute_band_area_weights(
        self,
        a_core_mm: float = 13.25,
        c_core_mm: float = 34.0,
    ) -> np.ndarray:
        """Compute the surface area fraction of each theta band on the ellipsoid.

        The surface element on an ellipsoid of revolution is:
            dS = 2π · a·sin(θ) · √(a²cos²θ + c²sin²θ) dθ

        Bands near the pole (θ≈0) have tiny area (sin(θ)≈0).
        Bands near the equator (θ≈90°) have large area.

        Returns:
            Array of area fractions (sums to 1.0) for each band.
        """
        a = a_core_mm
        c = c_core_mm
        edges_rad = np.deg2rad(self.theta_edges_deg)
        areas = np.zeros(self.n_bands)

        for i in range(self.n_bands):
            # Numerical integration with 100 points per band
            theta = np.linspace(edges_rad[i], edges_rad[i + 1], 100)
            # Surface element integrand (without the 2π factor, cancels in ratio)
            integrand = a * np.sin(theta) * np.sqrt(
                (a * np.cos(theta))**2 + (c * np.sin(theta))**2
            )
            areas[i] = np.trapz(integrand, theta)

        total = areas.sum()
        if total > 0:
            return areas / total
        return np.ones(self.n_bands) / self.n_bands

    def _build_spline(self, values: np.ndarray):
        """Build a CubicSpline through band midpoints with clamped BC.

        Args:
            values: One value per band (e.g. thickness_mm or fracture thickness).

        Returns:
            Callable that maps theta (radians) → interpolated value.
            For n_bands < 2, returns a constant function.
        """
        if self.n_bands < 2:
            const = float(values[0])
            return lambda theta: np.full_like(np.asarray(theta, dtype=float), const)
        from scipy.interpolate import CubicSpline
        return CubicSpline(self.theta_midpoints_rad, values, bc_type='clamped')

    def evaluate_thickness_at(self, theta_rad: np.ndarray) -> np.ndarray:
        """Evaluate protected thickness at arbitrary theta angles.

        Args:
            theta_rad: Array of theta angles in radians.

        Returns:
            Thickness in mm at each angle. If use_spline=False, returns
            piecewise-constant band lookup. If use_spline=True, evaluates
            the cubic spline and clips to [min, max] bounds.
        """
        theta_rad = np.asarray(theta_rad, dtype=float)
        if not self.use_spline:
            edges_rad = np.deg2rad(self.theta_edges_deg)
            band_idx = np.searchsorted(edges_rad, theta_rad, side='right') - 1
            band_idx = np.clip(band_idx, 0, self.n_bands - 1)
            return self.thickness_mm[band_idx]
        spline = self._build_spline(self.thickness_mm)
        return np.clip(spline(theta_rad), self.min_thickness_mm, self.max_thickness_mm)

    def evaluate_fracture_thickness_at(self, theta_rad: np.ndarray) -> np.ndarray:
        """Evaluate fracture zone thickness at arbitrary theta angles.

        Args:
            theta_rad: Array of theta angles in radians.

        Returns:
            Fracture thickness in mm (= thickness × fracture_factor) at each angle.
        """
        theta_rad = np.asarray(theta_rad, dtype=float)
        frac_thick = self.fracture_thickness_mm  # per-band values
        if not self.use_spline:
            edges_rad = np.deg2rad(self.theta_edges_deg)
            band_idx = np.searchsorted(edges_rad, theta_rad, side='right') - 1
            band_idx = np.clip(band_idx, 0, self.n_bands - 1)
            return frac_thick[band_idx]
        spline = self._build_spline(frac_thick)
        min_frac = float(np.min(frac_thick))
        max_frac = float(np.max(frac_thick))
        return np.clip(spline(theta_rad), min_frac, max_frac)

    def get_thickness(self, band_idx: int) -> float:
        """Get thickness for a specific band."""
        return float(self.thickness_mm[band_idx])

    def set_thickness(self, band_idx: int, thickness_mm: float):
        """Set thickness for a specific band (with bounds enforcement)."""
        thickness_mm = float(thickness_mm)

        # Enforce bounds
        thickness_mm = np.clip(thickness_mm, self.min_thickness_mm, self.max_thickness_mm)

        self.thickness_mm[band_idx] = thickness_mm

    def update_thickness(self, delta_mm: np.ndarray):
        """
        Update protected thickness by adding delta (with bounds enforcement).

        Args:
            delta_mm: Change in thickness for each band (array of length n_bands)
        """
        if len(delta_mm) != self.n_bands:
            raise ValueError(
                f"delta_mm length ({len(delta_mm)}) must equal n_bands ({self.n_bands})"
            )

        # Apply update
        self.thickness_mm += delta_mm

        # Enforce bounds
        self.thickness_mm = np.clip(
            self.thickness_mm,
            self.min_thickness_mm,
            self.max_thickness_mm
        )

    def update_fracture_factor(self, delta_factor: np.ndarray, min_fracture_thickness_mm: float = 0.4):
        """
        Update fracture factor by adding delta (with bounds enforcement).

        Args:
            delta_factor: Change in fracture factor for each band (array of length n_bands)
            min_fracture_thickness_mm: Minimum fracture zone thickness (manufacturing constraint)
        """
        if len(delta_factor) != self.n_bands:
            raise ValueError(
                f"delta_factor length ({len(delta_factor)}) must equal n_bands ({self.n_bands})"
            )

        # Apply update
        self.fracture_factor += delta_factor

        # Enforce bounds
        self.fracture_factor = np.clip(
            self.fracture_factor,
            self.min_fracture_factor,
            self.max_fracture_factor
        )

        # Enforce minimum fracture thickness constraint
        self.enforce_min_fracture_thickness(min_fracture_thickness_mm)

    def enforce_min_fracture_thickness(self, min_fracture_thickness_mm: float = 0.4):
        """
        Ensure fracture_factor × thickness >= min_fracture_thickness for all bands.

        Adjusts fracture_factor upward if needed to meet the constraint.
        Call this after any thickness or factor change.

        Args:
            min_fracture_thickness_mm: Minimum fracture zone thickness
        """
        for i in range(self.n_bands):
            min_factor = min_fracture_thickness_mm / self.thickness_mm[i]
            if self.fracture_factor[i] < min_factor:
                self.fracture_factor[i] = min_factor

        # Re-apply max bound (in case min_factor exceeds max)
        self.fracture_factor = np.clip(
            self.fracture_factor,
            self.min_fracture_factor,
            self.max_fracture_factor
        )

    def enforce_max_fracture_thickness(self, max_fracture_thickness_mm: float):
        """
        Ensure fracture_factor × thickness <= max_fracture_thickness for all bands.

        Adjusts fracture_factor downward if needed. This prevents fracture zones
        from becoming unbreakable when the protected shell gets thick.

        Args:
            max_fracture_thickness_mm: Maximum fracture zone thickness
        """
        if max_fracture_thickness_mm <= 0:
            return
        for i in range(self.n_bands):
            frac_thick = self.fracture_factor[i] * self.thickness_mm[i]
            if frac_thick > max_fracture_thickness_mm:
                new_factor = max_fracture_thickness_mm / self.thickness_mm[i]
                self.fracture_factor[i] = new_factor

        # Re-apply bounds
        self.fracture_factor = np.clip(
            self.fracture_factor,
            self.min_fracture_factor,
            self.max_fracture_factor
        )

    def enforce_smoothness(self, max_gradient_mm: float = 2.0, lift_mode: bool = True):
        """
        Enforce smoothness constraint to prevent extreme thickness variations.

        Limits the thickness difference between adjacent bands to max_gradient_mm.
        This prevents bulging designs where one band is much thicker than neighbors,
        which can create stress concentrations and weak transition zones.

        Args:
            max_gradient_mm: Maximum allowed thickness difference between adjacent bands
            lift_mode: If True, LIFT thin bands to accommodate thick bands (bidirectional).
                       If False, only REDUCE thick bands (original behavior).

        With lift_mode=True:
            If pole needs to be thick (overstressed), thin bands are lifted to maintain
            gradient, allowing the whole structure to thicken as needed.
        """
        if max_gradient_mm <= 0:
            return  # Disabled

        # Iterate until no changes needed (max 10 passes to avoid infinite loop)
        for _ in range(10):
            changed = False

            if lift_mode:
                # BIDIRECTIONAL: Propagate from pole outward, lifting thin bands
                # This allows thick pole to "pull up" the rest of the structure
                for i in range(self.n_bands - 1):
                    if self.thickness_mm[i] - self.thickness_mm[i + 1] > max_gradient_mm:
                        # Band i+1 is too thin relative to band i
                        # LIFT band i+1 to satisfy gradient (instead of reducing band i)
                        min_required = self.thickness_mm[i] - max_gradient_mm
                        if min_required > self.thickness_mm[i + 1]:
                            self.thickness_mm[i + 1] = min_required
                            changed = True

                # Also propagate from equator toward pole (for equator-thick designs)
                for i in range(self.n_bands - 1, 0, -1):
                    if self.thickness_mm[i] - self.thickness_mm[i - 1] > max_gradient_mm:
                        # Band i-1 is too thin relative to band i
                        min_required = self.thickness_mm[i] - max_gradient_mm
                        if min_required > self.thickness_mm[i - 1]:
                            self.thickness_mm[i - 1] = min_required
                            changed = True
            else:
                # ORIGINAL: Only reduce thick bands (unidirectional)
                # Forward pass: ensure t[i+1] >= t[i] - max_gradient
                for i in range(self.n_bands - 1):
                    if self.thickness_mm[i] - self.thickness_mm[i + 1] > max_gradient_mm:
                        new_val = self.thickness_mm[i + 1] + max_gradient_mm
                        if new_val < self.thickness_mm[i]:
                            self.thickness_mm[i] = new_val
                            changed = True

                # Backward pass: ensure t[i] >= t[i+1] - max_gradient
                for i in range(self.n_bands - 1, 0, -1):
                    if self.thickness_mm[i] - self.thickness_mm[i - 1] > max_gradient_mm:
                        new_val = self.thickness_mm[i - 1] + max_gradient_mm
                        if new_val < self.thickness_mm[i]:
                            self.thickness_mm[i] = new_val
                            changed = True

            if not changed:
                break

        # Re-enforce bounds after smoothing
        self.thickness_mm = np.clip(
            self.thickness_mm,
            self.min_thickness_mm,
            self.max_thickness_mm
        )

    def get_theta_overrides(self) -> Dict:
        """
        Get theta mesh configuration overrides for mesh generation.

        Returns:
            Dictionary compatible with theta_mesh config overrides
        """
        return {
            "mesh": {
                "theta_band_edges_deg": self.theta_edges_deg.tolist(),
                "use_spline_thickness": self.use_spline,
            },
            "material": {
                "protected_thickness_theta_mm": self.thickness_mm.tolist(),
            },
            "fracture_design": {
                "fracture_factor_theta": self.fracture_factor.tolist(),
            },
        }

    def to_dict(self) -> Dict:
        """Export to dictionary for serialization."""
        return {
            "theta_edges_deg": self.theta_edges_deg.tolist(),
            "thickness_mm": self.thickness_mm.tolist(),
            "min_thickness_mm": self.min_thickness_mm,
            "max_thickness_mm": self.max_thickness_mm,
            "fracture_factor": self.fracture_factor.tolist(),
            "min_fracture_factor": self.min_fracture_factor,
            "max_fracture_factor": self.max_fracture_factor,
            "use_spline": self.use_spline,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> ThetaBandParams:
        """Load from dictionary."""
        return cls(
            theta_edges_deg=data["theta_edges_deg"],
            thickness_mm=data["thickness_mm"],
            min_thickness_mm=data.get("min_thickness_mm", 0.5),
            max_thickness_mm=data.get("max_thickness_mm", 10.0),
            fracture_factor=data.get("fracture_factor", None),
            min_fracture_factor=data.get("min_fracture_factor", 0.1),
            max_fracture_factor=data.get("max_fracture_factor", 0.9),
            use_spline=data.get("use_spline", False),
        )

    def copy(self) -> ThetaBandParams:
        """Create a deep copy."""
        return ThetaBandParams(
            theta_edges_deg=self.theta_edges_deg.copy(),
            thickness_mm=self.thickness_mm.copy(),
            min_thickness_mm=self.min_thickness_mm,
            max_thickness_mm=self.max_thickness_mm,
            fracture_factor=self.fracture_factor.copy(),
            min_fracture_factor=self.min_fracture_factor,
            max_fracture_factor=self.max_fracture_factor,
            use_spline=self.use_spline,
        )

    def __repr__(self) -> str:
        thick_str = ", ".join(f"{t:.2f}" for t in self.thickness_mm)
        frac_str = ", ".join(f"{f:.2f}" for f in self.fracture_factor)
        return f"ThetaBandParams({self.n_bands} bands: thick=[{thick_str}]mm, frac=[{frac_str}])"


def create_uniform_bands(
    n_bands: int = 3,
    initial_thickness_mm: float = 1.0,
    min_thickness_mm: float = 0.5,
    max_thickness_mm: float = 10.0,
    initial_fracture_factor: float = 0.6,
    min_fracture_factor: float = 0.1,
    max_fracture_factor: float = 0.9,
    use_spline: bool = False,
) -> ThetaBandParams:
    """
    Create uniform theta band parameterization.

    Divides [0°, 90°] into n_bands equal-sized bands with uniform initial values.

    Args:
        n_bands: Number of theta bands
        initial_thickness_mm: Initial protected thickness for all bands
        min_thickness_mm: Minimum allowed thickness
        max_thickness_mm: Maximum allowed thickness
        initial_fracture_factor: Initial fracture factor for all bands
        min_fracture_factor: Minimum allowed fracture factor
        max_fracture_factor: Maximum allowed fracture factor
        use_spline: If True, enable B-spline thickness interpolation

    Returns:
        ThetaBandParams with uniform thickness and fracture factor

    Example:
        >>> params = create_uniform_bands(n_bands=3, initial_thickness_mm=1.0)
        >>> print(params)
        ThetaBandParams(3 bands: thick=[1.00, 1.00, 1.00]mm, frac=[0.60, 0.60, 0.60])
    """
    theta_edges_deg = np.linspace(0, 90, n_bands + 1)
    thickness_mm = np.full(n_bands, initial_thickness_mm)
    fracture_factor = np.full(n_bands, initial_fracture_factor)

    return ThetaBandParams(
        theta_edges_deg=theta_edges_deg.tolist(),
        thickness_mm=thickness_mm.tolist(),
        min_thickness_mm=min_thickness_mm,
        max_thickness_mm=max_thickness_mm,
        fracture_factor=fracture_factor.tolist(),
        min_fracture_factor=min_fracture_factor,
        max_fracture_factor=max_fracture_factor,
        use_spline=use_spline,
    )
