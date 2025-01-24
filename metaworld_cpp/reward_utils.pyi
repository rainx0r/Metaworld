import numpy.typing as npt
import numpy as np
from enum import Enum

class SigmoidType(Enum):
    Gaussian = 0
    Hyperbolic = 1
    LongTail = 2
    Reciprocal = 3
    Cosine = 4
    Linear = 5
    Quadratic = 6
    TanhSquared = 7

def hamacher_product(a: float, b: float) -> float:
    """Returns the hamacher (t-norm) product of a and b.

    Computes (a * b) / ((a + b) - (a * b)).

    Args:
        a: 1st term of the hamacher product.
        b: 2nd term of the hamacher product.

    Returns:
        The hammacher product of a and b

    Raises:
        ValueError: a and b must range between 0 and 1
    """

def rect_prism_tolerance(
    curr: npt.NDArray[np.float64],
    zero: npt.NDArray[np.float64],
    one: npt.NDArray[np.float64],
) -> float:
    """Computes a reward if curr is inside a rectangular prism region.

    All inputs are 3D points with shape (3,).

    args:
        curr: the point that the prism reward region is being applied for.
        zero: the diagonal opposite corner of the prism with reward 0.
        one: the corner of the prism with reward 1.

    returns:
        a reward if curr is inside the prism, 1.0 otherwise.
    """

def tolerance(
    x: float | np.floating,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float | np.floating = 0.0,
    sigmoid: SigmoidType = SigmoidType.Gaussian,
) -> float:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: The input.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.

    Returns:
        A float with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
