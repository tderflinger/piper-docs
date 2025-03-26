---
title:  transforms.py
description: transforms.py
---

## Code Explained

The provided code implements a piecewise rational quadratic spline transformation, which is a flexible and invertible mapping used in normalizing flows for density estimation and generative modeling. Below is an explanation of the key components and their functionality:

---

### **1. Constants**
The constants `DEFAULT_MIN_BIN_WIDTH`, `DEFAULT_MIN_BIN_HEIGHT`, and `DEFAULT_MIN_DERIVATIVE` define the minimum values for bin widths, bin heights, and derivatives, respectively. These constraints ensure numerical stability and prevent degenerate cases where bins collapse to zero width or height.

---

### **2. `piecewise_rational_quadratic_transform`**
This function serves as the main entry point for applying the piecewise rational quadratic spline transformation.

- **Inputs**:
  - `inputs`: The data to be transformed.
  - `unnormalized_widths`, `unnormalized_heights`, `unnormalized_derivatives`: Parameters defining the spline.
  - `inverse`: A boolean indicating whether to apply the forward or inverse transformation.
  - `tails`: Specifies the behavior outside the spline's domain (e.g., "linear").
  - `tail_bound`: The boundary for the spline's domain.
  - `min_bin_width`, `min_bin_height`, `min_derivative`: Minimum constraints for numerical stability.

- **Behavior**:
  - If `tails` is `None`, the function uses the `rational_quadratic_spline` for bounded inputs.
  - If `tails` are specified, it uses `unconstrained_rational_quadratic_spline` to handle inputs outside the domain.

- **Outputs**:
  - `outputs`: The transformed data.
  - `logabsdet`: The log absolute determinant of the Jacobian, used in normalizing flows for density computation.

This function abstracts the spline transformation, allowing for both bounded and unbounded input handling.

---

### **3. `searchsorted`**
This utility function determines the bin index for each input value by comparing it against bin boundaries.

- **Inputs**:
  - `bin_locations`: The cumulative bin boundaries.
  - `inputs`: The data to be binned.
  - `eps`: A small value added to the last bin boundary to ensure numerical stability.

- **Behavior**:
  - Compares each input value against the bin boundaries and returns the index of the bin it falls into.

This function is critical for identifying the appropriate bin for each input during the spline transformation.

---

### **4. `unconstrained_rational_quadratic_spline`**
This function extends the spline transformation to handle inputs outside a specified domain.

- **Inputs**:
  - Similar to `piecewise_rational_quadratic_transform`, with additional parameters for handling tails.

- **Behavior**:
  - For inputs within the domain (`inside_interval_mask`), it applies the `rational_quadratic_spline`.
  - For inputs outside the domain (`outside_interval_mask`), it applies a linear transformation if `tails="linear"`.
  - Pads the derivatives to ensure smooth transitions at the boundaries.

- **Outputs**:
  - `outputs`: The transformed data.
  - `logabsdet`: The log absolute determinant of the Jacobian.

This function ensures that the spline transformation can handle unbounded inputs gracefully.

---

### **5. `rational_quadratic_spline`**
This function implements the core piecewise rational quadratic spline transformation.

- **Inputs**:
  - `inputs`: The data to be transformed.
  - `unnormalized_widths`, `unnormalized_heights`, `unnormalized_derivatives`: Parameters defining the spline.
  - `inverse`: Whether to apply the forward or inverse transformation.
  - `left`, `right`, `bottom`, `top`: The boundaries of the spline's domain.
  - `min_bin_width`, `min_bin_height`, `min_derivative`: Minimum constraints for numerical stability.

- **Behavior**:
  - Computes bin widths, heights, and derivatives using softmax and softplus to ensure positivity.
  - Calculates cumulative widths and heights to define the spline's piecewise structure.
  - Identifies the bin for each input using `searchsorted`.
  - Applies the forward or inverse transformation based on the quadratic spline equations.

- **Outputs**:
  - `outputs`: The transformed data.
  - `logabsdet`: The log absolute determinant of the Jacobian.

This function is the backbone of the spline transformation, enabling flexible and invertible mappings.

---

### **Applications**
These functions collectively implement a powerful transformation used in normalizing flows. Applications include:
- **Density Estimation**: Modeling complex probability distributions.
- **Generative Modeling**: Synthesizing data by sampling from learned distributions.
- **Audio and Image Processing**: Transforming data in tasks like speech synthesis and image generation.

The modular design allows for flexibility in handling bounded and unbounded inputs, making it suitable for a wide range of machine learning tasks.

## Source Code

```py
import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    # bin_locations[..., -1] += eps
    bin_locations[..., bin_locations.size(-1) - 1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        # unnormalized_derivatives[..., -1] = constant
        unnormalized_derivatives[..., unnormalized_derivatives.size(-1) - 1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    # if torch.min(inputs) < left or torch.max(inputs) > right:
    #     raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    # if min_bin_width * num_bins > 1.0:
    #     raise ValueError("Minimal bin width too large for the number of bins")
    # if min_bin_height * num_bins > 1.0:
    #     raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    # cumwidths[..., -1] = right
    cumwidths[..., cumwidths.size(-1) - 1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    # cumheights[..., -1] = top
    cumheights[..., cumheights.size(-1) - 1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all(), discriminant

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet
```
