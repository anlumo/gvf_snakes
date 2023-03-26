# GVF Snakes

A Rust crate implementing the Gradient Vector Flow (GVF) Snakes algorithm for image segmentation and object boundary detection.

**Work in progress!**

## Overview

The GVF Snakes algorithm is an active contour model that uses an external force field derived from the image gradient to attract the snake towards object boundaries. The algorithm also includes internal forces that maintain the snake's smoothness and regularity.

This crate provides a single function, `run_gvf_snakes`, which applies the GVF Snakes algorithm on an input image and saves the output image with the snake contour drawn on it.

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
gvf_snakes = "0.1.0"
```

## Usage

To use the GVF Snakes algorithm in your Rust project, import the run_gvf_snakes function and call it with the appropriate parameters:

```rust
use gvf_snakes::run_gvf_snakes;

fn main() {
    run_gvf_snakes(
        "input_image.jpg",
        "output_image.jpg",
        50,
        (100.0, 100.0),
        30.0,
        0.2,
        100,
        0.1,
        1.0,
        1.5,
        100,
    );
}
```

This example runs the GVF Snakes algorithm on an input image "input_image.jpg" and saves the output image with the snake contour as "output_image.jpg". The initial snake is a circle with 50 points, centered at (100, 100), and with a radius of 30. The GVF field computation uses a regularization parameter of 0.2 and 100 iterations. The snake update uses internal energy coefficients of 0.1 (alpha) and 1.0 (beta), an external energy coefficient of 1.5 (gamma), and 100 iterations.

## Parameters

The run_gvf_snakes function accepts several parameters that control the behavior of the GVF Snakes algorithm. Refer to the function documentation for a detailed explanation of each parameter and suggested starting values.

## License

This code was completely written by ChatGPT 4 and thus does not fall under copyright protection.

## Contributing

Contributions to improve the GVF Snakes crate are welcome. Please submit issues or pull requests on the GitHub repository.
