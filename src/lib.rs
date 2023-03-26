#![allow(clippy::type_complexity, clippy::too_many_arguments)]
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgb};
use imageproc::drawing::draw_polygon_mut;
// use imageproc::drawing::{draw_polygon_mut, Polygon};
use std::path::Path;

fn read_image(file_path: &str) -> DynamicImage {
    image::open(file_path).unwrap()
}

fn gradient_magnitude_and_direction(
    image: &DynamicImage,
) -> (
    ImageBuffer<Luma<f32>, Vec<f32>>,
    ImageBuffer<Luma<f32>, Vec<f32>>,
) {
    let (width, height) = image.dimensions();

    let mut gradient_magnitude = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height);
    let mut gradient_direction = ImageBuffer::<Luma<f32>, Vec<f32>>::new(width, height);

    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

    let sobel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut gx = 0.0;
            let mut gy = 0.0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = image
                        .get_pixel(x + kx as u32 - 1, y + ky as u32 - 1)
                        .to_luma()[0] as f32;

                    gx += pixel * sobel_x[ky][kx];
                    gy += pixel * sobel_y[ky][kx];
                }
            }

            let magnitude = (gx * gx + gy * gy).sqrt();
            let direction = gy.atan2(gx);

            gradient_magnitude.put_pixel(x, y, Luma([magnitude]));
            gradient_direction.put_pixel(x, y, Luma([direction]));
        }
    }

    (gradient_magnitude, gradient_direction)
}

fn compute_gvf_field(
    gradient_magnitude: &ImageBuffer<Luma<f32>, Vec<f32>>,
    gradient_direction: &ImageBuffer<Luma<f32>, Vec<f32>>,
    mu: f32,
    iterations: usize,
) -> (
    ImageBuffer<Luma<f32>, Vec<f32>>,
    ImageBuffer<Luma<f32>, Vec<f32>>,
) {
    let (width, height) = gradient_magnitude.dimensions();

    let mut gvf_x = (*gradient_direction).clone();
    let mut gvf_y = (*gradient_direction).clone();

    for _ in 0..iterations {
        let mut updated_gvf_x = gvf_x.clone();
        let mut updated_gvf_y = gvf_y.clone();

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let u = gvf_x.get_pixel(x, y)[0];
                let v = gvf_y.get_pixel(x, y)[0];

                let grad_u = (gvf_x.get_pixel(x + 1, y)[0] - 2.0 * u
                    + gvf_x.get_pixel(x - 1, y)[0]
                    + gvf_x.get_pixel(x, y + 1)[0]
                    - 2.0 * u
                    + gvf_x.get_pixel(x, y - 1)[0])
                    * mu;

                let grad_v = (gvf_y.get_pixel(x + 1, y)[0] - 2.0 * v
                    + gvf_y.get_pixel(x - 1, y)[0]
                    + gvf_y.get_pixel(x, y + 1)[0]
                    - 2.0 * v
                    + gvf_y.get_pixel(x, y - 1)[0])
                    * mu;

                let s = gradient_magnitude.get_pixel(x, y)[0];

                let new_u = u + grad_u - s * (u - gradient_direction.get_pixel(x, y)[0]);
                let new_v = v + grad_v - s * (v - gradient_direction.get_pixel(x, y)[0]);

                updated_gvf_x.put_pixel(x, y, Luma([new_u]));
                updated_gvf_y.put_pixel(x, y, Luma([new_v]));
            }
        }

        gvf_x = updated_gvf_x;
        gvf_y = updated_gvf_y;
    }

    (gvf_x, gvf_y)
}

use std::f32::consts::PI;

type Point = imageproc::point::Point<f32>;
type Snake = Vec<Point>;

fn initialize_snake(n_points: usize, center: Point, radius: f32) -> Snake {
    let Point {
        x: center_x,
        y: center_y,
    } = center;

    let mut snake = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let angle = 2.0 * PI * (i as f32) / (n_points as f32);
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        snake.push(Point::new(x, y));
    }

    snake
}

fn interpolate_gvf(gvf: &ImageBuffer<Luma<f32>, Vec<f32>>, x: f32, y: f32) -> f32 {
    let width = gvf.width();
    let height = gvf.height();

    let x_floor = x.floor() as u32;
    let y_floor = y.floor() as u32;

    if x_floor + 1 >= width || y_floor + 1 >= height {
        return 0.0;
    }

    let x_frac = x - x_floor as f32;
    let y_frac = y - y_floor as f32;

    let g00 = gvf.get_pixel(x_floor, y_floor)[0];
    let g01 = gvf.get_pixel(x_floor, y_floor + 1)[0];
    let g10 = gvf.get_pixel(x_floor + 1, y_floor)[0];
    let g11 = gvf.get_pixel(x_floor + 1, y_floor + 1)[0];

    let gx = (1.0 - x_frac) * g00 + x_frac * g10;
    let gy = (1.0 - x_frac) * g01 + x_frac * g11;

    (1.0 - y_frac) * gx + y_frac * gy
}

fn update_snake(
    snake: &mut Snake,
    gvf_x: &ImageBuffer<Luma<f32>, Vec<f32>>,
    gvf_y: &ImageBuffer<Luma<f32>, Vec<f32>>,
    alpha: f32,
    beta: f32,
    gamma: f32,
    iterations: usize,
) {
    let n_points = snake.len();
    let mut new_snake = snake.clone();

    for _ in 0..iterations {
        for i in 0..n_points {
            let Point { x, y } = snake[i];

            let force_x = interpolate_gvf(gvf_x, x, y);
            let force_y = interpolate_gvf(gvf_y, x, y);

            let prev_point = snake[(i + n_points - 1) % n_points];
            let next_point = snake[(i + 1) % n_points];

            let internal_force_x = alpha * (prev_point.x - 2.0 * x + next_point.x)
                + beta * (prev_point.x - 4.0 * x + next_point.x);
            let internal_force_y = alpha * (prev_point.y - 2.0 * y + next_point.y)
                + beta * (prev_point.y - 4.0 * y + next_point.y);

            let new_x = x + gamma * (force_x + internal_force_x);
            let new_y = y + gamma * (force_y + internal_force_y);

            new_snake[i] = Point::new(new_x, new_y);
        }

        *snake = new_snake.clone();
    }
}

fn draw_snake_on_image(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, snake: &Snake) {
    let mut points = Vec::with_capacity(snake.len());
    for &point in snake {
        points.push(imageproc::point::Point::new(
            point.x.round() as i32,
            point.y.round() as i32,
        ));
    }

    let color = Rgb([255, 0, 0]);

    // Draw the snake contour
    draw_polygon_mut(image, &points, color);

    // Connect the last point to the first point to close the contour
    if let (Some(&first_point), Some(&last_point)) = (points.first(), points.last()) {
        imageproc::drawing::draw_line_segment_mut(
            image,
            (first_point.x as f32, first_point.y as f32),
            (last_point.x as f32, last_point.y as f32),
            color,
        );
    }
}

/// Runs the Gradient Vector Flow (GVF) Snakes algorithm on an input image.
///
/// This function performs the following steps:
/// 1. Reads the input image.
/// 2. Computes the gradient magnitudes and directions.
/// 3. Computes the GVF field.
/// 4. Initializes the snake.
/// 5. Updates the snake based on the GVF field.
/// 6. Draws the updated snake on the image.
/// 7. Saves the output image to the specified file path.
///
/// # Arguments
///
/// * `input_image_path` - The file path of the input image.
/// * `output_image_path` - The file path where the output image with the snake contour should be saved.
/// * `snake_points` - The number of points in the initial snake.
/// * `center` - A tuple (x, y) representing the center of the initial snake circle.
/// * `radius` - The radius of the initial snake circle.
/// * `mu` - The regularization parameter for the GVF field computation.
/// * `gvf_iterations` - The number of iterations for the GVF field computation.
/// * `alpha` - The internal energy coefficient controlling the snake's stretching.
/// * `beta` - The internal energy coefficient controlling the snake's bending.
/// * `gamma` - The external energy coefficient controlling the snake's movement towards the GVF field.
/// * `snake_iterations` - The number of iterations for updating the snake.
///
/// # Example
///
/// ```
/// run_gvf_snakes(
///     "input_image.jpg",
///     "output_image.jpg",
///     50,
///     (100.0, 100.0),
///     30.0,
///     0.2,
///     100,
///     0.1,
///     1.0,
///     1.5,
///     100,
/// );
/// ```
///
/// This example runs the GVF Snakes algorithm on an input image "input_image.jpg" and saves the output image with the snake contour as "output_image.jpg". The initial snake is a circle with 50 points, centered at (100, 100), and with a radius of 30. The GVF field computation uses a regularization parameter of 0.2 and 100 iterations. The snake update uses internal energy coefficients of 0.1 (alpha) and 1.0 (beta), an external energy coefficient of 1.5 (gamma), and 100 iterations.
pub fn run_gvf_snakes(
    input_image_path: &str,
    output_image_path: &str,
    snake_points: usize,
    center: Point,
    radius: f32,
    mu: f32,
    gvf_iterations: usize,
    alpha: f32,
    beta: f32,
    gamma: f32,
    snake_iterations: usize,
) {
    let input_image = read_image(input_image_path);
    let (gradient_magnitude, gradient_direction) = gradient_magnitude_and_direction(&input_image);
    let (gvf_x, gvf_y) =
        compute_gvf_field(&gradient_magnitude, &gradient_direction, mu, gvf_iterations);
    let mut snake = initialize_snake(snake_points, center, radius);
    update_snake(
        &mut snake,
        &gvf_x,
        &gvf_y,
        alpha,
        beta,
        gamma,
        snake_iterations,
    );

    let mut output_image = input_image.to_rgb8();
    draw_snake_on_image(&mut output_image, &snake);

    output_image.save(Path::new(output_image_path)).unwrap();
}

// snake_points: 30-50 points, depending on the desired smoothness and detail level of the snake.
// center: The center of the initial circle should be close to the object of interest in the image.
// radius: The initial radius should be large enough to cover the object of interest, but not too large that it spans over the background or other unwanted regions.
// mu: A common value for the regularization parameter is 0.2. You may need to fine-tune this value based on the noise level in the image and the desired smoothness of the GVF field.
// gvf_iterations: 50-200 iterations. A higher number of iterations may result in a more accurate GVF field, but at the cost of increased computation time.
// alpha: The internal energy coefficient that controls the snake's stretching. Typical values range from 0.01 to 1. A higher value results in a more evenly spaced snake.
// beta: The internal energy coefficient that controls the snake's bending. Typical values range from 0.1 to 10. A higher value results in a smoother snake.
// gamma: The external energy coefficient that controls the snake's movement towards the GVF field. Typical values range from 0.5 to 2. A higher value results in a more aggressive movement of the snake towards the GVF field.
// snake_iterations: 50-200 iterations. A higher number of iterations may result in a more accurate snake contour, but at the cost of increased computation time.
