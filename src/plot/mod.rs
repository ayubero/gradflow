use std::cell::RefCell;
use std::rc::Rc;

use ndarray::Array2;
use plotters::prelude::*;
use plotters::style::RGBColor;

use crate::nn::Module;
use crate::tensor::{self, Tensor};

const RED: RGBColor = RGBColor(231, 0, 11);
const BLUE: RGBColor = RGBColor(21, 93, 252);

fn compute_plot_ranges(points: &Vec<(f64, f64, u8)>) -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
    let xs: Vec<f64> = points.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<f64> = points.iter().map(|(_, y, _)| *y).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    (xs, ys, x_min, x_max, y_min, y_max)
}

pub fn plot_scatter(points: &Vec<(f64, f64, u8)>, filename: &'static str) -> Result<(), Box<dyn std::error::Error>> {
    // Compute plot ranges (padding included)
    let (_xs, _ys, x_min, x_max, y_min, y_max) = compute_plot_ranges(&points);
    let pad_x = (x_max - x_min) * 0.12 + 0.1;
    let pad_y = (y_max - y_min) * 0.12 + 0.1;

    // Draw scatter plot
    
    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("2-class dataset (2 features)", ("Ubuntu", 20).into_font())
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - pad_x)..(x_max + pad_x),
            (y_min - pad_y)..(y_max + pad_y),
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .y_label_formatter(&|y| format!("{:.1}", y))
        .x_label_style(("Ubuntu", 15).into_font())
        .y_label_style(("Ubuntu", 15).into_font())
        .axis_desc_style(("Ubuntu", 20).into_font())
        .light_line_style(ShapeStyle::from(&WHITE).stroke_width(0)) // Hide secondary lines
        .draw()?;

    // Draw points: class 0 blue circles, class 1 red triangles
    let class0 = points.iter().filter(|(_, _, l)| *l == 0).map(|(x, y, _)| (*x, *y));
    let class1 = points.iter().filter(|(_, _, l)| *l == 1).map(|(x, y, _)| (*x, *y));

    chart
        .draw_series(
            class0.map(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&BLUE).filled())),
        )?
        .label("Class 0")
        .legend(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&BLUE).filled()));

    chart
        .draw_series(
            class1.map(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&RED).filled())),
        )?
        .label("Class 1")
        .legend(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&RED).filled()));

    // Add legend
    chart
        .draw_series([
            Rectangle::new([(x_max - pad_x*0.9, y_max - pad_y*0.9), (x_max - pad_x*0.6, y_max - pad_y*0.8)], &WHITE.mix(0.0)) // invisible box to anchor
        ])?;
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("Ubuntu", 15).into_font())
        .draw()?;

    println!("Wrote scatter plot to {}", filename);

    Ok(())
}

pub fn plot_decision_regions(points: &Vec<(f64, f64, u8)>, filename: &'static str, model: &dyn Module) -> Result<(), Box<dyn std::error::Error>> {
    // Compute plot ranges (padding included)
    let (_xs, _ys, x_min, x_max, y_min, y_max) = compute_plot_ranges(&points);
    let pad_x = (x_max - x_min) * 0.12 + 0.1;
    let pad_y = (y_max - y_min) * 0.12 + 0.1;

    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("2-class dataset (2 features)", ("Ubuntu", 20).into_font())
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - pad_x)..(x_max + pad_x),
            (y_min - pad_y)..(y_max + pad_y),
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .y_label_formatter(&|y| format!("{:.1}", y))
        .x_label_style(("Ubuntu", 15).into_font())
        .y_label_style(("Ubuntu", 15).into_font())
        .axis_desc_style(("Ubuntu", 20).into_font())
        .light_line_style(ShapeStyle::from(&WHITE).stroke_width(0)) // Hide secondary lines
        .draw()?;

    // Grid to plot decision regions
    let resolution = 200; // Grid resolution (higher = smoother, but slower)
    let x_step = (x_max + pad_x - (x_min - pad_x)) / resolution as f64;
    let y_step = (y_max + pad_y - (y_min - pad_y)) / resolution as f64;

    let model_rc = Rc::new(RefCell::new(model));

    chart.draw_series(
        (0..resolution).flat_map(|i| {
            let model_rc = model_rc.clone(); // Clone Rc for each closure
            (0..resolution).map(move |j| {
                let x = x_min - pad_x + i as f64 * x_step;
                let y = y_min - pad_y + j as f64 * y_step;
                /*let x_squared = x.powi(2);
                let y_squared = y.powi(2);
                let x_by_y = x*y;*/

                // Predict
                /*let tensor_data = Array2::from_shape_vec(
                    (1, 5), vec![x, y, x_squared, y_squared, x_by_y]
                ).unwrap();*/
                let tensor_data = Array2::from_shape_vec(
                    (1, 2), vec![x, y]
                ).unwrap();
                println!("Input shape: {:?}", tensor_data.shape());
                let input = Rc::new(RefCell::new(
                    Tensor::new(tensor_data.into_dyn(), false)
                ));
                let prediction = model_rc.borrow().forward(input.clone());
                let probability_value = prediction.borrow().data[[0, 0]];
                let class = if probability_value >= 0.5 { 1 } else { 0 };

                let color = if class == 0 {
                    BLUE.mix(0.2) // semi-transparent blue
                } else {
                    RED.mix(0.2)  // semi-transparent red
                };

                // Draw small rectangle cell
                Rectangle::new(
                    [(x, y), (x + x_step, y + y_step)],
                    color.filled(),
                )
            })
        })
    )?;

    // Draw points: class 0 blue circles, class 1 red triangles
    let class0 = points.iter().filter(|(_, _, l)| *l == 0).map(|(x, y, _)| (*x, *y));
    let class1 = points.iter().filter(|(_, _, l)| *l == 1).map(|(x, y, _)| (*x, *y));

    chart
        .draw_series(
            class0.map(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&BLUE).filled())),
        )?
        .label("Class 0")
        .legend(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&BLUE).filled()));

    chart
        .draw_series(
            class1.map(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&RED).filled())),
        )?
        .label("Class 1")
        .legend(|(x, y)| Circle::new((x, y), 3, ShapeStyle::from(&RED).filled()));

    // Add legend
    chart
        .draw_series([
            Rectangle::new([(x_max - pad_x*0.9, y_max - pad_y*0.9), (x_max - pad_x*0.6, y_max - pad_y*0.8)], &WHITE.mix(0.0)) // invisible box to anchor
        ])?;
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("Ubuntu", 15).into_font())
        .draw()?;

    println!("Wrote scatter plot to {}", filename);

    Ok(())
}