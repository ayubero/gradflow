use ndarray::{Array2, ArrayD};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::Distribution;
use statrs::distribution::MultivariateNormal;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;
use plotters::prelude::*;
use plotters::style::RGBColor;

use gradflow::modules;
use gradflow::nn::{Linear, Module, ReLU, Sequential, Sigmoid};
use gradflow::optimizer::SGD;
use gradflow::tensor::{bce_loss, Tensor};

const RED: RGBColor = RGBColor(231, 0, 11);
const BLUE: RGBColor = RGBColor(21, 93, 252);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let n_per_class = 200;
    let seed = 42u64; // reproducible
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Gaussian blob parameters: means and covariance matrices
    let mean0 = vec![-1.0f64, -0.5f64];
    let mean1 = vec![1.0f64, 0.5f64];

    // Covariance matrices as Vec<Vec<f64>>
    // Covariance matrices as flattened 1D vectors (row-major)
    let cov0 = vec![0.36, 0.1, 0.1, 0.36];
    let cov1 = vec![0.36, -0.1, -0.1, 0.36];

    let mvn0 = MultivariateNormal::new(mean0, cov0)?;
    let mvn1 = MultivariateNormal::new(mean1, cov1)?;

    // Storage
    let mut points: Vec<(f64, f64, u8)> = Vec::with_capacity(n_per_class * 2);

    // Generate class 0
    for _ in 0..n_per_class {
        let sample = mvn0.sample(&mut rng);
        points.push((sample[0], sample[1], 0));
    }

    // Generate class 1
    for _ in 0..n_per_class {
        let sample = mvn1.sample(&mut rng);
        points.push((sample[0], sample[1], 1));
    }

    // Optionally shuffle the dataset
    points.shuffle(&mut rng);

    // Write CSV
    let file = File::create("data/data.csv")?;
    let mut w = BufWriter::new(file);
    writeln!(w, "x,y,label")?;
    for (x, y, label) in &points {
        writeln!(w, "{:.6},{:.6},{}", x, y, label)?;
    }
    w.flush()?;
    println!("Wrote dataset to data.csv ({} samples).", points.len());

    // Compute plot ranges (padding included)
    let xs: Vec<f64> = points.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<f64> = points.iter().map(|(_, y, _)| *y).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let pad_x = (x_max - x_min) * 0.12 + 0.1;
    let pad_y = (y_max - y_min) * 0.12 + 0.1;

    // Draw scatter plot
    //let filename = "data/scatter.png";
    //let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    let filename = "data/scatter.svg";
    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("2-class logistic-like dataset (2 features)", ("Ubuntu", 20).into_font())
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - pad_x)..(x_max + pad_x),
            (y_min - pad_y)..(y_max + pad_y),
        )?;

    //chart.configure_mesh().draw()?;
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

    // Separate features and labels
    let n_samples = points.len();
    let mut x_data = Array2::<f64>::zeros((n_samples, 2)); // x, y
    let mut y_data = Array2::<f64>::zeros((n_samples, 1)); // label

    for (i, (x, y, label)) in points.iter().enumerate() {
        x_data[[i, 0]] = *x;
        x_data[[i, 1]] = *y;
        y_data[[i, 0]] = *label as f64;
    }

    // Training
    let model = Sequential::new(modules![
        Linear::new(2, 16),
        ReLU::new(),
        Linear::new(16, 8),
        ReLU::new(),
        Linear::new(8, 1),
        Sigmoid::new(),
    ]);
    let optimizer = SGD { params: model.parameters(), lr: 0.01 };

    for i in 0..100 {
        // Get data
        let x = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![2], 1.0), false)));
        let y_true = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![1], 2.0), false)));

        // Zero all the gradients
        optimizer.zero_grad();

        // Make predictions
        let y_pred = model.forward(x);

        // Compute loss and its gradients
        let loss = bce_loss(&y_pred, &y_true); // TO-DO: Fake loss
        loss.borrow_mut().backward();

        // Adjust learning weights
        optimizer.step();

        // Logging
        println!("Iteration {:?} | Loss: {:?}", i, loss.borrow().data);
    }

    Ok(())
}