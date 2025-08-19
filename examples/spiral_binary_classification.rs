use gradflow::init::InitType;
use ndarray::Array2;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::Distribution;
use statrs::distribution::MultivariateNormal;
use statrs::statistics::Statistics;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::rc::Rc;
use plotters::prelude::*;
use plotters::style::RGBColor;

use gradflow::modules;
use gradflow::data::{generate_spiral, train_test_split, DataLoader};
use gradflow::nn::{Linear, Module, ReLU, Sequential, Sigmoid, Tanh};
use gradflow::optimizer::{Adam, ExponentialDecay, LinearDecay, SGD};
use gradflow::tensor::{bce_loss, Tensor};

const RED: RGBColor = RGBColor(231, 0, 11);
const BLUE: RGBColor = RGBColor(21, 93, 252);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*// Parameters
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
    points.shuffle(&mut rng);*/

    let n_per_class = 200;
    let seed = 42;
    let noise = 0.1;

    let points = generate_spiral(n_per_class, noise, seed);

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
        .caption("2-class dataset (2 features)", ("Ubuntu", 20).into_font())
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
    let mut x_data = Array2::<f64>::zeros((n_samples, 5)); // x, y, x^2, y^2, x*y
    let mut y_data = Array2::<f64>::zeros((n_samples, 1)); // label

    for (i, (x, y, label)) in points.iter().enumerate() {
        x_data[[i, 0]] = *x;
        x_data[[i, 1]] = *y;
        x_data[[i, 2]] = x.powi(2);
        x_data[[i, 3]] = y.powi(2);
        x_data[[i, 4]] = x*y;

        y_data[[i, 0]] = *label as f64;
    }

    let batch_size = 32;
    let epochs = 300;
    let seed = 42;

    let (x_train, x_test, y_train, y_test) = train_test_split(&x_data, &y_data, 0.2);
    let mut train_dataloader = DataLoader::new(&x_train, &y_train, batch_size, seed);
    let mut test_dataloader = DataLoader::new(&x_test, &y_test, batch_size, seed);

    // Training
    let mut model = Sequential::new(modules![
        Linear::new(5, 16),
        ReLU::new(),
        Linear::new(16, 8),
        ReLU::new(),
        Linear::new(8, 1),
        Sigmoid::new(),
    ]);
    model.apply_init(InitType::KaimingNormal);
    
    let mut optimizer = Adam::new(model.parameters(), 0.003);

    for epoch in 0..epochs {
        let mut epoch_losses: Vec<f64> = vec![];
        for (x_batch, y_batch) in train_dataloader.iter() {
            let x_tensor = Rc::new(RefCell::new(Tensor::new(x_batch.into_dyn(), false)));
            let y_tensor = Rc::new(RefCell::new(Tensor::new(y_batch.into_dyn(), false)));

            // Zero all the gradients
            optimizer.zero_grad();

            // Make predictions
            let y_pred = model.forward(x_tensor);

            // Compute loss and its gradients
            let loss = bce_loss(&y_pred, &y_tensor);
            epoch_losses.push(loss.borrow().data[[0]]);
            loss.borrow_mut().backward();

            // Adjust learning weights
            optimizer.step();
        }
        // Logging
        println!("Epoch {:?} | Loss: {:?}", epoch, epoch_losses.mean());
    }

    // Inference
    let prediction = model.forward(
        Rc::new(RefCell::new(Tensor::new(Array2::from_elem((1, 5), 1.0).into_dyn(), false)))
    );
    println!("Probality of belonging to class 1: {:?}", prediction.borrow().data);
    
    // Plot decision regions
    let filename = "data/decision_regions.svg";
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
                let x_squared = x.powi(2);
                let y_squared = y.powi(2);
                let x_by_y = x*y;

                // Predict
                let tensor_data = Array2::from_shape_vec(
                    (1, 5), vec![x, y, x_squared, y_squared, x_by_y]
                ).unwrap();
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

    // Metrics
    let mut y_true: Vec<usize> = Vec::new();
    let mut y_pred: Vec<usize> = Vec::new();
    for (x_batch, y_batch) in test_dataloader.iter() {
        let input = Rc::new(RefCell::new(Tensor::new(x_batch.into_dyn(), false)));
        let prediction = model_rc.borrow().forward(input.clone());
        let probability_value: ndarray::Array1<f64> = prediction
            .borrow()
            .data.clone()
            .iter()
            .cloned()
            .collect();

        for (pred, true_label) in probability_value.iter().zip(y_batch.iter()) {
            // Binary classification threshold
            let class = if *pred >= 0.5 { 1 } else { 0 };
            y_pred.push(class);
            y_true.push(*true_label as usize);
        }
    }

    // Compute confusion matrix
    let mut confusion_matrix = [[0; 2]; 2];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        confusion_matrix[t][p] += 1;
    }
    println!("Confusion Matrix:");
    println!("[[TN, FP], [FN, TP]] = {:?}", confusion_matrix);

    // Compute precision and recall
    //let tn = confusion_matrix[0][0] as f64;
    let fp = confusion_matrix[0][1] as f64;
    let fn_ = confusion_matrix[1][0] as f64;
    let tp = confusion_matrix[1][1] as f64;

    // Precision: TP / (TP + FP)
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };

    // Recall: TP / (TP + FN)
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

    println!("Precision: {:.4}", precision);
    println!("Recall: {:.4}", recall);

    Ok(())
}