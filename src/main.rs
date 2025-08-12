/*mod tensor;
mod nn;
mod optimizer;

use std::{cell::RefCell, rc::Rc};
use ndarray::ArrayD;

use tensor::{add, Tensor};
use nn::Linear;
use optimizer::SGD;

fn train() {
    let model = Linear::new(2, 1);
    let optimizer = SGD { lr: 0.01 };

    for _ in 0..100 {
        let x = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![2], 1.0), false)));
        let y_true = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![1], 2.0), false)));

        let y_pred = model.forward(&x);
        let loss = add(&y_pred, &y_true); // Fake loss for demo

        loss.borrow_mut().backward();
        optimizer.step(vec![model.weights.clone(), model.bias.clone()]);
    }
}

fn main() {
    let a = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![2], 3.0), true)));
    let b = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![2], 2.0), true)));

    let c = add(&a, &b); // c = a + b
    c.borrow_mut().grad = Some(ArrayD::from_elem(vec![2], 1.0)); // Set initial gradient

    c.borrow_mut().backward(); // Compute gradients

    println!("Gradient of a: {:?}", a.borrow().grad);
    println!("Gradient of b: {:?}", b.borrow().grad);

    //train();
}
*/

use plotters::prelude::*;
use rand::SeedableRng;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let n_per_class = 200;
    let seed = 42u64; // reproducible
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Gaussian blob parameters: means and standard deviation
    let mean0 = (-1.0f64, -0.5f64);
    let mean1 = (1.0f64, 0.5f64);
    let std_dev = 0.6f64;

    let normal = Normal::new(0.0, std_dev).unwrap();

    // Storage
    let mut points: Vec<(f64, f64, u8)> = Vec::with_capacity(n_per_class * 2);

    // Generate class 0
    for _ in 0..n_per_class {
        let x = mean0.0 + normal.sample(&mut rng);
        let y = mean0.1 + normal.sample(&mut rng);
        points.push((x, y, 0));
    }

    // Generate class 1
    for _ in 0..n_per_class {
        let x = mean1.0 + normal.sample(&mut rng);
        let y = mean1.1 + normal.sample(&mut rng);
        points.push((x, y, 1));
    }

    // Optionally shuffle the dataset
    points.shuffle(&mut rng);

    // Write CSV
    let file = File::create("data.csv")?;
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
    let filename = "scatter.png";
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("2-class logistic-like dataset (2 features)", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - pad_x)..(x_max + pad_x),
            (y_min - pad_y)..(y_max + pad_y),
        )?;

    chart.configure_mesh().draw()?;

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
        .label_font(("sans-serif", 15))
        .draw()?;
    // manual legend: draw small markers and text

    println!("Wrote scatter plot to {}", filename);

    Ok(())
}

