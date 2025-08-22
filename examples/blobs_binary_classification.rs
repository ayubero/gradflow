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

use gradflow::modules;
use gradflow::data::{train_test_split, DataLoader};
use gradflow::nn::{Linear, Module, ReLU, Sequential, Sigmoid};
use gradflow::optimizer::SGD;
use gradflow::tensor::{bce_loss, Tensor};
use gradflow::plot::{plot_decision_regions, plot_scatter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let n_per_class = 200;
    let seed = 42u64; // reproducible
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Gaussian blob parameters: means and covariance matrices
    let mean0 = vec![-1.0f64, -0.5f64];
    let mean1 = vec![1.0f64, 0.5f64];

    // Covariance matrices
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
    let file = File::create("data/blobs_data.csv")?;
    let mut w = BufWriter::new(file);
    writeln!(w, "x,y,label")?;
    for (x, y, label) in &points {
        writeln!(w, "{:.6},{:.6},{}", x, y, label)?;
    }
    w.flush()?;
    println!("Wrote dataset to data.csv ({} samples).", points.len());

    // Draw scatter plot
    let filename = "data/blobs_scatter.svg";
    plot_scatter(&points, filename).expect("Couldn't plot scatter");

    // Separate features and labels
    let n_samples = points.len();
    let mut x_data = Array2::<f64>::zeros((n_samples, 2)); // x, y
    let mut y_data = Array2::<f64>::zeros((n_samples, 1)); // label

    for (i, (x, y, label)) in points.iter().enumerate() {
        x_data[[i, 0]] = *x;
        x_data[[i, 1]] = *y;
        y_data[[i, 0]] = *label as f64;
    }

    let (x_train, x_test, y_train, y_test) = train_test_split(&x_data, &y_data, 0.2);
    let mut train_dataloader = DataLoader::new(&x_train, &y_train, 32, 42);
    let mut test_dataloader = DataLoader::new(&x_test, &y_test, 32, 42);

    // Training
    let model = Sequential::new(modules![
        Linear::new(2, 16),
        ReLU::new(),
        Linear::new(16, 8),
        ReLU::new(),
        Linear::new(8, 1),
        Sigmoid::new(),
    ]);

    let epochs = 300;
    
    let mut optimizer = SGD {
        params: model.parameters(), 
        lr: 0.5, 
        scheduler: None, 
        step_count: 0
    };

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
        //println!("Epoch {:?} | Loss: {:?}", epoch, epoch_losses.mean());
    }

    // Inference
    let prediction = model.forward(
        Rc::new(RefCell::new(Tensor::new(Array2::from_elem((1, 2), 1.0).into_dyn(), false)))
    );
    println!("Probability of belonging to class 1: {:?}", prediction.borrow().data);

    println!("x_train {:?}", x_train.shape());
    println!("x_test {:?}", x_test.shape());
    
    // Plot decision regions
    let filename = "data/blobs_decision_regions.svg";
    plot_decision_regions(&points, filename, &model).expect("Couldn't plot decision regions");

    // Metrics
    let mut y_true: Vec<usize> = Vec::new();
    let mut y_pred: Vec<usize> = Vec::new();
    for (x_batch, y_batch) in test_dataloader.iter() {
        let input = Rc::new(RefCell::new(Tensor::new(x_batch.into_dyn(), false)));
        let prediction = model.forward(input.clone());
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