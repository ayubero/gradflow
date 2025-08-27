use ndarray::Array2;
use statrs::statistics::Statistics;
use std::cell::RefCell;
use std::rc::Rc;

use gradflow::modules;
use gradflow::data::{generate_spiral, train_test_split, DataLoader};
use gradflow::init::InitType;
use gradflow::nn::{Linear, Module, PolyExpand, ReLU, Sequential, Sigmoid};
use gradflow::optimizer::Adam;
use gradflow::tensor::{bce_loss, Tensor};
use gradflow::plot::{plot_decision_regions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_per_class = 200;
    let seed = 42;
    let noise = 0.1;

    let points = generate_spiral(n_per_class, noise, seed);

    // Separate features and labels
    let n_samples = points.len();
    let mut x_data = Array2::<f64>::zeros((n_samples, 2)); // x, y
    let mut y_data = Array2::<f64>::zeros((n_samples, 1)); // Label

    for (i, (x, y, label)) in points.iter().enumerate() {
        x_data[[i, 0]] = *x;
        x_data[[i, 1]] = *y;
        y_data[[i, 0]] = *label as f64;
    }

    let batch_size = 32;
    let epochs = 300;
    let seed = 42;

    let (x_train, _x_test, y_train, _y_test) = train_test_split(&x_data, &y_data, 0.2);
    let mut train_dataloader = DataLoader::new(&x_train, &y_train, batch_size, seed);

    // Training
    let mut model = Sequential::new(modules![
        PolyExpand::new(2), // Polynomial expansion (x,y → [x,y,x^2,y^2,xy])
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
        // Create animation frame
        /*let frame_name = format!("data/example_spiral_frames/frame_{}.svg", epoch);
        plot_decision_regions(&points, &frame_name, &model).expect("Couldn't plot decision regions");*/

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
    println!("Probability of belonging to class 1: {:?}", prediction.borrow().data);
    
    // Plot decision regions
    let filename = "data/spiral_decision_regions.svg";
    plot_decision_regions(&points, filename, &model).expect("Couldn't plot decision regions");

    Ok(())
}