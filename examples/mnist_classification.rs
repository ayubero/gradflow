use std::cell::RefCell;
use std::rc::Rc;

use gradflow::init::InitType;
use gradflow::nn::{Flatten, Linear, MaxPool2d, ReLU};
use gradflow::optimizer::Adam;
use gradflow::tensor::{cross_entropy_loss, Tensor};
use mnist::*;
use ndarray::prelude::*;

use gradflow::modules;
use gradflow::{data::DataLoader, nn::{Module, Conv2d, Sequential}};
use statrs::statistics::Statistics;

fn main() {
    let train_set_length = 3200 as usize;
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(train_set_length as u32)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((train_set_length, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);
    let train_data = train_data.insert_axis(Axis(1));
    println!("{:#.1?}\n",train_data.slice(s![image_num, .., .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((train_set_length, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let mut train_dataloader = DataLoader::new(&train_data, &train_labels, 32, 42);
   
    let epochs = 100;

    // Training
    // Model: LeNet-5
    let mut model = Sequential::new(modules![
        Conv2d::new(1, 16, 3, 1, 1),
        ReLU::new(),
        MaxPool2d::new(2, None, 0),
        Conv2d::new(16, 32, 3, 1, 1),
        ReLU::new(),
        MaxPool2d::new(2, None, 0),
        Flatten::new(),
        Linear::new(21632, 64),
        Linear::new(64, 10)
    ]);
    model.apply_init(InitType::KaimingNormal);
    
    let mut optimizer = Adam::new(model.parameters(), 0.003);

    for epoch in 0..epochs {
        let mut epoch_losses: Vec<f64> = vec![];
        for (i, (x_batch, y_batch)) in train_dataloader.iter().enumerate() {
            let x_tensor = Rc::new(RefCell::new(Tensor::new(x_batch.into_dyn(), false)));
            let y_tensor = Rc::new(RefCell::new(Tensor::new(y_batch.into_dyn(), false)));

            // Zero all the gradients
            optimizer.zero_grad();

            // Make predictions
            let y_pred = model.forward(x_tensor);

            // Compute loss and its gradients
            let loss = cross_entropy_loss(&y_pred, &y_tensor);
            epoch_losses.push(loss.borrow().data[[0]]);
            loss.borrow_mut().backward();

            // Adjust learning weights
            optimizer.step();
            //break;
            println!("Batch {:?} completed", i);
        }
        // Logging
        println!("Epoch {:?} | Loss: {:?}", epoch, epoch_losses.mean());
    }
}