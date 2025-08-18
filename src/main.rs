mod tensor;
mod nn;
mod optimizer;

use std::{cell::RefCell, rc::Rc};
use ndarray::ArrayD;

use tensor::{add, Tensor};
use nn::{Linear, Module};
use optimizer::SGD;

use crate::tensor::bce_loss;

fn train() {
    let model = Linear::new(2, 1);
    let optimizer = SGD { params: model.parameters(), lr: 0.01 };

    for _ in 0..100 {
        // Get data
        let x = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![2], 1.0), false)));
        let y_true = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![1], 2.0), false)));

        // Zero all the gradients
        optimizer.zero_grad();

        // Make predictions
        let y_pred = model.forward(x);

        // Compute loss and its gradients
        let loss = bce_loss(&y_pred, &y_true);
        loss.borrow_mut().backward();

        // Adjust learning weights
        optimizer.step();
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

