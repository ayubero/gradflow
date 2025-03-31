mod tensor;
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

