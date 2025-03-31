use std::cell::RefCell;
use std::rc::Rc;
use ndarray::ArrayD;

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>,
    pub grad: Option<ArrayD<f64>>,
    requires_grad: bool,
    grad_fn: Option<Rc<RefCell<dyn Fn(&mut Tensor) -> ()>>>, // Backward function
}

impl Tensor {
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        Tensor {
            data,
            grad: None,
            requires_grad,
            grad_fn: None,
        }
    }

    pub fn backward(&mut self) {
        if let Some(grad_fn) = &self.grad_fn {
            let grad_fn_clone = Rc::clone(grad_fn);
            grad_fn_clone.borrow_mut()(self);
        }
    }
}

pub fn add(a: &Rc<RefCell<Tensor>>, b: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let result_data = &a.borrow().data + &b.borrow().data;
    let result = Rc::new(RefCell::new(Tensor::new(result_data, true)));

    if a.borrow().requires_grad || b.borrow().requires_grad {
        let a_clone = Rc::clone(a);
        let b_clone = Rc::clone(b);
        let result_clone = Rc::clone(&result);

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }
        
            // Fix: Avoid borrowing conflicts by storing shape first
            if a_clone.borrow().grad.is_none() {
                let shape = a_clone.borrow().data.raw_dim(); // Immutable borrow ends
                a_clone.borrow_mut().grad = Some(ArrayD::zeros(shape)); // Mutable borrow starts
            }
        
            if b_clone.borrow().grad.is_none() {
                let shape = b_clone.borrow().data.raw_dim();
                b_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
        
            *a_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap();
            *b_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap();
        })));        
    }

    result
}
