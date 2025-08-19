use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use std::f64::consts::PI;
use std::iter::Iterator;

pub struct DataLoader<'a> {
    x: &'a Array2<f64>,
    y: &'a Array2<f64>,
    batch_size: usize,
    rng: StdRng,
}

impl<'a> DataLoader<'a> {
    pub fn new(x: &'a Array2<f64>, y: &'a Array2<f64>, batch_size: usize, seed: u64) -> Self {
        Self {
            x,
            y,
            batch_size,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    // Returns a fresh iterator over batches
    pub fn iter(&mut self) -> DataLoaderIter<'a> {
        let n_samples = self.x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut self.rng); // shuffle at start of epoch
        DataLoaderIter {
            x: self.x,
            y: self.y,
            indices,
            batch_size: self.batch_size,
            current: 0,
        }
    }
}

// The actual iterator returned for each epoch
pub struct DataLoaderIter<'a> {
    x: &'a Array2<f64>,
    y: &'a Array2<f64>,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (Array2<f64>, Array2<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];

        let x_batch = Array2::from_shape_fn((batch_indices.len(), self.x.ncols()), |(i, j)| {
            self.x[[batch_indices[i], j]]
        });

        let y_batch = Array2::from_shape_fn((batch_indices.len(), self.y.ncols()), |(i, j)| {
            self.y[[batch_indices[i], j]]
        });

        self.current = end;
        Some((x_batch, y_batch))
    }
}

/// Splits dataset into train and test sets
pub fn train_test_split(
    x_data: &Array2<f64>,
    y_data: &Array2<f64>,
    test_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_samples = x_data.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let test_size = (n_samples as f64 * test_ratio).round() as usize;

    let test_indices = &indices[..test_size];
    let train_indices = &indices[test_size..];

    let x_train = Array2::from_shape_fn((train_indices.len(), x_data.ncols()), |(i, j)| {
        x_data[[train_indices[i], j]]
    });

    let x_test = Array2::from_shape_fn((test_indices.len(), x_data.ncols()), |(i, j)| {
        x_data[[test_indices[i], j]]
    });

    let y_train = Array2::from_shape_fn((train_indices.len(), y_data.ncols()), |(i, j)| {
        y_data[[train_indices[i], j]]
    });

    let y_test = Array2::from_shape_fn((test_indices.len(), y_data.ncols()), |(i, j)| {
        y_data[[test_indices[i], j]]
    });

    (x_train, x_test, y_train, y_test)
}

// ========
// Datasets
// ========

pub fn generate_spiral(n_per_class: usize, noise: f64, seed: u64) -> Vec<(f64, f64, u8)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n_per_class * 2);

    for class in 0..2 {
        for i in 0..n_per_class {
            // t goes from 0 to 1
            let t = i as f64 / n_per_class as f64;

            // Angle: 2π * t * number of turns + offset for class
            let angle = 4.0 * PI * t + (class as f64) * PI;

            // Radius grows linearly with t
            let r = t;

            // Add Gaussian noise
            let x = r * angle.cos() + noise * rng.gen::<f64>();
            let y = r * angle.sin() + noise * rng.gen::<f64>();

            points.push((x, y, class as u8));
        }
    }

    points.shuffle(&mut rng);
    points
}