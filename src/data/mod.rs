use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
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
