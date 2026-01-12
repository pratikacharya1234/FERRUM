//! Dataset trait and implementations.

use ferrum_core::Tensor;

/// A dataset that provides access to data samples.
///
/// Similar to PyTorch's `Dataset` class.
pub trait Dataset: Send + Sync {
    /// The type of a single sample.
    type Sample;

    /// Get the number of samples in the dataset.
    fn len(&self) -> usize;

    /// Check if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a sample by index.
    fn get(&self, index: usize) -> Self::Sample;
}

/// A simple dataset that holds tensors in memory.
pub struct TensorDataset {
    /// Input tensors.
    inputs: Vec<Tensor>,
    /// Target tensors.
    targets: Vec<Tensor>,
}

impl TensorDataset {
    /// Create a new tensor dataset.
    pub fn new(inputs: Vec<Tensor>, targets: Vec<Tensor>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Inputs and targets must have the same length"
        );
        Self { inputs, targets }
    }

    /// Create from a single input and target tensor (split along first dimension).
    pub fn from_tensors(inputs: Tensor, targets: Tensor) -> Self {
        // For simplicity, we'll store them as single tensors
        // In a real implementation, we'd split along batch dimension
        Self {
            inputs: vec![inputs],
            targets: vec![targets],
        }
    }
}

impl Dataset for TensorDataset {
    type Sample = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, index: usize) -> Self::Sample {
        (self.inputs[index].clone(), self.targets[index].clone())
    }
}

/// A dataset that wraps another dataset and applies a transformation.
pub struct MapDataset<D, F> {
    dataset: D,
    transform: F,
}

impl<D, F, T> MapDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Sample) -> T + Send + Sync,
{
    /// Create a new mapped dataset.
    pub fn new(dataset: D, transform: F) -> Self {
        Self { dataset, transform }
    }
}

impl<D, F, T> Dataset for MapDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Sample) -> T + Send + Sync,
    T: Send,
{
    type Sample = T;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Self::Sample {
        (self.transform)(self.dataset.get(index))
    }
}

/// A dataset that concatenates multiple datasets.
pub struct ConcatDataset<D> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Create a new concatenated dataset.
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;
        for d in &datasets {
            total += d.len();
            cumulative_sizes.push(total);
        }
        Self {
            datasets,
            cumulative_sizes,
        }
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Sample = D::Sample;

    fn len(&self) -> usize {
        *self.cumulative_sizes.last().unwrap_or(&0)
    }

    fn get(&self, index: usize) -> Self::Sample {
        // Find which dataset this index belongs to
        let dataset_idx = self
            .cumulative_sizes
            .iter()
            .position(|&size| index < size)
            .expect("Index out of bounds");

        let prev_size = if dataset_idx == 0 {
            0
        } else {
            self.cumulative_sizes[dataset_idx - 1]
        };

        self.datasets[dataset_idx].get(index - prev_size)
    }
}

/// A subset of a dataset.
pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    /// Create a new subset.
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Sample = D::Sample;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Self::Sample {
        self.dataset.get(self.indices[index])
    }
}

/// Split a dataset into train and test subsets.
pub fn train_test_split<D: Dataset>(
    dataset: D,
    test_ratio: f64,
    shuffle: bool,
) -> (Subset<D>, Subset<D>)
where
    D: Clone,
{
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let n = dataset.len();
    let test_size = (n as f64 * test_ratio).ceil() as usize;
    let train_size = n - test_size;

    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        indices.shuffle(&mut thread_rng());
    }

    let train_indices = indices[..train_size].to_vec();
    let test_indices = indices[train_size..].to_vec();

    (
        Subset::new(dataset.clone(), train_indices),
        Subset::new(dataset, test_indices),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{DType, Device};

    struct SimpleDataset {
        data: Vec<i32>,
    }

    impl Dataset for SimpleDataset {
        type Sample = i32;

        fn len(&self) -> usize {
            self.data.len()
        }

        fn get(&self, index: usize) -> i32 {
            self.data[index]
        }
    }

    #[test]
    fn test_simple_dataset() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0), 1);
        assert_eq!(dataset.get(4), 5);
    }

    #[test]
    fn test_map_dataset() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        let mapped = MapDataset::new(dataset, |x| x * 2);
        assert_eq!(mapped.len(), 5);
        assert_eq!(mapped.get(0), 2);
        assert_eq!(mapped.get(2), 6);
    }

    #[test]
    fn test_concat_dataset() {
        let d1 = SimpleDataset {
            data: vec![1, 2, 3],
        };
        let d2 = SimpleDataset {
            data: vec![4, 5, 6],
        };
        let concat = ConcatDataset::new(vec![d1, d2]);
        assert_eq!(concat.len(), 6);
        assert_eq!(concat.get(0), 1);
        assert_eq!(concat.get(3), 4);
    }

    #[test]
    fn test_subset() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        let subset = Subset::new(dataset, vec![0, 2, 4]);
        assert_eq!(subset.len(), 3);
        assert_eq!(subset.get(0), 1);
        assert_eq!(subset.get(1), 3);
        assert_eq!(subset.get(2), 5);
    }
}
