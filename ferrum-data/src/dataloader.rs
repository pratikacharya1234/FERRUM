//! DataLoader for batched iteration over datasets.

use std::sync::Arc;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::dataset::Dataset;
use crate::sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};

/// A data loader that provides batched iteration over a dataset.
///
/// Similar to PyTorch's `DataLoader`.
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new data loader.
    pub fn new(dataset: D) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
        }
    }

    /// Set the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Drop the last incomplete batch.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set number of worker threads for parallel loading.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Get the number of batches.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get reference to the dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Create an iterator over batches.
    pub fn iter(&self) -> DataLoaderIter<D> {
        let sampler: Box<dyn Sampler> = if self.shuffle {
            Box::new(RandomSampler::new(self.dataset.len()))
        } else {
            Box::new(SequentialSampler::new(self.dataset.len()))
        };

        let batch_sampler = BatchSampler::new(sampler, self.batch_size, self.drop_last);

        DataLoaderIter {
            dataset: self.dataset.clone(),
            batch_sampler: Mutex::new(batch_sampler),
            num_workers: self.num_workers,
        }
    }
}

impl<D: Dataset> IntoIterator for DataLoader<D>
where
    D::Sample: Send,
{
    type Item = Vec<D::Sample>;
    type IntoIter = DataLoaderIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        let sampler: Box<dyn Sampler> = if self.shuffle {
            Box::new(RandomSampler::new(self.dataset.len()))
        } else {
            Box::new(SequentialSampler::new(self.dataset.len()))
        };

        let batch_sampler = BatchSampler::new(sampler, self.batch_size, self.drop_last);

        DataLoaderIter {
            dataset: self.dataset,
            batch_sampler: Mutex::new(batch_sampler),
            num_workers: self.num_workers,
        }
    }
}

impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D>
where
    D::Sample: Send,
{
    type Item = Vec<D::Sample>;
    type IntoIter = DataLoaderIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over batches from a DataLoader.
pub struct DataLoaderIter<D: Dataset> {
    dataset: Arc<D>,
    batch_sampler: Mutex<BatchSampler<Box<dyn Sampler>>>,
    num_workers: usize,
}

impl<D: Dataset> Iterator for DataLoaderIter<D>
where
    D::Sample: Send,
{
    type Item = Vec<D::Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = {
            let mut sampler = self.batch_sampler.lock();
            sampler.next()?
        };

        if self.num_workers > 0 {
            // Parallel loading
            let samples: Vec<D::Sample> = indices
                .par_iter()
                .map(|&idx| self.dataset.get(idx))
                .collect();
            Some(samples)
        } else {
            // Sequential loading
            let samples: Vec<D::Sample> = indices
                .iter()
                .map(|&idx| self.dataset.get(idx))
                .collect();
            Some(samples)
        }
    }
}

/// Collate function type for combining samples into a batch.
pub type CollateFn<S, B> = Box<dyn Fn(Vec<S>) -> B + Send + Sync>;

/// A data loader with a custom collate function.
pub struct CollatedDataLoader<D: Dataset, B> {
    loader: DataLoader<D>,
    collate_fn: CollateFn<D::Sample, B>,
}

impl<D: Dataset, B> CollatedDataLoader<D, B> {
    /// Create a new collated data loader.
    pub fn new(loader: DataLoader<D>, collate_fn: CollateFn<D::Sample, B>) -> Self {
        Self { loader, collate_fn }
    }

    /// Create an iterator over collated batches.
    pub fn iter(&self) -> impl Iterator<Item = B> + '_
    where
        D::Sample: Send,
    {
        self.loader.iter().map(|batch| (self.collate_fn)(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use ferrum_core::{DType, Device, Tensor};

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
    fn test_dataloader_sequential() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        let loader = DataLoader::new(dataset).batch_size(2);
        let batches: Vec<_> = loader.iter().collect();
        
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![1, 2]);
        assert_eq!(batches[1], vec![3, 4]);
        assert_eq!(batches[2], vec![5]);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        let loader = DataLoader::new(dataset).batch_size(2).drop_last(true);
        let batches: Vec<_> = loader.iter().collect();
        
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset = SimpleDataset {
            data: (0..100).collect(),
        };
        let loader = DataLoader::new(dataset).batch_size(10).shuffle(true);
        let batches: Vec<_> = loader.iter().collect();
        
        assert_eq!(batches.len(), 10);
        // Check that data is shuffled (very unlikely to be in order)
        let first_batch = &batches[0];
        let is_sequential = first_batch == &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        // This could randomly pass, but it's very unlikely
        assert!(!is_sequential || batches[1] != vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    }

    #[test]
    fn test_dataloader_len() {
        let dataset = SimpleDataset {
            data: vec![1, 2, 3, 4, 5],
        };
        
        let loader = DataLoader::new(dataset).batch_size(2);
        assert_eq!(loader.len(), 3); // ceil(5/2) = 3
    }
}
