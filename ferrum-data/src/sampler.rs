//! Samplers for data loading.

use rand::prelude::*;

/// A sampler that yields indices.
pub trait Sampler: Send {
    /// Get the length of the sampler.
    fn len(&self) -> usize;

    /// Check if the sampler is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the next index.
    fn next_index(&mut self) -> Option<usize>;
}

// Implement Sampler for boxed trait objects
impl Sampler for Box<dyn Sampler> {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn next_index(&mut self) -> Option<usize> {
        (**self).next_index()
    }
}

/// Sequential sampler - yields indices in order.
pub struct SequentialSampler {
    current: usize,
    len: usize,
}

impl SequentialSampler {
    /// Create a new sequential sampler.
    pub fn new(len: usize) -> Self {
        Self { current: 0, len }
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.next_index()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.current;
        (remaining, Some(remaining))
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.len
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current < self.len {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Random sampler - yields indices in random order.
pub struct RandomSampler {
    indices: Vec<usize>,
    current: usize,
}

impl RandomSampler {
    /// Create a new random sampler.
    pub fn new(len: usize) -> Self {
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut thread_rng());
        Self { indices, current: 0 }
    }

    /// Create with a specific seed for reproducibility.
    pub fn with_seed(len: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut rng);
        Self { indices, current: 0 }
    }

    /// Reshuffle the sampler.
    pub fn reshuffle(&mut self) {
        self.indices.shuffle(&mut thread_rng());
        self.current = 0;
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.next_index()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.current;
        (remaining, Some(remaining))
    }
}

impl Sampler for RandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Weighted random sampler - samples with replacement based on weights.
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    cumulative: Vec<f64>,
    num_samples: usize,
    current: usize,
}

impl WeightedRandomSampler {
    /// Create a new weighted random sampler.
    pub fn new(weights: Vec<f64>, num_samples: usize) -> Self {
        let total: f64 = weights.iter().sum();
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut sum = 0.0;
        for w in &weights {
            sum += w / total;
            cumulative.push(sum);
        }
        Self {
            weights,
            cumulative,
            num_samples,
            current: 0,
        }
    }
}

impl Iterator for WeightedRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.next_index()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_samples - self.current;
        (remaining, Some(remaining))
    }
}

impl Sampler for WeightedRandomSampler {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current >= self.num_samples {
            return None;
        }
        self.current += 1;

        let mut rng = thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        let idx = self
            .cumulative
            .iter()
            .position(|&c| r < c)
            .unwrap_or(self.weights.len() - 1);
        Some(idx)
    }
}

/// Batch sampler - wraps another sampler and yields batches of indices.
pub struct BatchSampler<S> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Create a new batch sampler.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }
}

impl<S: Sampler> Iterator for BatchSampler<S> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.sampler.next_index() {
                Some(idx) => batch.push(idx),
                None => break,
            }
        }

        if batch.is_empty() {
            None
        } else if batch.len() < self.batch_size && self.drop_last {
            None
        } else {
            Some(batch)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.sampler.len();
        let batches = if self.drop_last {
            remaining / self.batch_size
        } else {
            (remaining + self.batch_size - 1) / self.batch_size
        };
        (batches, Some(batches))
    }
}

/// Subset random sampler - samples from a subset of indices.
pub struct SubsetRandomSampler {
    indices: Vec<usize>,
    current: usize,
}

impl SubsetRandomSampler {
    /// Create a new subset random sampler.
    pub fn new(mut indices: Vec<usize>) -> Self {
        indices.shuffle(&mut thread_rng());
        Self { indices, current: 0 }
    }
}

impl Iterator for SubsetRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.next_index()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.current;
        (remaining, Some(remaining))
    }
}

impl Sampler for SubsetRandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Distributed sampler for multi-GPU training.
pub struct DistributedSampler {
    indices: Vec<usize>,
    current: usize,
}

impl DistributedSampler {
    /// Create a new distributed sampler.
    ///
    /// # Arguments
    /// * `len` - Total dataset length
    /// * `num_replicas` - Total number of processes
    /// * `rank` - Rank of current process (0-indexed)
    /// * `shuffle` - Whether to shuffle indices
    pub fn new(len: usize, num_replicas: usize, rank: usize, shuffle: bool) -> Self {
        // Pad dataset to be evenly divisible
        let padded_len = ((len + num_replicas - 1) / num_replicas) * num_replicas;
        
        let mut indices: Vec<usize> = (0..len).collect();
        
        // Pad with wrapped indices
        while indices.len() < padded_len {
            indices.push(indices[indices.len() % len]);
        }
        
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        
        // Get this rank's portion
        let per_replica = padded_len / num_replicas;
        let start = rank * per_replica;
        let end = start + per_replica;
        let indices = indices[start..end].to_vec();
        
        Self { indices, current: 0 }
    }

    /// Set the epoch for shuffling (for reproducibility across ranks).
    pub fn set_epoch(&mut self, epoch: usize) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(epoch as u64);
        self.indices.shuffle(&mut rng);
        self.current = 0;
    }
}

impl Iterator for DistributedSampler {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        self.next_index()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.current;
        (remaining, Some(remaining))
    }
}

impl Sampler for DistributedSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn next_index(&mut self) -> Option<usize> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let sampler = SequentialSampler::new(5);
        let indices: Vec<_> = sampler.collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler() {
        let sampler = RandomSampler::new(5);
        let indices: Vec<_> = sampler.collect();
        assert_eq!(indices.len(), 5);
        // Check all indices are present (shuffled)
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler_reproducibility() {
        let s1 = RandomSampler::with_seed(10, 42);
        let s2 = RandomSampler::with_seed(10, 42);
        let i1: Vec<_> = s1.collect();
        let i2: Vec<_> = s2.collect();
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_batch_sampler() {
        let sampler = SequentialSampler::new(10);
        let batch_sampler = BatchSampler::new(sampler, 3, false);
        let batches: Vec<_> = batch_sampler.collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[3], vec![9]); // Last batch is partial
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let sampler = SequentialSampler::new(10);
        let batch_sampler = BatchSampler::new(sampler, 3, true);
        let batches: Vec<_> = batch_sampler.collect();
        assert_eq!(batches.len(), 3); // Last partial batch dropped
    }

    #[test]
    fn test_distributed_sampler() {
        // 10 samples, 3 replicas, rank 0
        let sampler = DistributedSampler::new(10, 3, 0, false);
        let indices: Vec<_> = sampler.collect();
        // Each replica gets ceil(10/3) = 4 samples
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_distributed_sampler_all_ranks() {
        let len = 10;
        let num_replicas = 3;
        let mut all_indices = Vec::new();
        
        for rank in 0..num_replicas {
            let sampler = DistributedSampler::new(len, num_replicas, rank, false);
            all_indices.extend(sampler);
        }
        
        // All original indices should be covered
        for i in 0..len {
            assert!(all_indices.contains(&i), "Missing index {}", i);
        }
    }

    #[test]
    fn test_weighted_sampler() {
        let weights = vec![0.0, 1.0, 0.0]; // Only sample index 1
        let sampler = WeightedRandomSampler::new(weights, 10);
        let indices: Vec<_> = sampler.collect();
        assert_eq!(indices.len(), 10);
        // All should be index 1
        for idx in indices {
            assert_eq!(idx, 1);
        }
    }

    #[test]
    fn test_subset_random_sampler() {
        let subset = vec![2, 5, 7];
        let sampler = SubsetRandomSampler::new(subset.clone());
        let indices: Vec<_> = sampler.collect();
        assert_eq!(indices.len(), 3);
        // All indices should be from subset
        for idx in indices {
            assert!(subset.contains(&idx));
        }
    }
}
