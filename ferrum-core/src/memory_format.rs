//! Memory Format Support for Tensors.
//!
//! This module provides memory format abstractions for efficient
//! layout management, particularly for NHWC (channels-last) format
//! which enables Tensor Core acceleration on NVIDIA GPUs.

/// Memory layout format for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Default contiguous format (NCHW for 4D, NCHW for 5D).
    Contiguous,
    /// Channels-last format (NHWC for 4D, NDHWC for 5D).
    /// Enables Tensor Core acceleration on Ampere+ GPUs.
    ChannelsLast,
    /// Channels-last 3D format (NDHWC for 5D).
    ChannelsLast3d,
}

impl MemoryFormat {
    /// Check if this is a channels-last format.
    pub fn is_channels_last(&self) -> bool {
        matches!(
            self,
            MemoryFormat::ChannelsLast | MemoryFormat::ChannelsLast3d
        )
    }

    /// Get the appropriate format for the given dimensionality.
    pub fn for_ndim(ndim: usize) -> Self {
        match ndim {
            4 => MemoryFormat::ChannelsLast,
            5 => MemoryFormat::ChannelsLast3d,
            _ => MemoryFormat::Contiguous,
        }
    }
}

impl Default for MemoryFormat {
    fn default() -> Self {
        MemoryFormat::Contiguous
    }
}

impl std::fmt::Display for MemoryFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryFormat::Contiguous => write!(f, "Contiguous (NCHW)"),
            MemoryFormat::ChannelsLast => write!(f, "ChannelsLast (NHWC)"),
            MemoryFormat::ChannelsLast3d => write!(f, "ChannelsLast3d (NDHWC)"),
        }
    }
}

/// Strides for different memory formats.
pub struct FormatStrides;

impl FormatStrides {
    /// Compute strides for Contiguous (NCHW) format.
    pub fn contiguous(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Compute strides for ChannelsLast (NHWC) format.
    pub fn channels_last(shape: &[usize]) -> Vec<usize> {
        assert_eq!(shape.len(), 4, "NHWC requires 4D tensor");
        let [n, c, h, w] = [shape[0], shape[1], shape[2], shape[3]];
        // NHWC: strides = [C*H*W, 1, W*C, C]
        vec![c * h * w, 1, w * c, c]
    }

    /// Compute strides for ChannelsLast3D (NDHWC) format.
    pub fn channels_last_3d(shape: &[usize]) -> Vec<usize> {
        assert_eq!(shape.len(), 5, "NDHWC requires 5D tensor");
        let [n, c, d, h, w] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
        // NDHWC: strides = [C*D*H*W, 1, W*H*C, W*C, C]
        vec![c * d * h * w, 1, h * w * c, w * c, c]
    }

    /// Compute strides for a given memory format.
    pub fn for_format(shape: &[usize], format: MemoryFormat) -> Vec<usize> {
        match format {
            MemoryFormat::Contiguous => Self::contiguous(shape),
            MemoryFormat::ChannelsLast => Self::channels_last(shape),
            MemoryFormat::ChannelsLast3d => Self::channels_last_3d(shape),
        }
    }
}

/// Check if a tensor is contiguous in the given format.
pub fn is_contiguous(shape: &[usize], strides: &[usize], format: MemoryFormat) -> bool {
    let expected = FormatStrides::for_format(shape, format);
    strides == expected
}

/// Convert tensor data from one memory format to another.
///
/// This performs a physical data copy if the formats differ.
pub fn convert_format<T: Copy>(
    data: &[T],
    src_shape: &[usize],
    src_strides: &[usize],
    dst_format: MemoryFormat,
) -> (Vec<T>, Vec<usize>) {
    let dst_strides = FormatStrides::for_format(src_shape, dst_format);

    if src_strides == dst_strides {
        // Already in target format, no copy needed
        return (data.to_vec(), dst_strides);
    }

    // Compute total elements
    let total: usize = src_shape.iter().product();
    let mut result = vec![data[0]; total]; // uninitialized would be faster

    // Compute offset for each element
    for flat_idx in 0..total {
        // Compute multi-dimensional index from flat index using src strides
        let mut remaining = flat_idx;
        let mut src_offset = 0;
        for (dim, &stride) in src_strides.iter().enumerate() {
            let idx = remaining / stride;
            remaining %= stride;
            src_offset += idx * dst_strides[dim];
        }
        result[src_offset] = data[flat_idx];
    }

    (result, dst_strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        let shape = vec![2, 3, 4, 5];
        let strides = FormatStrides::contiguous(&shape);
        assert_eq!(strides, vec![60, 20, 5, 1]);
    }

    #[test]
    fn test_nhwc_strides() {
        let shape = vec![2, 3, 4, 5];
        let strides = FormatStrides::channels_last(&shape);
        // NHWC: [C*H*W, 1, W*C, C] = [60, 1, 15, 3]
        assert_eq!(strides, vec![60, 1, 15, 3]);
    }

    #[test]
    fn test_ndhwc_strides() {
        let shape = vec![2, 3, 4, 5, 6];
        let strides = FormatStrides::channels_last_3d(&shape);
        // NDHWC: [C*D*H*W, 1, H*W*C, W*C, C] = [360, 1, 90, 18, 3]
        assert_eq!(strides, vec![360, 1, 90, 18, 3]);
    }

    #[test]
    fn test_is_contiguous() {
        let shape = vec![2, 3, 4, 5];
        let strides = FormatStrides::contiguous(&shape);
        assert!(is_contiguous(&shape, &strides, MemoryFormat::Contiguous));
        assert!(!is_contiguous(&shape, &strides, MemoryFormat::ChannelsLast));
    }

    #[test]
    fn test_convert_format() {
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let shape = vec![2, 3, 4, 5]; // NHWC requires 4D
        let src_strides = FormatStrides::contiguous(&shape);

        let (result, dst_strides) =
            convert_format(&data, &shape, &src_strides, MemoryFormat::ChannelsLast);

        assert_eq!(dst_strides, FormatStrides::channels_last(&shape));
        assert_eq!(result.len(), 120);
    }

    #[test]
    fn test_format_display() {
        assert_eq!(MemoryFormat::Contiguous.to_string(), "Contiguous (NCHW)");
        assert_eq!(MemoryFormat::ChannelsLast.to_string(), "ChannelsLast (NHWC)");
    }
}
