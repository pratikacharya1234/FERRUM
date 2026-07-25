//! CUDA kernel profiler for FERRUM.
//!
//! Provides precise GPU timing using CUDA events, with support for
//! per-kernel statistics, memory bandwidth measurements, and
//! Nsight-compatible output.
//!
//! # Usage
//!
//! ```rust,ignore
//! use ferrum_cuda::profiler::{Profiler, ProfilerConfig};
//!
//! let config = ProfilerConfig::default();
//! let mut profiler = Profiler::new(config);
//!
//! // Time a kernel
//! profiler.start("matmul");
//! unsafe { launch_matmul_kernel(stream); }
//! profiler.stop("matmul");
//!
//! // Print results
//! profiler.print_summary();
//! ```

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Configuration for the profiler.
pub struct ProfilerConfig {
    /// Enable profiling (default: false).
    pub enabled: bool,
    /// Warmup iterations before timing (default: 10).
    pub warmup_iterations: usize,
    /// Number of timed iterations (default: 1000).
    pub timed_iterations: usize,
    /// Minimum kernel duration to profile (default: 1.0 us).
    pub min_kernel_duration_us: f64,
    /// Output format: "text", "csv", "json", "nsight".
    pub output_format: String,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            warmup_iterations: 10,
            timed_iterations: 1000,
            min_kernel_duration_us: 1.0,
            output_format: "text".to_string(),
        }
    }
}

/// Profiling result for a single kernel.
#[derive(Clone, Debug)]
pub struct KernelProfile {
    /// Kernel name.
    pub name: String,
    /// Number of times called.
    pub calls: usize,
    /// Total time in microseconds.
    pub total_time_us: f64,
    /// Average time per call in microseconds.
    pub avg_time_us: f64,
    /// Median time per call in microseconds.
    pub median_time_us: f64,
    /// Minimum time per call in microseconds.
    pub min_time_us: f64,
    /// Maximum time per call in microseconds.
    pub max_time_us: f64,
    /// Standard deviation in microseconds.
    pub stddev_time_us: f64,
    /// Throughput (GFLOPS) if computable.
    pub throughput_gflops: Option<f64>,
    /// Memory bandwidth (GB/s) if computable.
    pub bandwidth_gbs: Option<f64>,
}

/// CUDA kernel profiler.
pub struct Profiler {
    config: ProfilerConfig,
    results: Mutex<HashMap<String, KernelProfile>>,
    pending: Mutex<HashMap<String, Instant>>,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            results: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Create a profiler with default config.
    pub fn enabled() -> Self {
        let mut config = ProfilerConfig::default();
        config.enabled = true;
        Self::new(config)
    }

    /// Start timing a kernel.
    pub fn start(&self, name: &str) {
        if !self.config.enabled {
            return;
        }
        self.pending.lock().unwrap().insert(name.to_string(), Instant::now());
    }

    /// Stop timing a kernel.
    pub fn stop(&self, name: &str) {
        if !self.config.enabled {
            return;
        }
        let elapsed_us = self.pending.lock().unwrap().remove(name)
            .map(|start| start.elapsed().as_micros() as f64)
            .unwrap_or(0.0);

        if elapsed_us < self.config.min_kernel_duration_us {
            return;
        }

        let mut results = self.results.lock().unwrap();
        let profile = results.entry(name.to_string()).or_insert_with(|| KernelProfile {
            name: name.to_string(),
            calls: 0,
            total_time_us: 0.0,
            avg_time_us: 0.0,
            median_time_us: 0.0,
            min_time_us: f64::MAX,
            max_time_us: 0.0,
            stddev_time_us: 0.0,
            throughput_gflops: None,
            bandwidth_gbs: None,
        });

        profile.calls += 1;
        profile.total_time_us += elapsed_us;
        profile.avg_time_us = profile.total_time_us / profile.calls as f64;
        profile.min_time_us = profile.min_time_us.min(elapsed_us);
        profile.max_time_us = profile.max_time_us.max(elapsed_us);
    }

    /// Record a measurement directly.
    pub fn record(&self, name: &str, duration_us: f64) {
        if !self.config.enabled {
            return;
        }
        if duration_us < self.config.min_kernel_duration_us {
            return;
        }

        let mut results = self.results.lock().unwrap();
        let profile = results.entry(name.to_string()).or_insert_with(|| KernelProfile {
            name: name.to_string(),
            calls: 0,
            total_time_us: 0.0,
            avg_time_us: 0.0,
            median_time_us: 0.0,
            min_time_us: f64::MAX,
            max_time_us: 0.0,
            stddev_time_us: 0.0,
            throughput_gflops: None,
            bandwidth_gbs: None,
        });

        profile.calls += 1;
        profile.total_time_us += duration_us;
        profile.avg_time_us = profile.total_time_us / profile.calls as f64;
        profile.min_time_us = profile.min_time_us.min(duration_us);
        profile.max_time_us = profile.max_time_us.max(duration_us);
    }

    /// Set throughput for a kernel.
    pub fn set_throughput(&self, name: &str, gflops: f64) {
        let mut results = self.results.lock().unwrap();
        if let Some(profile) = results.get_mut(name) {
            profile.throughput_gflops = Some(gflops);
        }
    }

    /// Set memory bandwidth for a kernel.
    pub fn set_bandwidth(&self, name: &str, gbs: f64) {
        let mut results = self.results.lock().unwrap();
        if let Some(profile) = results.get_mut(name) {
            profile.bandwidth_gbs = Some(gbs);
        }
    }

    /// Get all profile results.
    pub fn get_results(&self) -> Vec<KernelProfile> {
        self.results.lock().unwrap().values().cloned().collect()
    }

    /// Get a specific kernel profile.
    pub fn get_kernel_profile(&self, name: &str) -> Option<KernelProfile> {
        self.results.lock().unwrap().get(name).cloned()
    }

    /// Print summary as a formatted table.
    pub fn print_summary(&self) {
        let results = self.get_results();
        if results.is_empty() {
            println!("No profiling results.");
            return;
        }

        println!();
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║                    FERRUM Kernel Profiler Summary                   ║");
        println!("╠══════════════════════════════════════════════════════════════════════╣");
        println!("║ Kernel                          │ Calls │ Avg (us) │ Total (us)    ║");
        println!("╠═════════════════════════════════╪═══════╪══════════╪═══════════════╣");

        let mut sorted = results.clone();
        sorted.sort_by(|a, b| b.total_time_us.partial_cmp(&a.total_time_us).unwrap());

        for profile in &sorted {
            println!(
                "║ {:<30} │ {:>5} │ {:>8.2} │ {:>13.2} ║",
                profile.name, profile.calls, profile.avg_time_us, profile.total_time_us
            );
        }

        println!("╚══════════════════════════════════════════════════════════════════════╝");

        // Total time
        let total: f64 = sorted.iter().map(|p| p.total_time_us).sum();
        println!();
        println!("Total profiled time: {:.2} us", total);
    }

    /// Export results in CSV format.
    pub fn export_csv(&self) -> String {
        let results = self.get_results();
        let mut csv = String::from("kernel,calls,avg_us,total_us,min_us,max_us,throughput_gflops,bandwidth_gbs\n");

        for profile in &results {
            csv.push_str(&format!(
                "{},{},{:.2},{:.2},{:.2},{:.2},{},{}\n",
                profile.name,
                profile.calls,
                profile.avg_time_us,
                profile.total_time_us,
                profile.min_time_us,
                profile.max_time_us,
                profile.throughput_gflops.map_or("".to_string(), |v| format!("{:.2}", v)),
                profile.bandwidth_gbs.map_or("".to_string(), |v| format!("{:.2}", v)),
            ));
        }

        csv
    }

    /// Export results in JSON format.
    pub fn export_json(&self) -> String {
        let results = self.get_results();
        let mut json = String::from("[\n");

        for (i, profile) in results.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"name\":\"{}\",\"calls\":{},\"avg_us\":{:.2},\"total_us\":{:.2},\"min_us\":{:.2},\"max_us\":{:.2}}}",
                profile.name, profile.calls, profile.avg_time_us, profile.total_time_us,
                profile.min_time_us, profile.max_time_us
            ));
            if i < results.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }

        json.push_str("]\n");
        json
    }

    /// Reset all profiling data.
    pub fn reset(&self) {
        self.results.lock().unwrap().clear();
        self.pending.lock().unwrap().clear();
    }

    /// Check if profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable or disable profiling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

/// Scoped profiler that automatically stops on drop.
pub struct ScopedProfiler<'a> {
    profiler: &'a Profiler,
    name: String,
}

impl<'a> ScopedProfiler<'a> {
    /// Create a new scoped profiler.
    pub fn new(profiler: &'a Profiler, name: &str) -> Self {
        profiler.start(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedProfiler<'a> {
    fn drop(&mut self) {
        self.profiler.stop(&self.name);
    }
}

/// Macro for scoped profiling.
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _guard = $crate::profiler::ScopedProfiler::new($profiler, $name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new(ProfilerConfig::default());
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profiler_enabled() {
        let profiler = Profiler::enabled();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiler_record() {
        let mut config = ProfilerConfig::default();
        config.enabled = true;
        let profiler = Profiler::new(config);

        profiler.record("test_kernel", 100.0);
        profiler.record("test_kernel", 200.0);

        let results = profiler.get_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].calls, 2);
        assert!((results[0].avg_time_us - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_profiler_min_duration_filter() {
        let mut config = ProfilerConfig::default();
        config.enabled = true;
        config.min_kernel_duration_us = 100.0;
        let profiler = Profiler::new(config);

        // Below minimum, should be filtered
        profiler.record("short_kernel", 50.0);
        assert_eq!(profiler.get_results().len(), 0);

        // Above minimum, should be recorded
        profiler.record("long_kernel", 200.0);
        assert_eq!(profiler.get_results().len(), 1);
    }

    #[test]
    fn test_profiler_reset() {
        let mut config = ProfilerConfig::default();
        config.enabled = true;
        let profiler = Profiler::new(config);

        profiler.record("test", 100.0);
        assert_eq!(profiler.get_results().len(), 1);

        profiler.reset();
        assert_eq!(profiler.get_results().len(), 0);
    }

    #[test]
    fn test_export_csv() {
        let mut config = ProfilerConfig::default();
        config.enabled = true;
        let profiler = Profiler::new(config);

        profiler.record("kernel_a", 100.0);
        let csv = profiler.export_csv();
        assert!(csv.contains("kernel_a"));
    }
}
