//! CUDA C Code Generation from Fusion Graphs.
//!
//! Generates optimized CUDA C code for fused element-wise operations.
//! The generated code uses:
//! - `const float* __restrict__` for inputs (helps compiler optimize)
//! - Register variables for intermediates (no memory writes between ops)
//! - Flattened 1D indexing for arbitrary shapes
//! - `blockDim.x = 256`, grid = `(n + 255) / 256`

use crate::fusion::ir::{FusionGraph, FusionOp};

/// Code generator for fusion graphs.
pub struct CodeGen;

impl CodeGen {
    /// Generate CUDA C code for a fusion graph.
    ///
    /// # Arguments
    /// * `graph` - The fusion graph to compile
    /// * `kernel_name` - Name for the generated kernel function
    ///
    /// # Returns
    /// Tuple of (CUDA C source code, kernel function name)
    pub fn generate(graph: &FusionGraph, kernel_name: &str) -> (String, String) {
        let mut source = String::with_capacity(4096);

        // Kernel signature
        source.push_str(&format!(
            "extern \"C\" __global__ void {}(\n",
            kernel_name
        ));

        // Input parameters
        for (i, &input_idx) in graph.inputs.iter().enumerate() {
            source.push_str(&format!(
                "    const float* __restrict__ input_{},\n",
                i
            ));
        }

        // Output parameters
        for (i, &output_idx) in graph.outputs.iter().enumerate() {
            source.push_str(&format!(
                "    float* __restrict__ output_{}",
                i
            ));
            if i < graph.outputs.len() - 1 || graph.max_inputs() > 0 {
                source.push(',');
            }
            source.push('\n');
        }

        // Total elements parameter
        source.push_str("    int n\n");
        source.push_str(") {\n");

        // Thread index computation
        source.push_str("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
        source.push_str("    if (idx >= n) return;\n\n");

        // Build variable name map
        let mut var_names = std::collections::HashMap::new();
        for (i, &input_idx) in graph.inputs.iter().enumerate() {
            var_names.insert(input_idx, format!("input_{}", i));
        }

        // Load inputs from memory (indexed by idx)
        for (i, &input_idx) in graph.inputs.iter().enumerate() {
            let var = format!("input_{}", i);
            source.push_str(&format!(
                "    float {} = input_{}[idx];\n",
                var, i
            ));
            var_names.insert(input_idx, var);
        }

        // Generate intermediate computations
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            if graph.inputs.contains(&node_idx) {
                continue;
            }

            let output_var = format!("tmp_{}", node_idx);

            match &node.op {
                op if op.is_binary() => {
                    // Binary operation
                    let input_a = &var_names[&node.inputs[0]];
                    let input_b = &var_names[&node.inputs[1]];
                    let expr = op.cuda_expr(input_a, input_b);
                    source.push_str(&format!("    float {} = {};\n", output_var, expr));
                }
                op if !op.is_reduction() => {
                    // Unary operation
                    let input = &var_names[&node.inputs[0]];
                    let expr = op.cuda_expr(input, "");
                    source.push_str(&format!("    float {} = {};\n", output_var, expr));
                }
                _ => {
                    // Reductions are handled separately (not fused)
                    source.push_str(&format!(
                        "    float {} = input_0[idx]; // reduction placeholder\n",
                        output_var
                    ));
                }
            }

            var_names.insert(node_idx, output_var);
        }

        // Write outputs
        source.push('\n');
        for (i, &output_idx) in graph.outputs.iter().enumerate() {
            let var = &var_names[&output_idx];
            source.push_str(&format!("    output_{}[idx] = {};\n", i, var));
        }

        source.push_str("}\n");

        (source, kernel_name.to_string())
    }

    /// Generate a launch configuration for a fusion kernel.
    pub fn launch_config(n: usize) -> (u32, u32) {
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        (grid_size, block_size)
    }

    /// Generate a simple binary operation kernel (for testing).
    pub fn generate_binary_op(
        op: &FusionOp,
        kernel_name: &str,
    ) -> (String, String) {
        let source = format!(
            r#"extern "C" __global__ void {kernel_name}(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float va = a[idx];
    float vb = b[idx];
    out[idx] = {};
}}"#,
            op.cuda_expr("va", "vb")
        );
        (source, kernel_name.to_string())
    }

    /// Generate a simple unary operation kernel (for testing).
    pub fn generate_unary_op(
        op: &FusionOp,
        kernel_name: &str,
    ) -> (String, String) {
        let source = format!(
            r#"extern "C" __global__ void {kernel_name}(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = input[idx];
    output[idx] = {};
}}"#,
            op.cuda_expr("x", "")
        );
        (source, kernel_name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::ir::FusionGraph;

    #[test]
    fn test_generate_simple_fusion() {
        let mut graph = FusionGraph::new();
        let input0 = graph.add_input(vec![100]);
        let input1 = graph.add_input(vec![100]);
        let add = graph.add_op(FusionOp::Add, vec![input0, input1], vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![add], vec![100]);
        graph.mark_output(relu);

        let (source, name) = CodeGen::generate(&graph, "fused_add_relu");
        assert_eq!(name, "fused_add_relu");
        assert!(source.contains("extern \"C\" __global__ void"));
        assert!(source.contains("input_0[idx]"));
        assert!(source.contains("input_1[idx]"));
        assert!(source.contains("output_0[idx]"));
    }

    #[test]
    fn test_generate_binary_op() {
        let (source, _) = CodeGen::generate_binary_op(&FusionOp::Add, "test_add");
        assert!(source.contains("(va + vb)"));
        assert!(source.contains("blockIdx.x"));
    }

    #[test]
    fn test_generate_unary_op() {
        let (source, _) = CodeGen::generate_unary_op(&FusionOp::Relu, "test_relu");
        assert!(source.contains("fmaxf(x, 0.0f)"));
    }

    #[test]
    fn test_launch_config() {
        let (grid, block) = CodeGen::launch_config(1000);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_generate_complex_fusion() {
        // a + b -> mul c -> relu -> output
        let mut graph = FusionGraph::new();
        let a = graph.add_input(vec![100]);
        let b = graph.add_input(vec![100]);
        let c = graph.add_input(vec![100]);
        let add = graph.add_op(FusionOp::Add, vec![a, b], vec![100]);
        let mul = graph.add_op(FusionOp::Mul, vec![add, c], vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![mul], vec![100]);
        graph.mark_output(relu);

        let (source, _) = CodeGen::generate(&graph, "fused_complex");
        // Should have 3 inputs and 1 output
        assert!(source.contains("const float* __restrict__ input_0"));
        assert!(source.contains("const float* __restrict__ input_1"));
        assert!(source.contains("const float* __restrict__ input_2"));
        assert!(source.contains("float* __restrict__ output_0"));
    }
}
