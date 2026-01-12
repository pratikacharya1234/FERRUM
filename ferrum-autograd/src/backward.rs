//! Backward pass implementation for automatic differentiation.
//!
//! This module implements the core backward algorithm that traverses
//! the computation graph in reverse topological order.

use std::collections::{HashMap, HashSet};

use ferrum_core::{Result, Tensor};

use crate::graph::{ComputationGraph, Node, NodeId};

/// Compute gradients via reverse-mode automatic differentiation.
///
/// This function implements backpropagation by:
/// 1. Starting from the output with gradient = 1
/// 2. Traversing nodes in reverse topological order
/// 3. Accumulating gradients for nodes with multiple outputs
/// 4. Storing gradients in tensors that require them
///
/// # Arguments
///
/// * `output` - The scalar loss tensor
/// * `graph` - The computation graph recorded during forward pass
///
/// # Algorithm
///
/// ```text
/// grad_table[output] = 1.0
/// for node in reverse_topological_order(graph):
///     grad_output = grad_table[node.output]
///     grad_inputs = node.backward(grad_output)
///     for (input, grad_input) in zip(node.inputs, grad_inputs):
///         grad_table[input] += grad_input
/// ```
pub fn backward(output: &Tensor, graph: &ComputationGraph) -> Result<()> {
    // Disable autograd during backward pass to prevent recording gradient operations
    let _guard = ferrum_core::autograd_ops::NoGradGuard::new();

    // Verify output is scalar
    if output.numel() != 1 {
        return Err(ferrum_core::FerrumError::InvalidShape {
            message: format!(
                "backward() expects scalar output, got shape {:?}",
                output.shape()
            ),
        });
    }

    // Get all nodes and perform topological sort
    let nodes = graph.nodes();
    
    if nodes.is_empty() {
        return Ok(());
    }

    // Build reverse topological order
    let sorted_nodes = topological_sort(&nodes)?;

    // Initialize gradient accumulator
    // Maps node ID -> accumulated gradient
    let mut grad_table: HashMap<NodeId, Tensor> = HashMap::new();

    // Seed with output gradient = 1
    if let Some(last_node) = sorted_nodes.last() {
        let seed = Tensor::ones(output.shape_obj().clone(), output.dtype(), output.device());
        grad_table.insert(last_node.id, seed);
    }

    // Traverse in reverse order
    for node in sorted_nodes.into_iter().rev() {
        // Get accumulated gradient for this node's output
        let grad_output = match grad_table.get(&node.id) {
            Some(g) => g.clone(),
            None => continue, // No gradient to propagate
        };

        // Compute gradients for inputs
        let input_grads = node.backward(&grad_output)?;

        // Accumulate gradients for each input
        for (i, grad) in input_grads.into_iter().enumerate() {
            if let Some(grad) = grad {
                // Set/accumulate gradient on the saved tensor (if it requires grad)
                if i < node.saved_tensors.len() {
                    let input_tensor = &node.saved_tensors[i];
                    
                    if input_tensor.requires_grad() {
                        // Accumulate gradient on the tensor itself
                        if let Some(existing_grad) = input_tensor.grad() {
                            // Add to existing gradient
                            if let Ok(sum) = existing_grad.add(&grad) {
                                input_tensor.set_grad(Some(sum));
                            }
                        } else {
                            // Set initial gradient
                            input_tensor.set_grad(Some(grad.clone()));
                        }
                    }
                }

                // Also propagate through computation graph for non-leaf tensors
                if i < node.inputs.len() {
                    let input_id = node.inputs[i];
                    grad_table
                        .entry(input_id)
                        .and_modify(|existing| {
                            if let Ok(sum) = existing.add(&grad) {
                                *existing = sum;
                            }
                        })
                        .or_insert(grad);
                }
            }
        }
    }

    Ok(())
}

/// Topological sort of computation graph nodes.
///
/// Returns nodes in dependency order (inputs before outputs).
fn topological_sort(nodes: &[Node]) -> Result<Vec<Node>> {
    let mut sorted = Vec::with_capacity(nodes.len());
    let mut visited = HashSet::new();
    let mut temp_visited = HashSet::new();

    // Build node lookup
    let node_map: HashMap<NodeId, &Node> = nodes.iter().map(|n| (n.id, n)).collect();

    fn visit(
        node_id: NodeId,
        node_map: &HashMap<NodeId, &Node>,
        visited: &mut HashSet<NodeId>,
        temp_visited: &mut HashSet<NodeId>,
        sorted: &mut Vec<Node>,
    ) -> Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        if temp_visited.contains(&node_id) {
            return Err(ferrum_core::FerrumError::AutogradError {
                message: "Cycle detected in computation graph".to_string(),
            });
        }

        temp_visited.insert(node_id);

        if let Some(&node) = node_map.get(&node_id) {
            for &input_id in &node.inputs {
                visit(input_id, node_map, visited, temp_visited, sorted)?;
            }

            temp_visited.remove(&node_id);
            visited.insert(node_id);
            sorted.push(Node {
                id: node.id,
                function: node.function.clone_box(),
                inputs: node.inputs.clone(),
                num_outputs: node.num_outputs,
                saved_tensors: node.saved_tensors.clone(),
            });
        }

        Ok(())
    }

    // Visit all nodes
    for node in nodes {
        visit(
            node.id,
            &node_map,
            &mut visited,
            &mut temp_visited,
            &mut sorted,
        )?;
    }

    Ok(sorted)
}

/// Gradient checker for testing backward implementations.
///
/// Uses finite differences to numerically verify analytical gradients.
pub struct GradChecker {
    /// Step size for finite differences.
    epsilon: f64,
    /// Relative tolerance for comparison.
    rtol: f64,
    /// Absolute tolerance for comparison.
    atol: f64,
}

impl GradChecker {
    /// Create a new gradient checker with default tolerances.
    pub fn new() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
        }
    }

    /// Set the finite difference step size.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set the absolute tolerance.
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    /// Check gradient of a scalar function at a point.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a tensor and returns a scalar
    /// * `x` - Point to check gradient at
    /// * `analytical_grad` - The analytically computed gradient
    ///
    /// # Returns
    ///
    /// `true` if gradients match within tolerance, `false` otherwise.
    pub fn check<F>(&self, f: F, x: &Tensor, analytical_grad: &Tensor) -> Result<bool>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let x_data = x.to_vec::<f32>()?;
        let analytical_data = analytical_grad.to_vec::<f32>()?;
        let mut numerical_data = vec![0.0f32; x_data.len()];

        // Compute numerical gradient via central differences
        for i in 0..x_data.len() {
            // f(x + eps)
            let mut x_plus = x_data.clone();
            x_plus[i] += self.epsilon as f32;
            let x_plus_tensor = Tensor::from_slice(&x_plus, x.shape(), x.device())?;
            let f_plus = f(&x_plus_tensor)?.item()?;

            // f(x - eps)
            let mut x_minus = x_data.clone();
            x_minus[i] -= self.epsilon as f32;
            let x_minus_tensor = Tensor::from_slice(&x_minus, x.shape(), x.device())?;
            let f_minus = f(&x_minus_tensor)?.item()?;

            // Central difference
            numerical_data[i] = ((f_plus - f_minus) / (2.0 * self.epsilon)) as f32;
        }

        // Compare numerical and analytical gradients
        for (n, a) in numerical_data.iter().zip(analytical_data.iter()) {
            let diff = (n - a).abs();
            let tol = self.atol as f32 + self.rtol as f32 * a.abs();
            if diff > tol {
                eprintln!(
                    "Gradient mismatch: numerical={}, analytical={}, diff={}",
                    n, a, diff
                );
                return Ok(false);
            }
        }

        Ok(true)
    }
}

impl Default for GradChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::Device;

    #[test]
    fn test_grad_checker_quadratic() {
        // f(x) = x^2, f'(x) = 2x
        // Use larger tolerance for finite differences
        let checker = GradChecker::new().with_rtol(1e-2).with_atol(1e-3);

        let x = Tensor::from_slice(&[2.0f32], [1], Device::Cpu).unwrap();
        let grad = Tensor::from_slice(&[4.0f32], [1], Device::Cpu).unwrap(); // 2 * 2 = 4

        let result = checker
            .check(|t| t.pow(2.0)?.sum(), &x, &grad)
            .unwrap();

        assert!(result);
    }

    #[test]
    fn test_grad_checker_exp() {
        // f(x) = exp(x), f'(x) = exp(x)
        // Use larger tolerance for finite differences
        let checker = GradChecker::new().with_rtol(1e-2).with_atol(1e-3);

        let x = Tensor::from_slice(&[1.0f32], [1], Device::Cpu).unwrap();
        let grad = Tensor::from_slice(&[std::f32::consts::E], [1], Device::Cpu).unwrap();

        let result = checker.check(|t| t.exp()?.sum(), &x, &grad).unwrap();

        assert!(result);
    }
}
