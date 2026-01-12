//! Computation graph for reverse-mode automatic differentiation.
//!
//! The graph records operations during the forward pass and enables
//! gradient computation during the backward pass.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use smallvec::SmallVec;

use ferrum_core::{Result, Tensor};

use crate::function::Function;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Generate a new unique node ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the computation graph.
///
/// Each node represents an operation that was recorded during the forward pass.
/// During backward, it computes gradients for its inputs given the gradient
/// of its output.
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// The function that was applied (stores backward logic).
    pub function: Box<dyn Function>,
    /// IDs of input nodes (for topological sort).
    pub inputs: SmallVec<[NodeId; 4]>,
    /// Number of times this node's output is used (for gradient accumulation).
    pub num_outputs: usize,
    /// Saved tensors needed for backward computation.
    pub saved_tensors: Vec<Tensor>,
}

impl Node {
    /// Create a new node.
    pub fn new(
        function: Box<dyn Function>,
        inputs: SmallVec<[NodeId; 4]>,
        saved_tensors: Vec<Tensor>,
    ) -> Self {
        Self {
            id: NodeId::new(),
            function,
            inputs,
            num_outputs: 1,
            saved_tensors,
        }
    }

    /// Compute gradients for inputs given output gradient.
    pub fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        self.function.backward(&self.saved_tensors, grad_output)
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("function", &self.function.name())
            .field("inputs", &self.inputs)
            .field("num_outputs", &self.num_outputs)
            .finish()
    }
}

/// Dynamic computation graph.
///
/// Records operations during forward pass and enables reverse-mode
/// automatic differentiation during backward pass.
///
/// ## Thread Safety
///
/// The graph uses internal locking and can be safely shared between threads.
/// However, the typical pattern is to use one graph per thread/forward pass.
pub struct ComputationGraph {
    /// All nodes in the graph.
    nodes: RwLock<HashMap<NodeId, Node>>,
    /// Mapping from tensor ID to the node that produced it.
    tensor_to_node: RwLock<HashMap<u64, NodeId>>,
    /// Counter for tensor IDs.
    tensor_counter: AtomicU64,
}

impl ComputationGraph {
    /// Create a new empty computation graph.
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            tensor_to_node: RwLock::new(HashMap::new()),
            tensor_counter: AtomicU64::new(0),
        }
    }

    /// Record an operation in the graph.
    ///
    /// # Arguments
    ///
    /// * `function` - The differentiable function that was applied
    /// * `inputs` - Tensor IDs of the inputs
    /// * `output_tensor_id` - Tensor ID of the output (from the result tensor)
    /// * `saved_tensors` - Tensors saved for backward computation
    pub fn record(
        &self,
        function: Box<dyn Function>,
        input_tensor_ids: &[u64],
        output_tensor_id: u64,
        saved_tensors: Vec<Tensor>,
    ) {
        // Find node IDs for inputs (if they exist in the graph)
        let input_nodes: SmallVec<[NodeId; 4]> = {
            let tensor_to_node = self.tensor_to_node.read();
            input_tensor_ids
                .iter()
                .filter_map(|id| tensor_to_node.get(id).copied())
                .collect()
        };

        // Create the new node
        let node = Node::new(function, input_nodes.clone(), saved_tensors);
        let node_id = node.id;

        // Update output counts for input nodes
        {
            let mut nodes = self.nodes.write();
            for &input_id in &input_nodes {
                if let Some(input_node) = nodes.get_mut(&input_id) {
                    input_node.num_outputs += 1;
                }
            }
            nodes.insert(node_id, node);
        }

        // Map the output tensor ID to this node
        {
            let mut tensor_to_node = self.tensor_to_node.write();
            tensor_to_node.insert(output_tensor_id, node_id);
        }
    }

    /// Get the node ID for a tensor.
    pub fn get_node_for_tensor(&self, tensor_id: u64) -> Option<NodeId> {
        self.tensor_to_node.read().get(&tensor_id).copied()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        // We need to clone because we can't return a reference with a lock guard
        // In practice, we'd use a different approach for production
        let nodes = self.nodes.read();
        nodes.get(&id).map(|n| Node {
            id: n.id,
            function: n.function.clone_box(),
            inputs: n.inputs.clone(),
            num_outputs: n.num_outputs,
            saved_tensors: n.saved_tensors.clone(),
        })
    }

    /// Get all nodes (for topological traversal).
    pub fn nodes(&self) -> Vec<Node> {
        let nodes = self.nodes.read();
        nodes
            .values()
            .map(|n| Node {
                id: n.id,
                function: n.function.clone_box(),
                inputs: n.inputs.clone(),
                num_outputs: n.num_outputs,
                saved_tensors: n.saved_tensors.clone(),
            })
            .collect()
    }

    /// Clear the graph (used after backward pass or between iterations).
    pub fn clear(&self) {
        self.nodes.write().clear();
        self.tensor_to_node.write().clear();
    }

    /// Get the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ComputationGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationGraph")
            .field("num_nodes", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFunction;

    impl Function for DummyFunction {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
            Ok(vec![Some(grad_output.clone())])
        }

        fn clone_box(&self) -> Box<dyn Function> {
            Box::new(DummyFunction)
        }
    }

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_record_operation() {
        let graph = ComputationGraph::new();

        let output_tensor_id = 42;
        graph.record(Box::new(DummyFunction), &[], output_tensor_id, vec![]);

        assert_eq!(graph.len(), 1);
        assert!(graph.get_node_for_tensor(output_tensor_id).is_some());
    }
}
