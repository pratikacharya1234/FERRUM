//! Fusion Graph Intermediate Representation.
//!
//! The IR represents a sequence of element-wise operations that can be
//! fused into a single CUDA kernel. Each node is an operation with
//! inputs (from other nodes or graph inputs) and an output shape.

use std::fmt;

/// Operations that can be fused into a single kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum FusionOp {
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    // Unary ops
    Relu,
    Gelu,
    Silu,
    Elu,
    LeakyRelu(f32),
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Reciprocal,
    Abs,
    Neg,
    // Reductions (cannot fuse across these)
    Sum,
    Mean,
    MaxReduce,
    MinReduce,
}

impl FusionOp {
    /// Whether this operation is a reduction.
    pub fn is_reduction(&self) -> bool {
        matches!(
            self,
            FusionOp::Sum | FusionOp::Mean | FusionOp::MaxReduce | FusionOp::MinReduce
        )
    }

    /// Whether this is a binary operation.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            FusionOp::Add
                | FusionOp::Sub
                | FusionOp::Mul
                | FusionOp::Div
                | FusionOp::Pow
                | FusionOp::Max
                | FusionOp::Min
        )
    }

    /// Get the CUDA C expression for this operation.
    /// `a` is the first operand, `b` is the second (for binary ops).
    pub fn cuda_expr(&self, a: &str, b: &str) -> String {
        match self {
            FusionOp::Add => format!("({a} + {b})"),
            FusionOp::Sub => format!("({a} - {b})"),
            FusionOp::Mul => format!("({a} * {b})"),
            FusionOp::Div => format!("({a} / {b})"),
            FusionOp::Pow => format!("powf({a}, {b})"),
            FusionOp::Max => format!("fmaxf({a}, {b})"),
            FusionOp::Min => format!("fminf({a}, {b})"),
            FusionOp::Relu => format!("fmaxf({a}, 0.0f)"),
            FusionOp::Gelu => {
                format!(
                    "(0.5f * {a} * (1.0f + tanhf(0.7978845608f * ({a} + 0.044715f * {a} * {a} * {a}))))"
                )
            }
            FusionOp::Silu => format!("({a} / (1.0f + expf(-{a})))"),
            FusionOp::Elu => format!(
                "(({a}) > 0.0f ? {a} : (expf({a}) - 1.0f))"
            ),
            FusionOp::LeakyRelu(alpha) => {
                format!("(({a}) > 0.0f ? {a} : ({alpha}f * {a}))")
            }
            FusionOp::Sigmoid => format!("(1.0f / (1.0f + expf(-{a})))"),
            FusionOp::Tanh => format!("tanhf({a})"),
            FusionOp::Exp => format!("expf({a})"),
            FusionOp::Log => format!("logf({a})"),
            FusionOp::Sqrt => format!("sqrtf({a})"),
            FusionOp::Reciprocal => format!("(1.0f / {a})"),
            FusionOp::Abs => format!("fabsf({a})"),
            FusionOp::Neg => format!("(-{a})"),
            FusionOp::Sum | FusionOp::Mean | FusionOp::MaxReduce | FusionOp::MinReduce => {
                format!("{a}")
            }
        }
    }
}

impl fmt::Display for FusionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionOp::Add => write!(f, "add"),
            FusionOp::Sub => write!(f, "sub"),
            FusionOp::Mul => write!(f, "mul"),
            FusionOp::Div => write!(f, "div"),
            FusionOp::Pow => write!(f, "pow"),
            FusionOp::Max => write!(f, "max"),
            FusionOp::Min => write!(f, "min"),
            FusionOp::Relu => write!(f, "relu"),
            FusionOp::Gelu => write!(f, "gelu"),
            FusionOp::Silu => write!(f, "silu"),
            FusionOp::Elu => write!(f, "elu"),
            FusionOp::LeakyRelu(a) => write!(f, "leaky_relu({a})"),
            FusionOp::Sigmoid => write!(f, "sigmoid"),
            FusionOp::Tanh => write!(f, "tanh"),
            FusionOp::Exp => write!(f, "exp"),
            FusionOp::Log => write!(f, "log"),
            FusionOp::Sqrt => write!(f, "sqrt"),
            FusionOp::Reciprocal => write!(f, "reciprocal"),
            FusionOp::Abs => write!(f, "abs"),
            FusionOp::Neg => write!(f, "neg"),
            FusionOp::Sum => write!(f, "sum"),
            FusionOp::Mean => write!(f, "mean"),
            FusionOp::MaxReduce => write!(f, "max_reduce"),
            FusionOp::MinReduce => write!(f, "min_reduce"),
        }
    }
}

/// A single node in the fusion graph.
#[derive(Clone, Debug)]
pub struct FusionNode {
    /// The operation to perform.
    pub op: FusionOp,
    /// Indices into the graph's node list (inputs to this op).
    /// For unary ops: 1 input.
    /// For binary ops: 2 inputs.
    /// For graph inputs: empty (data comes from outside).
    pub inputs: Vec<usize>,
    /// Output shape (flattened to 1D for element-wise ops).
    pub shape: Vec<usize>,
    /// Unique node ID for debugging.
    pub id: usize,
}

impl FusionNode {
    /// Create a new fusion node.
    pub fn new(op: FusionOp, inputs: Vec<usize>, shape: Vec<usize>, id: usize) -> Self {
        Self {
            op,
            inputs,
            shape,
            id,
        }
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether this node is a reduction.
    pub fn is_reduction(&self) -> bool {
        self.op.is_reduction()
    }
}

/// A fusion graph representing a sequence of element-wise operations.
#[derive(Clone, Debug)]
pub struct FusionGraph {
    /// All nodes in the graph (topologically sorted).
    pub nodes: Vec<FusionNode>,
    /// Indices of graph input nodes (receive data from outside).
    pub inputs: Vec<usize>,
    /// Indices of graph output nodes (produce data for outside).
    pub outputs: Vec<usize>,
}

impl FusionGraph {
    /// Create an empty fusion graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a graph input node (no operation, just receives data).
    pub fn add_input(&mut self, shape: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(FusionNode::new(
            FusionOp::Add, // placeholder, won't be used
            vec![],
            shape,
            id,
        ));
        self.inputs.push(id);
        id
    }

    /// Add an operation node.
    pub fn add_op(&mut self, op: FusionOp, inputs: Vec<usize>, shape: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes
            .push(FusionNode::new(op, inputs, shape, id));
        id
    }

    /// Mark a node as an output.
    pub fn mark_output(&mut self, node_idx: usize) {
        if !self.outputs.contains(&node_idx) {
            self.outputs.push(node_idx);
        }
    }

    /// Get the total number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all unique input shapes.
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .map(|&i| self.nodes[i].shape.clone())
            .collect()
    }

    /// Check if all nodes have the same shape.
    pub fn homogeneous_shapes(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        let first = &self.nodes[0].shape;
        self.nodes.iter().all(|n| n.shape == *first)
    }

    /// Get the maximum number of inputs to any node.
    pub fn max_inputs(&self) -> usize {
        self.nodes.iter().map(|n| n.inputs.len()).max().unwrap_or(0)
    }
}

impl Default for FusionGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_op_is_reduction() {
        assert!(FusionOp::Sum.is_reduction());
        assert!(FusionOp::Mean.is_reduction());
        assert!(!FusionOp::Add.is_reduction());
        assert!(!FusionOp::Relu.is_reduction());
    }

    #[test]
    fn test_fusion_op_is_binary() {
        assert!(FusionOp::Add.is_binary());
        assert!(FusionOp::Mul.is_binary());
        assert!(!FusionOp::Relu.is_binary());
        assert!(!FusionOp::Exp.is_binary());
    }

    #[test]
    fn test_fusion_op_cuda_expr() {
        assert_eq!(FusionOp::Add.cuda_expr("a", "b"), "(a + b)");
        assert_eq!(FusionOp::Mul.cuda_expr("a", "b"), "(a * b)");
        assert_eq!(FusionOp::Relu.cuda_expr("a", ""), "fmaxf(a, 0.0f)");
    }

    #[test]
    fn test_fusion_graph_construction() {
        let mut graph = FusionGraph::new();
        let input0 = graph.add_input(vec![100]);
        let input1 = graph.add_input(vec![100]);
        let add = graph.add_op(FusionOp::Add, vec![input0, input1], vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![add], vec![100]);
        graph.mark_output(relu);

        assert_eq!(graph.len(), 4); // 2 inputs + add + relu
        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
        assert!(graph.homogeneous_shapes());
    }

    #[test]
    fn test_fusion_graph_numel() {
        let mut graph = FusionGraph::new();
        let input = graph.add_input(vec![32, 128, 768]);
        let relu = graph.add_op(FusionOp::Relu, vec![input], vec![32, 128, 768]);
        graph.mark_output(relu);

        assert_eq!(graph.nodes[relu].numel(), 32 * 128 * 768);
    }
}
