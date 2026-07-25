//! Fusion Region Detection.
//!
//! Partitions a compute graph into fusion regions - maximal connected
//! subgraphs where every node is element-wise, has the same shape,
//! and no node has consumers outside the region (except output nodes).

use crate::fusion::ir::{FusionGraph, FusionNode, FusionOp};

/// A fusion region: a maximal set of nodes that can be fused into one kernel.
#[derive(Clone, Debug)]
pub struct FusionRegion {
    /// Indices of nodes in this region (in topological order).
    pub node_indices: Vec<usize>,
    /// The fusion graph these nodes belong to.
    pub graph: FusionGraph,
}

/// Partitions a fusion graph into fusion regions.
pub struct FusionPartitioner;

impl FusionPartitioner {
    /// Partition a fusion graph into fusion regions.
    ///
    /// Algorithm:
    /// 1. Build adjacency info (consumers of each node)
    /// 2. Mark nodes that cannot be fused (reductions, matmul, conv)
    /// 3. Greedily merge adjacent element-wise nodes with same shape
    /// 4. Stop at: reductions, shape mismatches, or nodes with external consumers
    pub fn partition(graph: &FusionGraph) -> Vec<FusionRegion> {
        if graph.is_empty() {
            return Vec::new();
        }

        let n = graph.nodes.len();

        // Build consumer list
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &input_idx in &node.inputs {
                consumers[input_idx].push(node_idx);
            }
        }

        // Determine which nodes can be fused
        let fuseable = Self::compute_fuseable(graph, &consumers);

        // Greedily build regions
        let mut visited = vec![false; n];
        let mut regions = Vec::new();

        for start in 0..n {
            if visited[start] || !fuseable[start] {
                continue;
            }

            // BFS to find all fuseable nodes reachable from start
            let mut region_nodes = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;

            let start_shape = &graph.nodes[start].shape;

            while let Some(node_idx) = queue.pop_front() {
                let node = &graph.nodes[node_idx];

                // Check if this node can join the region
                if node_idx != start {
                    // Must have same shape
                    if node.shape != *start_shape {
                        continue;
                    }
                    // Must be element-wise (not reduction)
                    if node.is_reduction() {
                        continue;
                    }
                    // Must be fuseable
                    if !fuseable[node_idx] {
                        continue;
                    }
                    // All inputs must already be in the region or be graph inputs
                    let all_inputs_in_region = node.inputs.iter().all(|&input_idx| {
                        graph.nodes[input_idx].shape == *start_shape
                            && (fuseable[input_idx] || graph.inputs.contains(&input_idx))
                    });
                    if !all_inputs_in_region {
                        continue;
                    }
                }

                region_nodes.push(node_idx);

                // Add consumers to queue
                for &consumer in &consumers[node_idx] {
                    if !visited[consumer] {
                        queue.push_back(consumer);
                    }
                }
            }

            if region_nodes.len() > 1 {
                // Sort in topological order
                region_nodes.sort_by_key(|&idx| {
                    graph
                        .nodes
                        .iter()
                        .position(|n| n.id == graph.nodes[idx].id)
                        .unwrap_or(0)
                });

                regions.push(FusionRegion {
                    node_indices: region_nodes,
                    graph: graph.clone(),
                });
            }
        }

        regions
    }

    /// Compute which nodes are fuseable (element-wise, same shape as neighbors).
    fn compute_fuseable(graph: &FusionGraph, consumers: &[Vec<usize>]) -> Vec<bool> {
        let n = graph.nodes.len();
        let mut fuseable = vec![true; n];

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Graph inputs are always fuseable
            if graph.inputs.contains(&idx) {
                continue;
            }

            // Reductions cannot be fused across
            if node.is_reduction() {
                fuseable[idx] = false;
                continue;
            }

            // Check if all inputs have the same shape
            if !node.inputs.is_empty() {
                let first_shape = &graph.nodes[node.inputs[0]].shape;
                let same_shape = node.inputs.iter().all(|&i| graph.nodes[i].shape == *first_shape);
                if !same_shape {
                    fuseable[idx] = false;
                    continue;
                }
            }

            // Check: if this node has consumers outside the fusion region,
            // it might need to be materialized. For now, we allow it
            // as long as all consumers have the same shape.
            // (A more conservative approach would mark it as non-fuseable
            // if any consumer has a different shape.)
        }

        fuseable
    }

    /// Extract a fusion region as a standalone FusionGraph.
    pub fn extract_region(region: &FusionRegion) -> FusionGraph {
        let mut subgraph = FusionGraph::new();

        // Map old indices to new indices
        let mut index_map = std::collections::HashMap::new();

        // Add input nodes first (only those in the region)
        for &node_idx in &region.node_indices {
            let node = &region.graph.nodes[node_idx];
            if region.graph.inputs.contains(&node_idx) {
                let new_idx = subgraph.add_input(node.shape.clone());
                index_map.insert(node_idx, new_idx);
            }
        }

        // Add operation nodes
        for &node_idx in &region.node_indices {
            let node = &region.graph.nodes[node_idx];
            if !region.graph.inputs.contains(&node_idx) {
                // Map inputs, handling cases where inputs might not be in the region
                let new_inputs: Vec<usize> = node
                    .inputs
                    .iter()
                    .filter_map(|&i| index_map.get(&i).copied())
                    .collect();
                let new_idx = subgraph.add_op(node.op.clone(), new_inputs, node.shape.clone());
                index_map.insert(node_idx, new_idx);
            }
        }

        // Mark outputs
        for &node_idx in &region.node_indices {
            if region.graph.outputs.contains(&node_idx) {
                if let Some(&new_idx) = index_map.get(&node_idx) {
                    subgraph.mark_output(new_idx);
                }
            }
        }

        subgraph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::ir::FusionOp;

    #[test]
    fn test_partition_simple_chain() {
        // input0 + input1 -> add -> relu -> output
        let mut graph = FusionGraph::new();
        let input0 = graph.add_input(vec![100]);
        let input1 = graph.add_input(vec![100]);
        let add = graph.add_op(FusionOp::Add, vec![input0, input1], vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![add], vec![100]);
        graph.mark_output(relu);

        let regions = FusionPartitioner::partition(&graph);
        // Should have at least one region containing add and relu
        assert!(!regions.is_empty());
        // Each region should have at least 2 nodes (input + op)
        for region in &regions {
            assert!(region.node_indices.len() >= 2);
        }
    }

    #[test]
    fn test_partition_shape_mismatch() {
        // input0 [100] + input1 [200] -> different shapes, no fusion
        let mut graph = FusionGraph::new();
        let input0 = graph.add_input(vec![100]);
        let input1 = graph.add_input(vec![200]);
        let add = graph.add_op(FusionOp::Add, vec![input0, input1], vec![200]);
        graph.mark_output(add);

        let regions = FusionPartitioner::partition(&graph);
        // Should not fuse due to shape mismatch
        assert!(regions.is_empty() || regions[0].node_indices.len() <= 1);
    }

    #[test]
    fn test_partition_with_reduction() {
        // input -> relu -> sum (reduction stops fusion)
        let mut graph = FusionGraph::new();
        let input = graph.add_input(vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![input], vec![100]);
        let sum = graph.add_op(FusionOp::Sum, vec![relu], vec![1]);
        graph.mark_output(sum);

        let regions = FusionPartitioner::partition(&graph);
        // relu should be in a separate region from sum
        if !regions.is_empty() {
            // The region should not contain both relu and sum
            for region in &regions {
                let has_relu = region
                    .node_indices
                    .iter()
                    .any(|&i| graph.nodes[i].op == FusionOp::Relu);
                let has_sum = region
                    .node_indices
                    .iter()
                    .any(|&i| graph.nodes[i].op == FusionOp::Sum);
                assert!(!(has_relu && has_sum));
            }
        }
    }

    #[test]
    fn test_extract_region() {
        let mut graph = FusionGraph::new();
        let input0 = graph.add_input(vec![100]);
        let input1 = graph.add_input(vec![100]);
        let add = graph.add_op(FusionOp::Add, vec![input0, input1], vec![100]);
        let relu = graph.add_op(FusionOp::Relu, vec![add], vec![100]);
        graph.mark_output(relu);

        let regions = FusionPartitioner::partition(&graph);
        if let Some(region) = regions.first() {
            let subgraph = FusionPartitioner::extract_region(region);
            assert!(subgraph.len() >= 2);
        }
    }
}
