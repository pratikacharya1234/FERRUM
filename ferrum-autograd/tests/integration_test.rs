//! Integration tests for autograd functionality.

use ferrum_autograd::*;
use ferrum_core::{DType, Device, Tensor};

#[test]
fn test_computation_graph_creation() {
    let graph = ComputationGraph::new();
    assert!(graph.is_empty());
    assert_eq!(graph.len(), 0);
}

#[test]
fn test_add_backward_function() {
    use ferrum_autograd::function::AddBackward;
    use ferrum_core::Shape;

    let _a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
    let _b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], [3], Device::Cpu).unwrap();

    let grad_output = Tensor::ones([3], DType::F32, Device::Cpu);

    let add_back = AddBackward {
        a_shape: Shape::from([3]),
        b_shape: Shape::from([3]),
    };

    let grads = add_back.backward(&[], &grad_output).unwrap();

    assert_eq!(grads.len(), 2);
    assert!(grads[0].is_some());
    assert!(grads[1].is_some());

    // For addition, gradient flows through unchanged
    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();

    let grad_a_data = grad_a.to_vec::<f32>().unwrap();
    let grad_b_data = grad_b.to_vec::<f32>().unwrap();

    assert_eq!(grad_a_data, vec![1.0, 1.0, 1.0]);
    assert_eq!(grad_b_data, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_mul_backward_function() {
    use ferrum_autograd::function::MulBackward;

    let a = Tensor::from_slice(&[2.0f32, 3.0], [2], Device::Cpu).unwrap();
    let b = Tensor::from_slice(&[4.0f32, 5.0], [2], Device::Cpu).unwrap();

    let grad_output = Tensor::ones([2], DType::F32, Device::Cpu);

    let mul_back = MulBackward;
    let grads = mul_back
        .backward(&[a.clone(), b.clone()], &grad_output)
        .unwrap();

    assert_eq!(grads.len(), 2);

    // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    let grad_a = grads[0].as_ref().unwrap();
    let grad_b = grads[1].as_ref().unwrap();

    let grad_a_data = grad_a.to_vec::<f32>().unwrap();
    let grad_b_data = grad_b.to_vec::<f32>().unwrap();

    // grad_a should equal b (scaled by grad_output = 1)
    assert_eq!(grad_a_data, vec![4.0, 5.0]);
    // grad_b should equal a
    assert_eq!(grad_b_data, vec![2.0, 3.0]);
}

#[test]
fn test_relu_backward_function() {
    use ferrum_autograd::function::ReLUBackward;

    let input = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], [4], Device::Cpu).unwrap();

    let grad_output = Tensor::ones([4], DType::F32, Device::Cpu);

    let relu_back = ReLUBackward;
    let grads = relu_back.backward(&[input], &grad_output).unwrap();

    let grad = grads[0].as_ref().unwrap();
    let grad_data = grad.to_vec::<f32>().unwrap();

    // Gradient is 0 where input <= 0, 1 otherwise
    assert_eq!(grad_data, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_sigmoid_backward_function() {
    use ferrum_autograd::function::SigmoidBackward;

    // Sigmoid output (not input!)
    let sigmoid_out = Tensor::from_slice(&[0.5f32, 0.73106, 0.88080], [3], Device::Cpu).unwrap();

    let grad_output = Tensor::ones([3], DType::F32, Device::Cpu);

    let sigmoid_back = SigmoidBackward;
    let grads = sigmoid_back
        .backward(std::slice::from_ref(&sigmoid_out), &grad_output)
        .unwrap();

    let grad = grads[0].as_ref().unwrap();
    let grad_data = grad.to_vec::<f32>().unwrap();

    // ∂sigmoid/∂x = sigmoid * (1 - sigmoid)
    // At 0.5: 0.5 * 0.5 = 0.25
    assert!((grad_data[0] - 0.25).abs() < 1e-5);
}

#[test]
fn test_matmul_backward_function() {
    use ferrum_autograd::function::MatMulBackward;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2], Device::Cpu).unwrap();
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], [2, 2], Device::Cpu).unwrap();

    let grad_output = Tensor::ones([2, 2], DType::F32, Device::Cpu);

    let matmul_back = MatMulBackward;
    let grads = matmul_back
        .backward(&[a.clone(), b.clone()], &grad_output)
        .unwrap();

    assert_eq!(grads.len(), 2);
    assert!(grads[0].is_some());
    assert!(grads[1].is_some());

    // grad_a = grad_output @ b.T
    let grad_a = grads[0].as_ref().unwrap();
    assert_eq!(grad_a.shape(), &[2, 2]);

    // grad_b = a.T @ grad_output
    let grad_b = grads[1].as_ref().unwrap();
    assert_eq!(grad_b.shape(), &[2, 2]);
}

#[test]
fn test_power_backward_function() {
    use ferrum_autograd::function::PowBackward;

    let input = Tensor::from_slice(&[2.0f32, 3.0], [2], Device::Cpu).unwrap();
    let grad_output = Tensor::ones([2], DType::F32, Device::Cpu);

    // Test x^2, gradient should be 2x
    let pow_back = PowBackward { exponent: 2.0 };
    let grads = pow_back.backward(std::slice::from_ref(&input), &grad_output).unwrap();

    let grad = grads[0].as_ref().unwrap();
    let grad_data = grad.to_vec::<f32>().unwrap();

    // ∂(x^2)/∂x = 2x
    // At x=2: 2*2 = 4
    // At x=3: 2*3 = 6
    assert!((grad_data[0] - 4.0).abs() < 1e-5);
    assert!((grad_data[1] - 6.0).abs() < 1e-5);
}

#[test]
fn test_sum_backward_function() {
    use ferrum_autograd::function::SumBackward;
    use ferrum_core::Shape;

    let input_shape = Shape::from([2, 3]);
    let grad_output = Tensor::from_slice(&[1.0f32], [1], Device::Cpu).unwrap();

    let sum_back = SumBackward { input_shape };
    let grads = sum_back.backward(&[], &grad_output).unwrap();

    let grad = grads[0].as_ref().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);

    // Gradient of sum is broadcast 1 to all elements
    let grad_data = grad.to_vec::<f32>().unwrap();
    assert!(grad_data.iter().all(|&x| (x - 1.0).abs() < 1e-5));
}

#[test]
fn test_mean_backward_function() {
    use ferrum_autograd::function::MeanBackward;
    use ferrum_core::Shape;

    let input_shape = Shape::from([4]);
    let grad_output = Tensor::from_slice(&[1.0f32], [1], Device::Cpu).unwrap();

    let mean_back = MeanBackward { input_shape };
    let grads = mean_back.backward(&[], &grad_output).unwrap();

    let grad = grads[0].as_ref().unwrap();
    assert_eq!(grad.shape(), &[4]);

    // Gradient of mean is 1/n broadcast to all elements
    let grad_data = grad.to_vec::<f32>().unwrap();
    assert!(grad_data.iter().all(|&x| (x - 0.25).abs() < 1e-5));
}

#[test]
fn test_exp_backward_function() {
    use ferrum_autograd::function::ExpBackward;

    let input = Tensor::from_slice(&[0.0f32, 1.0], [2], Device::Cpu).unwrap();
    let exp_output = input.exp().unwrap();

    let grad_output = Tensor::ones([2], DType::F32, Device::Cpu);

    let exp_back = ExpBackward;
    let grads = exp_back
        .backward(std::slice::from_ref(&exp_output), &grad_output)
        .unwrap();

    let grad = grads[0].as_ref().unwrap();
    let grad_data = grad.to_vec::<f32>().unwrap();
    let exp_data = exp_output.to_vec::<f32>().unwrap();

    // ∂exp(x)/∂x = exp(x)
    for i in 0..2 {
        assert!((grad_data[i] - exp_data[i]).abs() < 1e-5);
    }
}

#[test]
fn test_log_backward_function() {
    use ferrum_autograd::function::LogBackward;

    let input = Tensor::from_slice(&[1.0f32, 2.0, 4.0], [3], Device::Cpu).unwrap();
    let grad_output = Tensor::ones([3], DType::F32, Device::Cpu);

    let log_back = LogBackward;
    let grads = log_back.backward(std::slice::from_ref(&input), &grad_output).unwrap();

    let grad = grads[0].as_ref().unwrap();
    let grad_data = grad.to_vec::<f32>().unwrap();

    // ∂log(x)/∂x = 1/x
    assert!((grad_data[0] - 1.0).abs() < 1e-5); // 1/1 = 1
    assert!((grad_data[1] - 0.5).abs() < 1e-5); // 1/2 = 0.5
    assert!((grad_data[2] - 0.25).abs() < 1e-5); // 1/4 = 0.25
}

#[test]
fn test_graph_record_and_retrieve() {
    use ferrum_autograd::function::AddBackward;
    use ferrum_core::Shape;

    let graph = ComputationGraph::new();

    let add_back = AddBackward {
        a_shape: Shape::from([2]),
        b_shape: Shape::from([2]),
    };

    let tensor_id = 42u64;
    graph.record(Box::new(add_back), &[1, 2], tensor_id, vec![]);

    assert_eq!(graph.len(), 1);
    assert!(graph.get_node_for_tensor(tensor_id).is_some());
}

#[test]
fn test_topological_sort() {
    // Create a simple computation graph: a + b = c
    use ferrum_autograd::function::AddBackward;
    use ferrum_core::Shape;

    let graph = ComputationGraph::new();

    let add_back = AddBackward {
        a_shape: Shape::from([2]),
        b_shape: Shape::from([2]),
    };

    graph.record(Box::new(add_back), &[], 1, vec![]);

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 1);
}
