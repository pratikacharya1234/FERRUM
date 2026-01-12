//! Recurrent Neural Network layers.
//!
//! Implements RNN, LSTM, and GRU cells and layers.

use ferrum_core::{Device, DType, Result, Tensor};
use crate::module::Module;

/// RNN Cell - single step of RNN.
pub struct RNNCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
    nonlinearity: Nonlinearity,
}

/// Nonlinearity for RNN.
#[derive(Clone, Copy)]
pub enum Nonlinearity {
    Tanh,
    ReLU,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self::with_config(input_size, hidden_size, true, Nonlinearity::Tanh)
    }

    pub fn with_config(input_size: usize, hidden_size: usize, bias: bool, nonlinearity: Nonlinearity) -> Self {
        let k = 1.0 / (hidden_size as f64).sqrt();

        let weight_ih = Tensor::uniform([hidden_size, input_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);
        let weight_hh = Tensor::uniform([hidden_size, hidden_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);

        let (bias_ih, bias_hh) = if bias {
            (
                Some(Tensor::uniform([hidden_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
                Some(Tensor::uniform([hidden_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
            )
        } else {
            (None, None)
        };

        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size, nonlinearity }
    }

    /// Forward pass for a single timestep.
    pub fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        // h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
        let mut h = input.matmul(&self.weight_ih.t()?)?;
        
        if let Some(ref b) = self.bias_ih {
            h = h.add(b)?;
        }
        
        let hh = hidden.matmul(&self.weight_hh.t()?)?;
        h = h.add(&hh)?;
        
        if let Some(ref b) = self.bias_hh {
            h = h.add(b)?;
        }

        match self.nonlinearity {
            Nonlinearity::Tanh => h.tanh(),
            Nonlinearity::ReLU => h.relu(),
        }
    }

    pub fn hidden_size(&self) -> usize { self.hidden_size }
}

/// LSTM Cell - single step of LSTM.
pub struct LSTMCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self::with_bias(input_size, hidden_size, true)
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 4 * hidden_size;

        let weight_ih = Tensor::uniform([gate_size, input_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);
        let weight_hh = Tensor::uniform([gate_size, hidden_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);

        let (bias_ih, bias_hh) = if bias {
            (
                Some(Tensor::uniform([gate_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
                Some(Tensor::uniform([gate_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
            )
        } else {
            (None, None)
        };

        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size }
    }

    /// Forward pass for single timestep.
    /// Returns (h', c')
    pub fn forward(&self, input: &Tensor, hx: (&Tensor, &Tensor)) -> Result<(Tensor, Tensor)> {
        let (h, c) = hx;
        
        // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
        let mut gates = input.matmul(&self.weight_ih.t()?)?;
        if let Some(ref b) = self.bias_ih { gates = gates.add(b)?; }
        let hh = h.matmul(&self.weight_hh.t()?)?;
        gates = gates.add(&hh)?;
        if let Some(ref b) = self.bias_hh { gates = gates.add(b)?; }

        // Split into i, f, g, o gates
        let gates_data = gates.to_vec::<f32>()?;
        let batch = input.shape()[0];
        let hs = self.hidden_size;

        let mut i_gate = vec![0.0f32; batch * hs];
        let mut f_gate = vec![0.0f32; batch * hs];
        let mut g_gate = vec![0.0f32; batch * hs];
        let mut o_gate = vec![0.0f32; batch * hs];

        for b_idx in 0..batch {
            for j in 0..hs {
                let base = b_idx * 4 * hs;
                i_gate[b_idx * hs + j] = sigmoid(gates_data[base + j]);
                f_gate[b_idx * hs + j] = sigmoid(gates_data[base + hs + j]);
                g_gate[b_idx * hs + j] = gates_data[base + 2 * hs + j].tanh();
                o_gate[b_idx * hs + j] = sigmoid(gates_data[base + 3 * hs + j]);
            }
        }

        let c_data = c.to_vec::<f32>()?;
        let mut c_new = vec![0.0f32; batch * hs];
        let mut h_new = vec![0.0f32; batch * hs];

        for i in 0..batch * hs {
            c_new[i] = f_gate[i] * c_data[i] + i_gate[i] * g_gate[i];
            h_new[i] = o_gate[i] * c_new[i].tanh();
        }

        let h_out = Tensor::from_slice(&h_new, (batch, hs), input.device())?;
        let c_out = Tensor::from_slice(&c_new, (batch, hs), input.device())?;

        Ok((h_out, c_out))
    }

    pub fn hidden_size(&self) -> usize { self.hidden_size }
}

/// GRU Cell - single step of GRU.
pub struct GRUCell {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    hidden_size: usize,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self::with_bias(input_size, hidden_size, true)
    }

    pub fn with_bias(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        let k = 1.0 / (hidden_size as f64).sqrt();
        let gate_size = 3 * hidden_size;

        let weight_ih = Tensor::uniform([gate_size, input_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);
        let weight_hh = Tensor::uniform([gate_size, hidden_size], -k, k, DType::F32, Device::Cpu)
            .with_requires_grad(true);

        let (bias_ih, bias_hh) = if bias {
            (
                Some(Tensor::uniform([gate_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
                Some(Tensor::uniform([gate_size], -k, k, DType::F32, Device::Cpu).with_requires_grad(true)),
            )
        } else {
            (None, None)
        };

        Self { weight_ih, weight_hh, bias_ih, bias_hh, hidden_size }
    }

    /// Forward pass for single timestep.
    pub fn forward(&self, input: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let hs = self.hidden_size;
        let batch = input.shape()[0];

        // Compute gates
        let mut gates_i = input.matmul(&self.weight_ih.t()?)?;
        if let Some(ref b) = self.bias_ih { gates_i = gates_i.add(b)?; }
        
        let mut gates_h = hidden.matmul(&self.weight_hh.t()?)?;
        if let Some(ref b) = self.bias_hh { gates_h = gates_h.add(b)?; }

        let gi = gates_i.to_vec::<f32>()?;
        let gh = gates_h.to_vec::<f32>()?;
        let h_data = hidden.to_vec::<f32>()?;

        let mut h_new = vec![0.0f32; batch * hs];

        for b_idx in 0..batch {
            for j in 0..hs {
                let base_i = b_idx * 3 * hs;
                let base_h = b_idx * 3 * hs;
                
                let r = sigmoid(gi[base_i + j] + gh[base_h + j]);
                let z = sigmoid(gi[base_i + hs + j] + gh[base_h + hs + j]);
                let n = (gi[base_i + 2 * hs + j] + r * gh[base_h + 2 * hs + j]).tanh();
                
                h_new[b_idx * hs + j] = (1.0 - z) * n + z * h_data[b_idx * hs + j];
            }
        }

        Tensor::from_slice(&h_new, (batch, hs), input.device())
    }

    pub fn hidden_size(&self) -> usize { self.hidden_size }
}

/// Full RNN layer for sequences.
pub struct RNN {
    cells: Vec<RNNCell>,
    num_layers: usize,
    hidden_size: usize,
    bidirectional: bool,
    training: bool,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::with_config(input_size, hidden_size, num_layers, true, false, Nonlinearity::Tanh)
    }

    pub fn with_config(
        input_size: usize, hidden_size: usize, num_layers: usize,
        bias: bool, bidirectional: bool, nonlinearity: Nonlinearity,
    ) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::new();

        for layer in 0..num_layers {
            for _ in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                cells.push(RNNCell::with_config(layer_input_size, hidden_size, bias, nonlinearity));
            }
        }

        Self { cells, num_layers, hidden_size, bidirectional, training: true }
    }

    /// Forward pass over sequence.
    /// input: [seq_len, batch, input_size]
    /// Returns: (output, h_n)
    pub fn forward_seq(&self, input: &Tensor, h_0: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let input_shape = input.shape();
        let seq_len = input_shape[0];
        let batch = input_shape[1];
        let num_directions = if self.bidirectional { 2 } else { 1 };

        // Initialize hidden state
        let h = if let Some(h0) = h_0 {
            h0.clone()
        } else {
            Tensor::zeros((self.num_layers * num_directions, batch, self.hidden_size), DType::F32, input.device())
        };

        // Process sequence (simplified - single layer, forward only for now)
        let mut outputs = Vec::new();
        let mut hidden = self.get_layer_hidden(&h, 0)?;

        for t in 0..seq_len {
            let x_t = self.get_timestep(input, t)?;
            hidden = self.cells[0].forward(&x_t, &hidden)?;
            outputs.push(hidden.clone());
        }

        // Stack outputs
        let output = self.stack_outputs(&outputs, seq_len, batch)?;
        
        Ok((output, hidden))
    }

    fn get_timestep(&self, input: &Tensor, t: usize) -> Result<Tensor> {
        let shape = input.shape();
        let batch = shape[1];
        let feat = shape[2];
        let data = input.to_vec::<f32>()?;
        let start = t * batch * feat;
        let slice = &data[start..start + batch * feat];
        Tensor::from_slice(slice, (batch, feat), input.device())
    }

    fn get_layer_hidden(&self, h: &Tensor, layer: usize) -> Result<Tensor> {
        let shape = h.shape();
        let batch = shape[1];
        let hs = shape[2];
        let data = h.to_vec::<f32>()?;
        let start = layer * batch * hs;
        let slice = &data[start..start + batch * hs];
        Tensor::from_slice(slice, (batch, hs), h.device())
    }

    fn stack_outputs(&self, outputs: &[Tensor], seq_len: usize, batch: usize) -> Result<Tensor> {
        let mut all_data = Vec::new();
        for out in outputs {
            all_data.extend(out.to_vec::<f32>()?);
        }
        Tensor::from_slice(&all_data, (seq_len, batch, self.hidden_size), outputs[0].device())
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.push(cell.weight_ih.clone());
            params.push(cell.weight_hh.clone());
            if let Some(ref b) = cell.bias_ih { params.push(b.clone()); }
            if let Some(ref b) = cell.bias_hh { params.push(b.clone()); }
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "RNN" }
}

/// Full LSTM layer for sequences.
pub struct LSTM {
    cells: Vec<LSTMCell>,
    num_layers: usize,
    hidden_size: usize,
    bidirectional: bool,
    training: bool,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::with_config(input_size, hidden_size, num_layers, true, false)
    }

    pub fn with_config(
        input_size: usize, hidden_size: usize, num_layers: usize,
        bias: bool, bidirectional: bool,
    ) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::new();

        for layer in 0..num_layers {
            for _ in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                cells.push(LSTMCell::with_bias(layer_input_size, hidden_size, bias));
            }
        }

        Self { cells, num_layers, hidden_size, bidirectional, training: true }
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let seq_len = shape[0];
        let batch = shape[1];

        let mut h = Tensor::zeros((batch, self.hidden_size), DType::F32, input.device());
        let mut c = Tensor::zeros((batch, self.hidden_size), DType::F32, input.device());

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = get_timestep(input, t)?;
            let (h_new, c_new) = self.cells[0].forward(&x_t, (&h, &c))?;
            h = h_new;
            c = c_new;
            outputs.push(h.clone());
        }

        stack_outputs(&outputs, seq_len, batch, self.hidden_size, input.device())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.push(cell.weight_ih.clone());
            params.push(cell.weight_hh.clone());
            if let Some(ref b) = cell.bias_ih { params.push(b.clone()); }
            if let Some(ref b) = cell.bias_hh { params.push(b.clone()); }
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "LSTM" }
}

/// Full GRU layer for sequences.
pub struct GRU {
    cells: Vec<GRUCell>,
    num_layers: usize,
    hidden_size: usize,
    bidirectional: bool,
    training: bool,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        Self::with_config(input_size, hidden_size, num_layers, true, false)
    }

    pub fn with_config(
        input_size: usize, hidden_size: usize, num_layers: usize,
        bias: bool, bidirectional: bool,
    ) -> Self {
        let num_directions = if bidirectional { 2 } else { 1 };
        let mut cells = Vec::new();

        for layer in 0..num_layers {
            for _ in 0..num_directions {
                let layer_input_size = if layer == 0 { input_size } else { hidden_size * num_directions };
                cells.push(GRUCell::with_bias(layer_input_size, hidden_size, bias));
            }
        }

        Self { cells, num_layers, hidden_size, bidirectional, training: true }
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let seq_len = shape[0];
        let batch = shape[1];

        let mut h = Tensor::zeros((batch, self.hidden_size), DType::F32, input.device());
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = get_timestep(input, t)?;
            h = self.cells[0].forward(&x_t, &h)?;
            outputs.push(h.clone());
        }

        stack_outputs(&outputs, seq_len, batch, self.hidden_size, input.device())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.cells {
            params.push(cell.weight_ih.clone());
            params.push(cell.weight_hh.clone());
            if let Some(ref b) = cell.bias_ih { params.push(b.clone()); }
            if let Some(ref b) = cell.bias_hh { params.push(b.clone()); }
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "GRU" }
}

// Helper functions
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn get_timestep(input: &Tensor, t: usize) -> Result<Tensor> {
    let shape = input.shape();
    let batch = shape[1];
    let feat = shape[2];
    let data = input.to_vec::<f32>()?;
    let start = t * batch * feat;
    let slice = &data[start..start + batch * feat];
    Tensor::from_slice(slice, (batch, feat), input.device())
}

fn stack_outputs(outputs: &[Tensor], seq_len: usize, batch: usize, hidden_size: usize, device: Device) -> Result<Tensor> {
    let mut all_data = Vec::new();
    for out in outputs {
        all_data.extend(out.to_vec::<f32>()?);
    }
    Tensor::from_slice(&all_data, (seq_len, batch, hidden_size), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_cell() {
        let cell = RNNCell::new(10, 20);
        let input = Tensor::randn((2, 10), DType::F32, Device::Cpu);
        let hidden = Tensor::zeros((2, 20), DType::F32, Device::Cpu);
        let output = cell.forward(&input, &hidden).unwrap();
        assert_eq!(output.shape(), &[2, 20]);
    }

    #[test]
    fn test_lstm_cell() {
        let cell = LSTMCell::new(10, 20);
        let input = Tensor::randn((2, 10), DType::F32, Device::Cpu);
        let h = Tensor::zeros((2, 20), DType::F32, Device::Cpu);
        let c = Tensor::zeros((2, 20), DType::F32, Device::Cpu);
        let (h_out, c_out) = cell.forward(&input, (&h, &c)).unwrap();
        assert_eq!(h_out.shape(), &[2, 20]);
        assert_eq!(c_out.shape(), &[2, 20]);
    }

    #[test]
    fn test_gru_cell() {
        let cell = GRUCell::new(10, 20);
        let input = Tensor::randn((2, 10), DType::F32, Device::Cpu);
        let hidden = Tensor::zeros((2, 20), DType::F32, Device::Cpu);
        let output = cell.forward(&input, &hidden).unwrap();
        assert_eq!(output.shape(), &[2, 20]);
    }

    #[test]
    fn test_lstm_sequence() {
        let lstm = LSTM::new(10, 20, 1);
        let input = Tensor::randn((5, 2, 10), DType::F32, Device::Cpu);
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), &[5, 2, 20]);
    }
}
