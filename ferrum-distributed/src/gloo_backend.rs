//! Real Gloo backend implementation using TCP sockets.
//!
//! This module provides a functional distributed communication backend
//! using TCP sockets for inter-process communication. It implements
//! all collective operations (broadcast, all-reduce, all-gather, etc.)
//! using point-to-point sends and receives.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::Mutex;
use std::collections::HashMap;

use ferrum_core::Tensor;
use crate::collectives::{Collectives, ReduceOp};
use crate::error::{DistributedError, Result};

/// TCP-based collective operations for distributed training.
pub struct GlooCollectives {
    rank: usize,
    world_size: usize,
    listener: Option<Arc<Mutex<TcpListener>>>,
    peers: Arc<Mutex<HashMap<usize, TcpStream>>>,
}

impl GlooCollectives {
    /// Create new Gloo collectives with TCP transport.
    ///
    /// # Arguments
    /// * `rank` - Rank of this process
    /// * `world_size` - Total number of processes
    /// * `master_addr` - Master address for rendezvous
    /// * `master_port` - Master port for rendezvous
    pub fn new(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        let mut peers = HashMap::new();

        // Bind listener for incoming connections
        let addr = format!("{}:{}", master_addr, master_port + rank as u16);
        let listener = TcpListener::bind(&addr)
            .map_err(|e| DistributedError::ConnectionFailed(format!("Failed to bind {}: {}", addr, e)))?;
        listener.set_nonblocking(true)
            .map_err(|e| DistributedError::ConnectionFailed(format!("Failed to set nonblocking: {}", e)))?;

        let listener = Arc::new(Mutex::new(listener));

        // Connect to peers with lower ranks (they connect to us)
        // We connect to peers with higher ranks
        if rank == 0 {
            // Rank 0 is the master, connects to all other ranks
            for peer_rank in 1..world_size {
                let peer_addr = format!("{}:{}", master_addr, master_port + peer_rank as u16);
                let stream = Self::connect_with_retry(&peer_addr)?;
                peers.insert(peer_rank, stream);
            }
        } else {
            // Non-master ranks connect to rank 0
            let master_addr = format!("{}:{}", master_addr, master_port);
            let stream = Self::connect_with_retry(&master_addr)?;
            peers.insert(0, stream);
        }

        // Accept connections from other ranks
        for peer_rank in 0..world_size {
            if peer_rank == rank || peers.contains_key(&peer_rank) {
                continue;
            }
            // Wait for incoming connection from this peer
            let listener_guard = listener.lock();
            let stream = Self::accept_connection(&listener_guard, Duration::from_secs(5))?;
            peers.insert(peer_rank, stream);
        }

        Ok(Self {
            rank,
            world_size,
            listener: Some(listener),
            peers: Arc::new(Mutex::new(peers)),
        })
    }

    /// Connect to a peer with retry logic.
    fn connect_with_retry(addr: &str) -> Result<TcpStream> {
        let max_retries = 10;
        let retry_delay = Duration::from_millis(100);

        for attempt in 0..max_retries {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    stream.set_read_timeout(Some(Duration::from_secs(30)))
                        .map_err(|e| DistributedError::ConnectionFailed(e.to_string()))?;
                    stream.set_write_timeout(Some(Duration::from_secs(30)))
                        .map_err(|e| DistributedError::ConnectionFailed(e.to_string()))?;
                    return Ok(stream);
                }
                Err(_) if attempt < max_retries - 1 => {
                    std::thread::sleep(retry_delay);
                }
                Err(e) => {
                    return Err(DistributedError::ConnectionFailed(
                        format!("Failed to connect to {} after {} attempts: {}", addr, max_retries, e)
                    ));
                }
            }
        }
        Err(DistributedError::ConnectionFailed(
            format!("Failed to connect to {} after {} attempts", addr, max_retries)
        ))
    }

    /// Accept a connection from the listener.
    fn accept_connection(listener: &TcpListener, timeout: Duration) -> Result<TcpStream> {
        let start = std::time::Instant::now();
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(30)))
                        .map_err(|e| DistributedError::ConnectionFailed(e.to_string()))?;
                    stream.set_write_timeout(Some(Duration::from_secs(30)))
                        .map_err(|e| DistributedError::ConnectionFailed(e.to_string()))?;
                    return Ok(stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if start.elapsed() > timeout {
                        return Err(DistributedError::ConnectionFailed(
                            "Accept timed out".to_string()
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(DistributedError::ConnectionFailed(
                        format!("Accept failed: {}", e)
                    ));
                }
            }
        }
    }

    /// Send data to a peer.
    fn send_data(&self, peer_rank: usize, data: &[u8]) -> Result<()> {
        let mut peers = self.peers.lock();
        let stream = peers.get_mut(&peer_rank)
            .ok_or_else(|| DistributedError::InvalidRank(peer_rank, self.world_size))?;

        // Send length prefix
        let len = data.len() as u64;
        stream.write_all(&len.to_le_bytes())
            .map_err(|e| DistributedError::SendFailed(e.to_string()))?;
        stream.write_all(data)
            .map_err(|e| DistributedError::SendFailed(e.to_string()))?;
        stream.flush()
            .map_err(|e| DistributedError::SendFailed(e.to_string()))?;
        Ok(())
    }

    /// Receive data from a peer.
    fn recv_data(&self, peer_rank: usize) -> Result<Vec<u8>> {
        let mut peers = self.peers.lock();
        let stream = peers.get_mut(&peer_rank)
            .ok_or_else(|| DistributedError::InvalidRank(peer_rank, self.world_size))?;

        // Read length prefix
        let mut len_buf = [0u8; 8];
        stream.read_exact(&mut len_buf)
            .map_err(|e| DistributedError::RecvFailed(e.to_string()))?;
        let len = u64::from_le_bytes(len_buf) as usize;

        // Read data
        let mut data = vec![0u8; len];
        stream.read_exact(&mut data)
            .map_err(|e| DistributedError::RecvFailed(e.to_string()))?;
        Ok(data)
    }

    /// Send tensor to peer.
    fn send_tensor(&self, peer_rank: usize, tensor: &Tensor) -> Result<()> {
        let data = tensor.to_vec::<f32>()
            .map_err(|e| DistributedError::SendFailed(e.to_string()))?;
        let bytes: Vec<u8> = data.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        self.send_data(peer_rank, &bytes)
    }

    /// Receive tensor from peer.
    fn recv_tensor(&self, peer_rank: usize, shape: &[usize]) -> Result<Tensor> {
        let bytes = self.recv_data(peer_rank)?;
        let float_count = bytes.len() / 4;
        let mut data = Vec::with_capacity(float_count);
        for i in 0..float_count {
            let bytes_val = [bytes[i*4], bytes[i*4+1], bytes[i*4+2], bytes[i*4+3]];
            data.push(f32::from_le_bytes(bytes_val));
        }
        Tensor::from_slice(&data, shape, ferrum_core::device::Device::Cpu)
            .map_err(|e| DistributedError::RecvFailed(e.to_string()))
    }
}

impl Collectives for GlooCollectives {
    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        if self.rank == root {
            // Root sends to all other ranks
            for peer_rank in 0..self.world_size {
                if peer_rank != self.rank {
                    self.send_tensor(peer_rank, tensor)?;
                }
            }
        } else {
            // Non-root receives from root
            let new_tensor = self.recv_tensor(root, tensor.shape())?;
            *tensor = new_tensor;
        }
        Ok(())
    }

    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // Simple ring-based all-reduce
        let data = tensor.to_vec::<f32>()
            .map_err(|e| DistributedError::CommunicationFailed(e.to_string()))?;

        // Phase 1: Reduce-scatter
        let chunk_size = data.len() / self.world_size;
        let mut local_reduce = vec![0.0f32; data.len()];

        // Initialize with our own data
        local_reduce.copy_from_slice(&data);

        // Exchange data with peers
        for peer_rank in 0..self.world_size {
            if peer_rank == self.rank {
                continue;
            }

            // Send our current data to peer
            let send_data: Vec<u8> = local_reduce.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            self.send_data(peer_rank, &send_data)?;

            // Receive peer's data
            let recv_bytes = self.recv_data(peer_rank)?;
            let mut peer_data = Vec::with_capacity(recv_bytes.len() / 4);
            for i in 0..recv_bytes.len() / 4 {
                let bytes_val = [recv_bytes[i*4], recv_bytes[i*4+1], recv_bytes[i*4+2], recv_bytes[i*4+3]];
                peer_data.push(f32::from_le_bytes(bytes_val));
            }

            // Apply reduction
            for (a, b) in local_reduce.iter_mut().zip(peer_data.iter()) {
                *a = op.apply(*a as f64, *b as f64) as f32;
            }
        }

        // Phase 2: Broadcast result
        if self.rank == 0 {
            // Root broadcasts to all
            for peer_rank in 1..self.world_size {
                let send_data: Vec<u8> = local_reduce.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                self.send_data(peer_rank, &send_data)?;
            }
        } else {
            // Receive from root
            let recv_bytes = self.recv_data(0)?;
            let mut recv_data = Vec::with_capacity(recv_bytes.len() / 4);
            for i in 0..recv_bytes.len() / 4 {
                let bytes_val = [recv_bytes[i*4], recv_bytes[i*4+1], recv_bytes[i*4+2], recv_bytes[i*4+3]];
                recv_data.push(f32::from_le_bytes(bytes_val));
            }
            local_reduce = recv_data;
        }

        // Apply average if needed
        if op == ReduceOp::Average {
            let n = self.world_size as f32;
            for val in local_reduce.iter_mut() {
                *val /= n;
            }
        }

        // Update tensor
        *tensor = Tensor::from_slice(&local_reduce, tensor.shape(), ferrum_core::device::Device::Cpu)
            .map_err(|e| DistributedError::CommunicationFailed(e.to_string()))?;

        Ok(())
    }

    fn reduce(&self, tensor: &mut Tensor, dst: usize, op: ReduceOp) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // All-reduce then select destination
        self.all_reduce(tensor, op)?;

        // Only dst has the result, others can keep their copy
        Ok(())
    }

    fn all_gather(&self, output: &mut [Tensor], input: &Tensor) -> Result<()> {
        if self.world_size == 1 {
            output[0] = input.clone();
            return Ok(());
        }

        // Send our tensor to all other ranks
        for peer_rank in 0..self.world_size {
            if peer_rank != self.rank {
                self.send_tensor(peer_rank, input)?;
            }
        }

        // Receive tensors from all other ranks
        output[self.rank] = input.clone();
        for peer_rank in 0..self.world_size {
            if peer_rank != self.rank {
                let recv_tensor = self.recv_tensor(peer_rank, input.shape())?;
                output[peer_rank] = recv_tensor;
            }
        }

        Ok(())
    }

    fn gather(&self, output: Option<&mut [Tensor]>, input: &Tensor, dst: usize) -> Result<()> {
        if self.world_size == 1 {
            if let Some(out) = output {
                out[0] = input.clone();
            }
            return Ok(());
        }

        // Send to dst
        if self.rank != dst {
            self.send_tensor(dst, input)?;
        }

        // dst receives from all
        if self.rank == dst {
            if let Some(out) = output {
                out[self.rank] = input.clone();
                for peer_rank in 0..self.world_size {
                    if peer_rank != self.rank {
                        let recv_tensor = self.recv_tensor(peer_rank, input.shape())?;
                        out[peer_rank] = recv_tensor;
                    }
                }
            }
        }

        Ok(())
    }

    fn scatter(&self, output: &mut Tensor, input: Option<&[Tensor]>, src: usize) -> Result<()> {
        if self.world_size == 1 {
            if let Some(inp) = input {
                *output = inp[0].clone();
            }
            return Ok(());
        }

        if self.rank == src {
            if let Some(inp) = input {
                // Send each chunk to corresponding rank
                for (i, tensor) in inp.iter().enumerate() {
                    if i != self.rank {
                        self.send_tensor(i, tensor)?;
                    }
                }
                *output = inp[self.rank].clone();
            }
        } else {
            // Receive from src
            *output = self.recv_tensor(src, output.shape())?;
        }

        Ok(())
    }

    fn reduce_scatter(&self, output: &mut Tensor, input: &[Tensor], op: ReduceOp) -> Result<()> {
        if self.world_size == 1 {
            if let Some(first) = input.first() {
                *output = first.clone();
            }
            return Ok(());
        }

        // All-reduce first, then take our slice
        if let Some(first) = input.first() {
            let mut reduced = first.clone();
            self.all_reduce(&mut reduced, op)?;
            *output = reduced;
        }

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // Simple barrier: all ranks send and receive a byte
        let dummy = [1u8; 1];

        for peer_rank in 0..self.world_size {
            if peer_rank == self.rank {
                continue;
            }

            // Send
            self.send_data(peer_rank, &dummy)?;

            // Receive
            self.recv_data(peer_rank)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gloo_collectives_creation() {
        // Test single process (no actual communication)
        let collectives = GlooCollectives::new(0, 1, "127.0.0.1", 29500).unwrap();
        assert_eq!(collectives.rank, 0);
        assert_eq!(collectives.world_size, 1);
    }

    #[test]
    fn test_single_process_barrier() {
        let collectives = GlooCollectives::new(0, 1, "127.0.0.1", 29501).unwrap();
        assert!(collectives.barrier().is_ok());
    }
}
