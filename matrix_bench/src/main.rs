use ndarray::{Array2};
use std::time::{Duration, Instant};
use rayon::prelude::*;

#[derive(Debug)]
struct MatrixMultiplyBenchmark {
    size: usize,
    iterations: usize,
    times: Vec<Duration>,
}

impl MatrixMultiplyBenchmark {
    fn new(size: usize, iterations: usize) -> Self {
        MatrixMultiplyBenchmark {
            size,
            iterations,
            times: Vec::with_capacity(iterations),
        }
    }

    fn run_benchmark(&mut self) {
        // Create test matrices
        let a = Array2::from_shape_fn((self.size, self.size), 
            |(_i, _j)| rand::random::<f64>());
        let b = Array2::from_shape_fn((self.size, self.size), 
            |(_i, _j)| rand::random::<f64>());

        // Reference result for verification
        let reference = &a.dot(&b);

        for _ in 0..self.iterations {
            let start = Instant::now();
            let result = matrix_multiply(&a, &b);
            let duration = start.elapsed();

            // Verify correctness
            assert!(verify_result(&result, reference));
            
            self.times.push(duration);
        }
    }

    fn report(&self) {
        let avg_time: Duration = self.times.iter().sum::<Duration>() / 
            self.iterations as u32;
        let min_time = self.times.iter().min().unwrap();
        let max_time = self.times.iter().max().unwrap();

        println!("Matrix Multiplication Benchmark Report");
        println!("=====================================");
        println!("Matrix size: {}x{}", self.size, self.size);
        println!("Iterations: {}", self.iterations);
        println!("Average time: {:?}", avg_time);
        println!("Min time: {:?}", min_time);
        println!("Max time: {:?}", max_time);
    }
}

fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    // Validate dimensions
    assert_eq!(a.ncols(), b.nrows(), "Invalid matrix dimensions for multiplication");
    
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    
    let mut result = Array2::zeros((m, n));
    
    // Parallel implementation using rayon
    result.outer_iter_mut().enumerate().par_bridge().for_each(|(i, mut row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k {
                sum += a[[i, k]] * b[[k, j]];
            }
            row[j] = sum;
        }
    });
    
    result
}

fn verify_result(result: &Array2<f64>, reference: &Array2<f64>) -> bool {
    const EPSILON: f64 = 1e-10;
    result.iter().zip(reference.iter())
        .all(|(a, b)| (a - b).abs() < EPSILON)
}

fn main() {
    let mut benchmark = MatrixMultiplyBenchmark::new(1000, 5);
    benchmark.run_benchmark();
    benchmark.report();
}
