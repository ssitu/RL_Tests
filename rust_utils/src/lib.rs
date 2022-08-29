mod tests;

use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyfunction]
fn discounted_rewards(rewards: &PyList, discount: f32) -> Vec<f32> {
    let n = rewards.len();
    let mut discounted_rewards = vec![0.; n];
    let mut cumulative_reward = 0.;
    for i in (0..n).rev() {
        cumulative_reward = cumulative_reward * discount +
            rewards.get_item(i).unwrap().extract::<f32>().unwrap();
        discounted_rewards[i] = cumulative_reward;
    }
    discounted_rewards
}

#[pyfunction]
fn td_lambda_return(rewards: &PyList, state_values: &PyList, discount: f32, lambda: f32) -> Vec<f32> {
    let n = rewards.len();
    let mut td_lambda_returns = vec![0.; n];
    td_lambda_returns[n - 1] = rewards.get_item(n - 1).unwrap().extract::<f32>().unwrap();
    let lambda_complement = 1. - lambda;
    for i in (0..n - 1).rev() {
        td_lambda_returns[i] = rewards.get_item(i).unwrap().extract::<f32>().unwrap() +
            discount * (lambda_complement *
                state_values.get_item(i + 1).unwrap().extract::<f32>().unwrap() +
                lambda * td_lambda_returns[i + 1]
            )
    }
    td_lambda_returns
}

#[pyfunction]
fn sample_distribution(distribution: &PyList, rand_f: f32) -> usize {
    let mut sum = 0.;
    for i in 0..distribution.len() {
        sum += distribution.get_item(i).unwrap().extract::<f32>().unwrap();
        if rand_f < sum {
            return i;
        }
    }
    0
}

#[pymodule]
fn rust_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(discounted_rewards))?;
    m.add_wrapped(wrap_pyfunction!(td_lambda_return))?;
    m.add_wrapped(wrap_pyfunction!(sample_distribution))?;
    Ok(())
}