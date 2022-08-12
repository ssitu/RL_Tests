use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyfunction]
fn discounted_rewards(rewards: &PyList, discount: f32) -> Vec<f32> {
    let n = rewards.len();
    let mut discounted_rewards = vec![0.; rewards.len()];
    let mut cumulative_reward = 0.;
    for i in (0..n).rev() {
        cumulative_reward = cumulative_reward * discount +
            rewards.get_item(i).unwrap().extract::<f32>().unwrap();
        discounted_rewards[i] = cumulative_reward;
    }
    discounted_rewards
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
    m.add_wrapped(wrap_pyfunction!(sample_distribution))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::ffi::Py_Initialize;
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    use super::*;

    #[test]
    fn test_discounted_rewards() {
        unsafe { Py_Initialize() };

        let gil = Python::acquire_gil();
        let py = gil.python();
        let rewards = vec![1., 2., 3., 4., 5.];
        let y = 0.9;
        let expected = vec![1. + y * (2. + y * (3. + y * (4. + y * 5.))),
                            2. + y * (3. + y * (4. + y * 5.)),
                            3. + y * (4. + y * 5.), 4. + y * 5., 5.];
        // Get the python interpreter
        let rewards = PyList::new(py, &rewards);
        let actual = discounted_rewards(rewards, y);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_sample_distribution() {
        unsafe { Py_Initialize() };
        let gil = Python::acquire_gil();
        let py = gil.python();

        let distribution: Vec<f32> = vec![0.1, 0.4, 0.2, 0.3];
        // Create count vector
        let mut count: Vec<f32> = vec![0.; distribution.len()];
        let iterations = 100000;
        // Setup the random number generator
        let mut rng = rand::thread_rng();
        // Create a PyList from the distribution
        let py_distribution = PyList::new(py, &distribution);
        // Sample the distribution over the number of iterations
        for _ in 0..iterations {
            let rand_f = rng.gen::<f32>();
            let index = sample_distribution(py_distribution, rand_f);
            count[index] += 1.;
        }
        // Create the expected counts
        let expected: Vec<f32> = vec![0.1 * iterations as f32,
                                      0.4 * iterations as f32,
                                      0.2 * iterations as f32,
                                      0.3 * iterations as f32];
        println!("Expected counts: {:?}", expected);
        println!("Observed counts: {:?}", count);
        // Perform chi-squared test
        let mut chi_square = 0.;
        for (i, c) in count.iter().enumerate() {
            let diff = *c - expected[i];
            chi_square += diff * diff / expected[i];
        }
        // Degrees of freedom = number of elements in the distribution - 1
        let dof = (distribution.len() - 1) as f32;
        // Probability of obtaining the difference between the count vector and the expected vector
        // with a Chi-squared distribution with dof degrees of freedom
        let chi = ChiSquared::new(dof.into()).unwrap();

        println!("chi_square: {:?}", chi_square);
        // cdf at chi_square value
        let cdf = chi.cdf(chi_square.into());
        println!("cdf: {:?}", cdf);
        // Significance level = 0.05
        let significance_level = 0.05;
        // Null hypothesis = the count vector is the same as the expected vector
        // If the cdf at chi_square value is less than 1 - significance_level,
        // then the null hypothesis is failed to be rejected
        let fail_to_reject = cdf < (1. - significance_level);
        println!("The observed counts did not come \
        from the true distribution according to \
        the Chi-Square Goodness of Fit test: {:?}", !fail_to_reject);
        assert!(fail_to_reject);
    }
}