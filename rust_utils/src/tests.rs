#[cfg(test)]
mod tests {
    use pyo3::Python;
    use pyo3::types::PyList;
    use rand::Rng;
    use statrs::assert_almost_eq;
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    use crate::*;

    fn assert_vec_equal(a: &Vec<f32>, b: &Vec<f32>, precision: f64) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert_almost_eq!(f64::from(*x), f64::from(*y), precision);
        }
    }

    #[test]
    fn test_discounted_rewards() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let rewards = vec![1., 2., 3., 4., 5.];
        let y = 0.9;
        let expected = vec![1. + y * (2. + y * (3. + y * (4. + y * 5.))),
                            2. + y * (3. + y * (4. + y * 5.)),
                            3. + y * (4. + y * 5.), 4. + y * 5., 5.];
        let rewards = PyList::new(py, &rewards);
        let actual = discounted_rewards(rewards, y);
        assert_vec_equal(&actual, &expected, 0.00000001);
    }

    #[test]
    fn test_td_lambda_return() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let rewards = vec![1., 0., 0., 1., 0., 0., 5., 0., 0., 1.];
        let rewards = PyList::new(py, &rewards);
        let n = rewards.len();
        let state_values = vec![0.; n];
        let state_values = PyList::new(py, &state_values);
        let y = 0.9;

        let precision = 0.00001;
        // Case 1: When lambda = 1, the return is equal to the Monte Carlo return
        let expected = discounted_rewards(rewards, y);
        let actual = td_lambda_return(rewards, state_values, y, 1.);
        assert_vec_equal(&expected, &actual, precision);

        // Case 2: When lambda = 0, the return is equal to one step TD return
        let state_values = PyList::new(py, vec![1.; n]);
        let actual = td_lambda_return(rewards, state_values, y, 0.);

        // Calculate one step TD return
        let mut expected: Vec<f32> = vec![0.; n];
        // Add a 0 value to the state_values for the terminal state
        for i in 0..n {
            let r = rewards.get_item(i).unwrap().extract::<f32>().unwrap();
            let v = if i == n - 1 {0.} else {
                state_values.get_item(i + 1).unwrap().extract::<f32>().unwrap()};
            expected[i] = r + y * v;
        }
        assert_vec_equal(&expected, &actual, precision);
    }

    #[test]
    fn test_sample_distribution() {
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
            let rand_f = rng.gen();
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
        // There is a chance that this will fail,
        // even though the sample is from the expected distribution
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
        // The probability to reject a sample that came from the correct distribution
        let significance_level = 0.001;
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