data {
  int<lower=0> T; // Timesteps
  int<lower=0> K; // Groups
  matrix[T+1, K] y;
  real sigma0;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  matrix[T, K] mu;
  real<lower=0> sigma_mu;
  real<lower=0> sigma_epsilon;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  sigma_epsilon ~ exponential(1.0 / sigma0);
  sigma_mu ~ exponential(2.0 / sigma0);
  for (t in 2:T) {
    mu[t] ~ normal(mu[t-1], sigma_mu);
    y[t] ~ normal(y[t-1] + mu[t], sigma_epsilon);
  }
}

generated quantities {
  vector[K] mu_tilde;
  vector[K] y_tilde;
  mu_tilde = normal_rng(mu[T], sigma_mu);
  y_tilde = normal_rng(y[T] + mu_tilde);
}
