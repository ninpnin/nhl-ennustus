data {
  int<lower=0> N;
  int<lower=0> N_test;
  int<lower=0> M;
  matrix[N,M] x;
  vector[N] y;
  matrix[N_test,M] x_test;
}

transformed data {
  real y_mean = mean(y);
  real y_std = sqrt(mean(y));
}

parameters {
  vector[M] beta;
  real beta_0;
}

model {
  // Prior
  beta ~ normal(0, 5);
  beta_0 ~ normal(y_mean, y_std);

  // Likelihood
  for (i in 1:N)
    y[i] ~ normal(beta_0 + dot_product(beta, x[i]), 1);
}

generated quantities {
  vector[N_test] y_test_hat;
  for (i in 1:N_test)
    y_test_hat[i] = normal_rng(beta_0 + dot_product(beta, x_test[i]), 5);
}
