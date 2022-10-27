data {
  int<lower=0> N;
  int<lower=0> N_test;
  int<lower=0> M;

  matrix[N,M] x;
  int z[N];
  vector[N] y;

  matrix[N_test,M] x_test;
  int z_test[N_test]; // Player group (left forward, defender, ..)
}

transformed data {
  real y_mean = mean(y);
  real y_std = sqrt(mean(y));
  int J = max(z);
}

parameters {
  vector[M] beta_all;
  //matrix[J, M] beta;
  real beta_0;
  vector[J] beta_0_J;
  real<lower=0> tau_0;
  real<lower=0> sigma;
}

model {
  // Prior
  beta_all ~ normal(0, y_mean * 0.5);
  tau_0 ~ normal(0, 1);
  //for 
  //beta_all ~ normal(0, y_mean * 0.5);

  // Intercept
  beta_0 ~ normal(y_mean, y_std);
  beta_0_J ~ normal(beta_0, tau_0 * y_std);
  
  sigma ~ normal(y_std, y_std);
  // Likelihood
  for (i in 1:N) {
    int group = z[i];
    y[i] ~ normal(beta_0_J[group] + dot_product(beta_all, x[i]), sigma);
  }
}

/*
generated quantities {
  vector[N_test] y_test_hat;
  for (i in 1:N_test) {
    int group = z_test[i];
    y_test_hat[i] = normal_rng(beta_0_J[group] + dot_product(beta[group], x_test[i]), sigma);
  }
}
*/