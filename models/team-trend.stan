
data {
  int<lower=0> T; // timesteps
  int<lower=0> K; // teams
  matrix[T, K] x; // years
  matrix[T, K] y;
  vector[K] x_tilde;
}


transformed data {
  matrix[T, K] x_prime = (x - min(x)) / ( max(x) - min(x));
  vector[K] x_tilde_prime = (x_tilde - min(x)) / ( max(x) - min(x));
}

parameters {
  vector[K] alpha;
  vector[K] mu;
  real<lower=0> sigma_epsilon;
  real<lower=0> sigma_alpha;
}

transformed parameters {
  matrix[T, K] theta;
  for (k in 1:K) {
    for (t in 1:T) {
      theta[t, k] = alpha[k] * x_prime[t, k] + mu[k];
    }
  }
}

model {
  sigma_alpha ~ exponential(0.1);
  sigma_epsilon ~ exponential(0.1);
  mu ~ normal(mean(y), sd(y) * 2.0);
  
  for (k in 1:K) {
    alpha[k] ~ normal(0.0, sigma_alpha);
  }
  for (k in 1:K) {
    for (t in 1:T) {
      y[t, k] ~ normal(theta[t, k], sigma_epsilon);
    }
  }
}

generated quantities {
  vector[K] y_tilde;
  for (k in 1:K) {
    y_tilde[k] = normal_rng(alpha[k] * x_tilde_prime[k] + mu[k], sigma_epsilon);
  }
}
