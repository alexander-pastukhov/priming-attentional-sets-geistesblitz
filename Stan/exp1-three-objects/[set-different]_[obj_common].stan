// Attention sets : fixed set-specific penalty for identifying the set
// Object memory trace representation : per set
// Object memory trace delta : common
data {
  int<lower=1> N;
  int<lower=1, upper=N> FITTING_START_TRIAL; // number of trial from which sampling starts, earlier trials are ignored

  int<lower=1> ObjN;
  int<lower=1> SetsN;
  int<lower=1> ParticipantsN;

  array[N] int<lower=1, upper=80> trial;
  array[N] real logRT;
  array[N] int<lower=1, upper=ObjN> response;
  array[N] int<lower=1, upper=ObjN> target_obj;
  array[N] int<lower=0, upper=1> correct;
  array[N] int<lower=1, upper=ParticipantsN> participant;
  array[N] int<lower=0, upper=1> mixed_block;
  array[N] int<lower=1, upper=SetsN> trial_attention_set;
  array[N] int<lower=0, upper=1> is_direct_attention_set;
  matrix<lower=0, upper=1>[N, ObjN] is_target;
}

transformed data {
  int LM_PARAMS = 3;

  // count trials for log-lik
  int LOGLIK_TRIALS_N = 0;
  for (irow in 1:N) LOGLIK_TRIALS_N += (trial[irow] >= FITTING_START_TRIAL);
}

parameters {
  real<lower=0> sigma;

  // relative obj saliency change   
  vector[ParticipantsN] z_obj_delta;
  real mu_obj_delta;
  real<lower=0> sigma_obj_delta;
  
  // correlated 1) motor - motor response time (intercept)
  //            2) obj - delay due to obj identification decision making (slope)
  //            3) set - delay due to set identification decision making (slope)
  matrix[LM_PARAMS * SetsN, ParticipantsN] z_motor_obj_set; 
  vector[LM_PARAMS * SetsN] mu_motor_obj_set;
  cholesky_factor_corr[LM_PARAMS * SetsN] l_rho_motor_obj_set;
  vector<lower=0>[LM_PARAMS * SetsN] sigma_motor_obj_set;
}

transformed parameters {
  vector[N] logMu;
  vector[ParticipantsN] obj_delta = inv_logit(mu_obj_delta + sigma_obj_delta * z_obj_delta);

  {
    vector[ObjN] obj_memory;
    real object_inconspicuousness; // how inconspicous (opposite of salient) is the responded no object
    
    // computing matrix and decomposing it into RT due to motor response and object processing
    matrix[LM_PARAMS * SetsN, ParticipantsN] motor_obj_set =  rep_matrix(mu_motor_obj_set, ParticipantsN) + diag_pre_multiply(sigma_motor_obj_set, l_rho_motor_obj_set) * z_motor_obj_set;
    matrix[SetsN, ParticipantsN] motor_rt = block(motor_obj_set, 1, 1, SetsN, ParticipantsN);
    matrix[SetsN, ParticipantsN] object_rt = exp(block(motor_obj_set, SetsN + 1, 1, SetsN, ParticipantsN)); // effect can only be positive: additional processing time
    matrix[SetsN, ParticipantsN] set_rt = exp(block(motor_obj_set, 2 * SetsN + 1, 1, SetsN, ParticipantsN)); // effect can only be positive: additional processing time

    for (irow in 1:N){
      // if itrial == 1 a new block starts: reset saliency
      if (trial[irow] == 1) obj_memory = rep_vector(1.0 / ObjN, ObjN);

      // computing target salience
      object_inconspicuousness = 1 - obj_memory[response[irow]] / sum(obj_memory);

      // compute mu using target saliency
      logMu[irow] = motor_rt[trial_attention_set[irow], participant[irow]] + 
                    object_rt[trial_attention_set[irow], participant[irow]] * object_inconspicuousness + 
                    set_rt[trial_attention_set[irow], participant[irow]] * mixed_block[irow];

        // first-order process: saliency changes as a proportion towards ceiling or floor
        for (iObj in 1:ObjN) obj_memory[iObj] += obj_delta[participant[irow]] * (is_target[irow, iObj] - obj_memory[iObj]);
    }
  }
}

model {
  for (irow in 1:N){
    // logRT not computed for first few trials of a block
    if (trial[irow] >= FITTING_START_TRIAL){
      logRT[irow] ~ normal(logMu[irow], sigma);
    }
  }

  // parameters of linear model for object salience -> RTobj
  mu_motor_obj_set[1:2] ~ normal(-0.5, 1); // fixed minimal response time : mode at about 0.57 seconds
  mu_motor_obj_set[3:4] ~ normal(-4, 2);   // effect of object salience: regularizing prior with mode close to zero
  mu_motor_obj_set[5:6] ~ normal(-4, 2);   // effect of set salience: regularizing prior with mode close to zero
  to_vector(z_motor_obj_set) ~ normal(0, 1);
  l_rho_motor_obj_set ~ lkj_corr_cholesky(4);
  sigma_motor_obj_set ~ exponential(1);

  // obj_delta
  mu_obj_delta ~ normal(0, 1);       // regularizing prior with mode at 0.5
  z_obj_delta ~ normal(0, 1);
  sigma_obj_delta ~ exponential(1);

  // variance
  sigma ~ exponential(1);
}

generated quantities {
  vector[LOGLIK_TRIALS_N] log_lik;
  {
    int iloglik = 1; // iloglik for correct logging and indexing of loglik
  
    for(irow in 1:N){
      if(trial[irow] >= FITTING_START_TRIAL){ // loglik is not computed for first few trials of a block
        log_lik[iloglik] = normal_lpdf(logRT[irow] | logMu[irow], sigma);
        iloglik += 1;
      }
    }
  }
}
