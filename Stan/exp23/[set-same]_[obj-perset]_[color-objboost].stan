// Attention sets : fixed set-specific penalty for identifying the set
// Object memory trace representation : per set
// Object memory trace delta : common
// Color memory trace : none

data {
  int<lower=1> N;
  int<lower=1, upper=N> FITTING_START_TRIAL; // number of trial from which sampling starts, earlier trials are ignored

  int<lower=1> ObjN;
  int<lower=1> ColorsN;
  int<lower=1, upper=ObjN> ObjectsInTrial;
  int<lower=1, upper=ColorsN> ColorsInTrial;
  int<lower=1> SetsN;
  int<lower=1> ParticipantsN;

  array[N] int<lower=1, upper=80> trial;
  array[N] real logRT;
  array[N] int<lower=1, upper=ObjN> response;
  array[N] int<lower=0, upper=1> correct;
  array[N] int<lower=1, upper=ParticipantsN> participant;
  
  array[N, ObjN] int<lower=0, upper=1> is_target;
  array[N, ColorsN] int<lower=0, upper=1> is_target_color;

  array[N, ObjN] int<lower=0, upper=1> is_round_object;
  array[N, ObjN] int<lower=0, upper=1> is_same_color_as_target;
  array[N, ColorsN] int<lower=0, upper=1> is_round_color;

  array[N] int<lower=0, upper=1> mixed_block;
  array[N] int<lower=1, upper=SetsN> trial_attention_set;
  array[N] int<lower=0, upper=1> is_direct_attention_set;
}

transformed data {
  int LM_PARAMS = 2;

  // count valid trials for which log_lik needs to be computed
  int LOGLIK_TRIALS_N = 0;
  for (irow in 1:N) LOGLIK_TRIALS_N +=  (trial[irow] >= FITTING_START_TRIAL);
}

parameters {
  real<lower=0> sigma;

  // relative obj saliency change   
  vector[ParticipantsN] z_obj_delta;
  real mu_obj_delta;
  real<lower=0> sigma_obj_delta;
  
  // correlated 1) motor - motor response time (intercept)
  //            2) obj - delay due to obj identification decision making (slope)
  matrix[LM_PARAMS * SetsN, ParticipantsN] z_motor_obj; 
  vector[LM_PARAMS * SetsN] mu_motor_obj;
  cholesky_factor_corr[LM_PARAMS * SetsN] l_rho_motor_obj;
  vector<lower=0>[LM_PARAMS * SetsN] sigma_motor_obj;

  // boost due to being on a card
  matrix[SetsN, ParticipantsN] presence_z;
  vector[SetsN] presence_mu;
  cholesky_factor_corr[SetsN] presence_l_rho;
  vector<lower=0>[SetsN] presence_sigma;

  // memory adjustment due to being the same color as the target
  matrix[SetsN, ParticipantsN] color_z;
  vector[SetsN] color_mu;
  cholesky_factor_corr[SetsN] color_l_rho;
  vector<lower=0>[SetsN] color_sigma;

  // penalty for figuring out set in mixed block
  vector[ParticipantsN] z_set;
  real mu_set;
  real<lower=0> sigma_set;  
}

transformed parameters {
  vector[N] logMu;
  vector[ParticipantsN] obj_delta = inv_logit(mu_obj_delta + sigma_obj_delta * z_obj_delta);

  {
    matrix [SetsN, ObjN] obj_memory;
    row_vector[ObjN] round_salience;
    real object_inconspicuousness; // how inconspicous (opposite of salient) is the responded no object
    
    // computing matrix and decomposing it into RT due to motor response and object processing
    matrix[LM_PARAMS * SetsN, ParticipantsN] motor_obj = rep_matrix(mu_motor_obj, ParticipantsN) + diag_pre_multiply(sigma_motor_obj , l_rho_motor_obj) * z_motor_obj;
    matrix[SetsN, ParticipantsN] motor_rt = block(motor_obj, 1, 1, SetsN, ParticipantsN);
    matrix[SetsN, ParticipantsN] object_rt = exp(block(motor_obj, SetsN + 1, 1, SetsN, ParticipantsN)); // effect can only be positive: additional processing time
    vector[ParticipantsN] set_rt = exp(mu_set + sigma_set * z_set); // effect can only be positive: additional processing time due to identifying set
  
    // boost due to presence on the card (can be at most 1 -> 100%)
    matrix[SetsN, ParticipantsN] presence = inv_logit(rep_matrix(presence_mu, ParticipantsN) + diag_pre_multiply(presence_sigma , presence_l_rho) * presence_z);

    // memory adjustment due to having same color as the target (can be at most 1)
    matrix[SetsN, ParticipantsN] color_modulation = inv_logit(rep_matrix(color_mu, ParticipantsN) + diag_pre_multiply(color_sigma , color_l_rho) * color_z);

    for (irow in 1:N){
      // if itrial == 1 a new block starts: reset saliency
      if (trial[irow] == 1) obj_memory = rep_matrix(1.0 / ObjN, SetsN, ObjN);

      // computing target inconspicuousness
      round_salience = obj_memory[trial_attention_set[irow]];
      for(iObj in 1:ObjN) {
        real extra_boost = presence[trial_attention_set[irow], participant[irow]] * is_round_object[irow, iObj];
        if (extra_boost > 1) extra_boost = 1;

        round_salience[iObj] += (1 - round_salience[iObj]) * extra_boost;
      }
      object_inconspicuousness = 1 - round_salience[response[irow]] / sum(round_salience);

      // compute logMu using response object saliency
      logMu[irow] = motor_rt[trial_attention_set[irow], participant[irow]] + 
                    object_rt[trial_attention_set[irow], participant[irow]] * object_inconspicuousness + 
                    set_rt[participant[irow]] * mixed_block[irow];

      // first-order process: saliency changes as a proportion towards ceiling or floor
      for (iObj in 1:ObjN){
        if (is_same_color_as_target[irow, iObj]) {
          obj_memory[trial_attention_set[irow], iObj] += color_modulation[trial_attention_set[irow], participant[irow]] * obj_delta[participant[irow]] * (1 - obj_memory[trial_attention_set[irow], iObj]);
        } else {
          obj_memory[trial_attention_set[irow], iObj] += obj_delta[participant[irow]] * (is_target[irow, iObj] - obj_memory[trial_attention_set[irow], iObj]);
        }
      }
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
  mu_motor_obj[1:2] ~ normal(-0.5, 1); // fixed minimal response time : mode at about 0.57 seconds
  mu_motor_obj[3:4] ~ normal(-4, 2);   // effect of object salience: regularizing prior with mode close to zero
  to_vector(z_motor_obj) ~ normal(0, 1);
  l_rho_motor_obj ~ lkj_corr_cholesky(4);
  sigma_motor_obj ~ exponential(1);

  // obj_delta
  mu_obj_delta ~ normal(0, 1);       // regularizing prior with mode at 0.5
  z_obj_delta ~ normal(0, 1);
  sigma_obj_delta ~ exponential(1);

  // boost due to being present on the card
  presence_mu ~ normal(-2, 2); // ~10% boost
  to_vector(presence_z) ~ normal(0, 1);
  presence_l_rho ~ lkj_corr_cholesky(2);
  presence_sigma ~ exponential(10);

  // memory adjustment due to being the same color as the target
  color_mu ~ normal(-2, 2); // ~10% of the target adjustment 
  to_vector(color_z) ~ normal(0, 1);
  color_l_rho ~ lkj_corr_cholesky(2);
  color_sigma ~ exponential(10);

  // penalty for figuring out set in mixed block
  mu_set ~ normal(-4, 2);
  z_set ~ normal(0, 1);
  sigma_set ~ exponential(1);  

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
