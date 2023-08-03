// Attention sets : fixed set-specific penalty for identifying the set
// Object memory trace representation : per set
// Object memory trace delta : common
// Color memory trace : evolves at object-delta speed

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

  array[N, ObjN] int<lower=1, upper=ColorsN> object_color;  

  array[N] int<lower=0, upper=1> mixed_block;
  array[N] int<lower=1, upper=SetsN> trial_attention_set;
  array[N] int<lower=0, upper=1> is_direct_attention_set;
}

transformed data {
  int LM_PARAMS = 3;

  // count valid trials for which log_lik needs to be computed
  int LOGLIK_TRIALS_N = 0;
  for (irow in 1:N) LOGLIK_TRIALS_N +=  (trial[irow] >= FITTING_START_TRIAL);
}

parameters {
  real<lower=0> sigma;

  // relative obj and color salience change   
  matrix[2, ParticipantsN] z_obj_set_delta;
  vector[2] mu_obj_set_delta;
  cholesky_factor_corr[2] l_rho_obj_set_delta;
  vector<lower=0>[2] sigma_obj_set_delta;

  // boost due to being on a card
  matrix[SetsN, ParticipantsN] presence_z;
  vector[SetsN] presence_mu;
  cholesky_factor_corr[SetsN] presence_l_rho;
  vector<lower=0>[SetsN] presence_sigma;

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
  matrix[2, ParticipantsN] obj_set_delta = inv_logit(rep_matrix(mu_obj_set_delta, ParticipantsN) + diag_pre_multiply(sigma_obj_set_delta, l_rho_obj_set_delta) * z_obj_set_delta);

  {
    matrix [SetsN, ObjN] obj_salience;
    row_vector [ObjN] round_salience;
    real object_inconspicuousness; // how inconspicous (opposite of salient) is the responded no object
    vector[SetsN] set_salience;

    // computing matrix and decomposing it into RT due to motor response and object processing
    matrix[LM_PARAMS * SetsN, ParticipantsN] motor_obj_set =  rep_matrix(mu_motor_obj_set, ParticipantsN) + diag_pre_multiply(sigma_motor_obj_set, l_rho_motor_obj_set) * z_motor_obj_set;
    matrix[SetsN, ParticipantsN] motor_rt = block(motor_obj_set, 1, 1, SetsN, ParticipantsN);
    matrix[SetsN, ParticipantsN] object_rt = exp(block(motor_obj_set, SetsN + 1, 1, SetsN, ParticipantsN)); // effect can only be positive: additional processing time
    matrix[SetsN, ParticipantsN] set_rt = exp(block(motor_obj_set, 2 * SetsN + 1, 1, SetsN, ParticipantsN)); // effect can only be positive: additional processing time

    // boost due to presence on the card (can be at most 1 -> 100%)
    matrix[SetsN, ParticipantsN] presence = inv_logit(rep_matrix(presence_mu, ParticipantsN) + diag_pre_multiply(presence_sigma , presence_l_rho) * presence_z);

    for (irow in 1:N){
      // if itrial == 1 a new block starts: reset saliency
      if (trial[irow] == 1) {
        obj_salience = rep_matrix(1.0 / ObjN, SetsN, ObjN);
        set_salience = rep_vector(1.0 / SetsN, SetsN);
      }

      // computing target inconspicuousness
      round_salience = obj_salience[trial_attention_set[irow]];
      for(iObj in 1:ObjN) round_salience[iObj] += (1 - round_salience[iObj]) * presence[trial_attention_set[irow], participant[irow]] * is_round_object[irow, iObj];
      object_inconspicuousness = 1 - round_salience[response[irow]] / sum(round_salience);

      // compute mu using target saliency
      logMu[irow] = motor_rt[trial_attention_set[irow], participant[irow]] + 
                    object_rt[trial_attention_set[irow], participant[irow]] * object_inconspicuousness + 
                    set_rt[trial_attention_set[irow], participant[irow]] * mixed_block[irow] * (1 - set_salience[trial_attention_set[irow]]);

      // first-order process: saliency changes as a proportion towards ceiling or floor
      for (iObj in 1:ObjN) obj_salience[trial_attention_set[irow], iObj] += obj_set_delta[1, participant[irow]] * (is_target[irow, iObj] - obj_salience[trial_attention_set[irow], iObj]);

      // attention set saliency changes (the other set is complimentary)
      set_salience[1] += obj_set_delta[2, participant[irow]] * (is_direct_attention_set[irow] - set_salience[1]);
      set_salience[2] = 1 - set_salience[1];
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

  // boost due to being present on the card
  presence_mu ~ normal(-2, 2); // ~10% boost
  to_vector(presence_z) ~ normal(0, 1);
  presence_l_rho ~ lkj_corr_cholesky(2);
  presence_sigma ~ exponential(10);

  // obj and color change
  mu_obj_set_delta ~ normal(0, 1);       // regularizing prior with mode at 0.5
  to_vector(z_obj_set_delta) ~ normal(0, 1);
  l_rho_obj_set_delta ~ lkj_corr_cholesky(2);
  sigma_obj_set_delta ~ exponential(10);

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
