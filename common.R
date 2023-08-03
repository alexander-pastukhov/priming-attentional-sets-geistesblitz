# Common utilities

compute_repetitions <- function(sequence) {
  # computes repetitions within sequence
  # A A B C B B A A A -> 0 1 0 0 0 1 0 1 2
  purrr::map(rle(as.character(sequence))$lengths, ~1:.) |> as_vector() - 1
}

lower_ci <- function(values, CI = 0.97) {
  quantile(values, (1 - CI) / 2)
}

upper_ci <- function(values, CI = 0.97) {
  quantile(values, 1 - (1 - CI) / 2)
} 

## Utility functions that compute repetition
# Utility functions that compute repetition based on prediction level (group or individual) and filter function
compute_repetition_for_predictions <- function(df, predictedlogRT, repetition_var, filter_function, level) {
  repetition_df <-
    df |>
    # repetition per participant
    mutate(PredictedLogRT = predictedlogRT) |>
    filter_function() |>
    group_by_at(c("participant.code", "TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
    summarize(meanRT = exp(mean(PredictedLogRT)),
              .groups = "drop")
  
  if (level == "group") {
    repetition_df |>
      
      # average per group
      group_by_at(c("TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(meanRT = mean(meanRT), .groups = "drop")
  } else { # |"participant"
    repetition_df
  }
}

sample_repetition <- function(df, repetition_var, filter_function, level) {
  repetition_df <-
    df |>
    sample_frac(size = 1, replace = TRUE) |>
    filter_function() |>
    group_by_at(c("participant.code", "TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
    summarize(meanRT = exp(mean(logRT)),
              .groups = "drop")
  
  if (level == "group") {
    repetition_df |>
      
      # average per group
      group_by_at(c("TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(meanRT = mean(meanRT), .groups = "drop") 
  } else { # |"participant"
    repetition_df
  }
}


# Wrappers to compute averages
compute_average_predicted_repetition <- function(repetition_var, filter_function, level, R=NULL) {
  rows_to_use <- 1:nrow(predicted_logRT)
  if (!is.null(R)) {
    rows_to_use <- sample(rows_to_use, R, replace = FALSE)
  }
  
  # prediction per sample
  predictions_df <- purrr::map_dfr(rows_to_use, 
                                   ~compute_repetition_for_predictions(results,
                                                                       as_vector(predicted_logRT[., ]),
                                                                       repetition_var,
                                                                       filter_function,
                                                                       level),
                                   .progress = TRUE)
  
  # averages
  if (level == 'group') {
    predictions_df |>
      group_by_at(c("TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(Avg = mean(meanRT),
                LowerCI = lower_ci(meanRT),
                UpperCI = upper_ci(meanRT),
                .groups = "drop")
  } else {
    predictions_df |>
      group_by_at(c("participant.code", "TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(Avg = mean(meanRT),
                LowerCI = lower_ci(meanRT),
                UpperCI = upper_ci(meanRT),
                .groups = "drop")
  }
}

sample_average_repetition <- function(repetition_var, filter_function, level, R=1000) {
  # sample
  sampled_repetition_df <- purrr::map_dfr(1:R,
                                          ~sample_repetition(results,
                                                             repetition_var,
                                                             filter_function,
                                                             level),
                                          .progress = TRUE)
  
  # averages
  if (level == 'group') {
    sampled_repetition_df |>
      group_by_at(c("TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(Avg = mean(meanRT),
                LowerCI = lower_ci(meanRT),
                UpperCI = upper_ci(meanRT),
                .groups = "drop")
  } else {
    sampled_repetition_df |>
      group_by_at(c("participant.code", "TrialBlockAttentionSet", paste0("Repetition", repetition_var))) |>
      summarise(Avg = mean(meanRT),
                LowerCI = lower_ci(meanRT),
                UpperCI = upper_ci(meanRT),
                .groups = "drop")
  }
}


# General plotting routine
plot_repetition <- function(title, repetition_var, predictions_df, behaviour_df, level) {
  plot <- 
    ggplot(behaviour_df, aes_string(x = paste0("Repetition", repetition_var), y = "Avg", color = "TrialBlockAttentionSet")) +
    geom_ribbon(data = predictions_df,
                aes(ymin = LowerCI, ymax = UpperCI, fill = TrialBlockAttentionSet), alpha = 0.25, color = NA) +
    geom_line() +
    geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI)) +
    geom_point() +
    xlab(paste0(repetition_var, " repetition")) +
    ylab("RT, geometric mean [s]") +
    theme(legend.position = "top", strip.background = element_blank(), strip.text = element_blank()) +
    labs(title = title)
  
  if (level == "group") {
    plot <- plot + facet_wrap(.~TrialBlockAttentionSet, ncol = length(unique(behaviour_df$TrialBlockAttentionSet)), scales = "free_y")
  } else { # "participant"
    plot <- plot + facet_wrap(participant.code~TrialBlockAttentionSet, ncol = length(unique(behaviour_df$TrialBlockAttentionSet)), scales = "free_y")
  }
  
  plot
}

# LOO for subset of rows
loo_using_filter <- function(df, log_lik, filter_function) {
  irow <- 
    df |> 
    filter_function() |>
    pull(iloglik)
  
  pbar <- progress_bar$new(format = "(:spin) [:bar] :percent", total = length(log_lik))
  filtered_loos <- list()
  for(imodel in 1:length(log_lik)) {
    pbar$tick()
    reff <- loo::relative_eff(as.matrix(log_lik[[imodel]][, irow + 3]), chain_id = log_lik[[imodel]]$.chain)
    filtered_loos[[imodel]] <- loo::loo(as.matrix(log_lik[[imodel]][, irow + 3]), r_eff = reff)
  }
  
  names(filtered_loos) <- names(log_lik)
  filtered_loos
}

# LOO summary 
compute_loo_summary_df <- function(loos, components = c("Set", "Object", "Color")) {
  # compute model weight weight
  model_weights <- loo::loo_model_weights(loos)
  weights_df <-
    tibble(Model = names(model_weights),
           Weight = round(as.numeric(model_weights), 2))
  
  # compare models
  model_comparison <-
    loo::loo_compare(loos) |>
    as.data.frame() |>
    rownames_to_column("Model") |>
    select(Model, elpd_diff, se_diff) |>
    mutate(across(elpd_diff:se_diff, ~round(., 2))) |>
    mutate(dELPD = glue("{elpd_diff}±{se_diff}")) |>
    select(-elpd_diff, -se_diff) |>
    left_join(weights_df, by = "Model") |>
    separate(Model, into = components, sep = "_") |>
    mutate(across(Set:Object, ~str_remove_all(., "\\[|\\]"))) |>
    mutate(across(Set:Object, ~str_remove_all(., "set-|obj-|color-"))) 
  }
