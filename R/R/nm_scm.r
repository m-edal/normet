#' Function to perform Synthetic Control Method (SCM) for a single treatment target
#'
#' \code{nm_scm} applies the synthetic control method to estimate the treatment effect
#' for a specified treatment target using control units.
#'
#' @param df Data frame containing the input data.
#' @param poll_col The name of the column containing the pollutant data.
#' @param code_col The name of the column containing the unit codes.
#' @param treat_target The code of the treatment target.
#' @param control_pool A vector of codes representing the control units.
#' @param cutoff_date The date used to split the data into pre-treatment and post-treatment periods.
#'
#' @return A data frame with the actual and synthetic control data, including the treatment effects.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(glmnet)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   pollutant = rnorm(100),
#'   unit_code = rep(c("A", "B", "C", "D"), each = 25)
#' )
#' result <- nm_scm(df, poll_col = "pollutant",
#'                  code_col = "unit_code", treat_target = "A",
#'                  control_pool = c("B", "C", "D"), cutoff_date = "2020-01-01")
#' }
#' @export
nm_scm <- function(df, poll_col, code_col, treat_target, control_pool, cutoff_date) {
  # Process the date column
  df <- df %>%
    nm_process_date()

  df <- df %>%
    filter(!!sym(code_col) %in% c(control_pool, treat_target))

  # Split data into pre-treatment and post-treatment periods
  pre_treatment_df <- df %>%
    filter(date < as.Date(cutoff_date))

  post_treatment_df <- df %>%
    filter(date >= as.Date(cutoff_date))

  # Filtering pre-treatment control data
  x_pre_control <- pre_treatment_df %>%
    filter(!!sym(code_col) != treat_target & !!sym(code_col) %in% control_pool) %>%
    select(date, !!sym(code_col), !!sym(poll_col)) %>%
    pivot_wider(names_from = !!sym(code_col), values_from = !!sym(poll_col)) %>%
    select(-date) %>%
    as.matrix()

  # Extracting target treatment data
  y_pre_treat <- pre_treatment_df %>%
    filter(!!sym(code_col) == treat_target) %>%
    group_by(date) %>%
    summarise(mean_poll = mean(!!sym(poll_col))) %>%
    pull(mean_poll)

  # Defining parameter grids
  lambda_grid <- 10^seq(10, -2, length = 100)
  alpha_grid <- seq(0, 1, length = 11)

  # Preparing to store cross-validation results
  cv_results <- list()
  min_mse <- Inf
  best_alpha <- NULL
  best_lambda <- NULL

  # Cross-validation to find the best alpha and lambda combination
  for (alpha in alpha_grid) {
    cv_ridge <- cv.glmnet(x_pre_control, y_pre_treat, alpha = alpha, lambda = lambda_grid, nfolds = 5, intercept = TRUE)

    if (cv_ridge$cvm[cv_ridge$lambda == cv_ridge$lambda.min] < min_mse) {
      min_mse <- cv_ridge$cvm[cv_ridge$lambda == cv_ridge$lambda.min]
      best_alpha <- alpha
      best_lambda <- cv_ridge$lambda.min
    }

    cv_results[[paste0("alpha_", alpha)]] <- cv_ridge
  }

  # Fitting the final ridge regression model with the best alpha and lambda
  ridge_final <- glmnet(x_pre_control, y_pre_treat, alpha = best_alpha, lambda = best_lambda, intercept = TRUE)

  # Extracting coefficients, including intercept
  coef_ridge_final <- coef(ridge_final)
  intercept <- coef_ridge_final[1]
  w <- as.vector(coef_ridge_final[-1])

  # Re-filter the data for synthetic control calculation
  sc <- df %>%
    filter(!!sym(code_col) != treat_target & !!sym(code_col) %in% control_pool) %>%
    select(date, !!sym(code_col), !!sym(poll_col)) %>%
    pivot_wider(names_from = !!sym(code_col), values_from = !!sym(poll_col)) %>%
    select(-date) %>%
    as.matrix()

  # Calculating synthetic control predictions, including intercept
  synthetic_control <- as.vector(sc %*% w + intercept)

  # Combining synthetic control results with actual data
  data <- df %>%
    filter(!!sym(code_col) == treat_target) %>%
    select(date, !!sym(code_col), !!sym(poll_col)) %>%
    mutate(synthetic = synthetic_control) %>%
    mutate(effects = !!sym(poll_col) - synthetic) %>%
    select(date, !!sym(code_col), !!sym(poll_col), synthetic, effects) %>%
    rename(factual = !!sym(poll_col), treat_target = !!sym(code_col)) %>%
    mutate(pollutant = poll_col)  # Add a new column 'pollutant'

  return(data)
}


#' Main function to perform SCM for multiple treatment targets.
#'
#' \code{nm_scm_all} applies the synthetic control method in parallel for multiple treatment targets.
#'
#' @param df Data frame containing the input data.
#' @param poll_col The name of the column containing the pollutant data.
#' @param code_col The name of the column containing the unit codes.
#' @param control_pool A vector of codes representing the control units.
#' @param cutoff_date The date used to split the data into pre-treatment and post-treatment periods.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#'
#' @return A data frame with the actual and synthetic control data for all treatment targets.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(parallel)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   pollutant = rnorm(100),
#'   unit_code = rep(c("A", "B", "C", "D"), each = 25)
#' )
#' result <- nm_scm_all(df, poll_col = "pollutant",
#'                           code_col = "unit_code", treat_targets = c("A"),
#'                           control_pool = c("B", "C", "D"), cutoff_date = "2020-01-01",
#'                           n_cores = 2)
#' }
#' @export
nm_scm_all <- function(df, poll_col, code_col, control_pool, cutoff_date, n_cores = NULL) {
  # Create the treatment pool
  treatment_pool <- unique(df[[code_col]])

  # Set up the progress bar
  pb <- progress_bar$new(total = length(treatment_pool),
                        format = "  Processing :current/:total [:bar] :percent eta: :eta",
                        width = 80)

  # Initialize the results list
  synthetic_all <- list()

  # Loop through each treatment target
  for (code in treatment_pool) {
    result <- nm_scm(df, poll_col, code_col, code, control_pool, cutoff_date)
    if (!is.null(result)) {
      synthetic_all <- append(synthetic_all, list(result))
    }
    pb$tick() # Update the progress bar
  }

  # Combine results into a single data frame
  synthetic_all <- bind_rows(synthetic_all)

  return(synthetic_all)
}


#' Single treatment target synthetic control with ML models
#'
#' \code{nm_mlsc} applies machine learning models to estimate the synthetic control for a single treatment target.
#'
#' @param df Data frame containing the input data.
#' @param poll_col The name of the column containing the pollutant data.
#' @param code_col The name of the column containing the unit codes.
#' @param treat_target The code of the treatment target.
#' @param control_pool A vector of codes representing the control units.
#' @param cutoff_date The date used to split the data into pre-treatment and post-treatment periods.
#' @param model_config A list containing configuration parameters for model training.
#' @param training_split The proportion of the data to be used for training. Default is 0.75.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the synthetic control data, model statistics, and the trained model.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   pollutant = rnorm(100),
#'   unit_code = rep(c("A", "B", "C", "D"), each = 25)
#' )
#' result <- nm_mlsc(df, poll_col = "pollutant",
#'                   code_col = "unit_code", treat_target = "A",
#'                   control_pool = c("B", "C", "D"), cutoff_date = "2020-01-01",
#'                   model_config = list(max_models = 5, time_budget = 600))
#' }
#' @export
nm_mlsc <- function(df, poll_col, code_col, treat_target, control_pool, cutoff_date, model_config, training_split = 0.75, verbose = TRUE) {
  df <- df %>%
    nm_process_date()

  control_pool_with_target <- c(control_pool, treat_target)

  # Filter and reshape data
  filtered_df <- df %>%
    filter(!!sym(code_col) %in% control_pool_with_target) %>%
    select(date, !!sym(code_col), !!sym(poll_col)) %>%
    pivot_wider(names_from = !!sym(code_col), values_from = !!sym(poll_col))

  # Split data into pre-treatment and post-treatment datasets
  pre_dataset <- filtered_df %>%
    filter(date < cutoff_date)

  post_dataset <- filtered_df %>%
    filter(date >= cutoff_date)

  # Split pre-treatment dataset
  pre_dataset_split <- nm_split_into_sets(pre_dataset, split_method = 'random', fraction = training_split)

  # Train model
  scmodel <- nm_train_model(pre_dataset_split, value = treat_target, variables = control_pool, verbose = verbose)

  # Get model statistics
  mod_stats <- nm_modStats(pre_dataset_split, scmodel, obs = treat_target) %>%
    mutate(pollutant = poll_col, treat_target = treat_target)

  # Predict using the trained model
  sc_predicts <- nm_predict(scmodel, filtered_df)

  # Prepare final dataset
  data <- df %>%
    filter(!!sym(code_col) == treat_target) %>%
    select(date, !!sym(code_col), !!sym(poll_col)) %>%
    rename(factual = !!sym(poll_col), treat_target = !!sym(code_col)) %>%
    mutate(synthetic = sc_predicts, effects = factual - synthetic, pollutant = poll_col)

  return(list(data, mod_stats, scmodel))
}


#' Apply synthetic control with ML models to all treatment targets
#'
#' \code{nm_mlsc_all} applies machine learning models to estimate the synthetic control for all treatment targets in parallel.
#'
#' @param df Data frame containing the input data.
#' @param poll_col The name of the column containing the pollutant data.
#' @param code_col The name of the column containing the unit codes.
#' @param control_pool A vector of codes representing the control units.
#' @param cutoff_date The date used to split the data into pre-treatment and post-treatment periods.
#' @param model_config A list containing configuration parameters for model training.
#' @param training_split The proportion of the data to be used for training. Default is 0.75.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is FALSE.
#'
#' @return A list containing the synthetic control data for all treatment targets, model statistics, and the trained models.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   pollutant = rnorm(100),
#'   unit_code = rep(c("A", "B", "C", "D"), each = 25)
#' )
#' result <- nm_mlsc_all(df, poll_col = "pollutant",
#'                       code_col = "unit_code", control_pool = c("B", "C", "D"),
#'                       cutoff_date = "2020-01-01", model_config = list(max_models = 5, time_budget = 600))
#' }
#' @export
nm_mlsc_all <- function(df, poll_col, code_col, control_pool, cutoff_date, model_config, training_split = 0.75, n_cores = NULL, verbose = FALSE) {

  # Check if H2O is already initialized and initialize if not
  nm_init_h2o()

  # Get unique treatment targets from the dataset
  treatment_pool <- unique(df[[code_col]])

  # Initialize lists to store results
  df_synthetic_list <- list()
  mod_stats_list <- list()
  models_list <- list()

  # Determine the number of CPU cores to use for parallel processing
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Record the start time of the process
  start_time <- Sys.time()

  # Initialize the progress bar to track progress
  pb <- progress_bar$new(
    format = "  Treatment :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(treatment_pool),
    clear = FALSE,
    width = 80
  )

  # Loop through each treatment target
  for (i in seq_along(treatment_pool)) {
    code <- treatment_pool[i]
    success <- FALSE

    # Try to process each treatment target until successful
    while (!success) {
      tryCatch({
        # Call the nm_mlsc function to get synthetic control results
        res <- nm_mlsc(df, poll_col, code_col, code, control_pool, cutoff_date, model_config, training_split, verbose = verbose)

        # Extract the synthetic control data, model statistics, and model object
        synthetic_data <- res[[1]]
        mod_stats_data <- res[[2]]
        model_data <- res[[3]]

        # Store the results in respective lists
        df_synthetic_list[[i]] <- synthetic_data
        mod_stats_list[[i]] <- mod_stats_data
        models_list[[i]] <- model_data

        # Mark as success
        success <- TRUE
      }, error = function(e) {
        # Handle H2O connection errors by reinitializing H2O
        if (grepl("H2O connection error", e$message)) {
          cat(sprintf("%s: H2O connection lost during treatment %d/%d: %s. Reinitializing H2O...\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), i, length(treatment_pool), e$message))
          nm_init_h2o(n_cores)
        } else {
          # Handle other errors and retry after waiting for 10 seconds
          cat(sprintf("%s: Error during treatment %d/%d: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), i, length(treatment_pool), e$message))
          Sys.sleep(10)  # Wait for 10 seconds before retrying
        }
      })
    }

    # Update the progress bar
    if (success) {
      pb$tick()
    }
  }

  # Combine all synthetic control results into one data frame
  synthetic_all <- bind_rows(df_synthetic_list)

  # Combine all model statistics into one data frame
  mod_stats_all <- bind_rows(mod_stats_list)

  # Store all models in a list
  models_all <- models_list

  # Return the combined results as a list
  return(list(synthetic_all = synthetic_all, mod_stats_all = mod_stats_all, models_all = models_all))
}
