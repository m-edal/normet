#' Train a model using H2O's AutoML
#'
#' \code{nm_train_model} is a function to train a model using H2O's AutoML.
#' It initializes H2O, checks for duplicate and missing variables, extracts relevant data for training,
#' sets up parallel processing, and trains the model using AutoML.
#'
#' @param df Input data frame containing the data to be used for training.
#' @param value The target variable name as a string. Default is "value".
#' @param variables Independent/explanatory variables used for training the model.
#' @param model_config A list containing configuration parameters for model training. If not provided, defaults will be used.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for the model training. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is TRUE.
#' @param max_retries Number of times to retry training the model if a connection error occurs. Default is 3.
#'
#' @return The trained AutoML model.
#'
#' @examples
#' \donttest{
#' # Load necessary libraries
#' library(h2o)
#' library(dplyr)
#'
#' # Prepare example data
#' data_example <- data.frame(
#'   value = rnorm(100),
#'   var1 = rnorm(100),
#'   var2 = rnorm(100),
#'   set = rep(c("training", "testing"), 50)
#' )
#'
#' # Train AutoML model using the example data
#' model <- nm_train_model(
#'   df = data_example,
#'   value = "value",
#'   variables = c("var1", "var2"),
#'   model_config = list(max_models = 5, time_budget = 600)
#' )
#' }
#' @export
nm_train_model <- function(df, value = "value", variables = NULL, model_config = NULL, seed = 7654321,
                           n_cores = NULL, verbose = TRUE, max_retries = 3) {

  # Check for duplicate variables
  if (length(unique(variables)) != length(variables)) {
    stop("`variables` contains duplicate elements.")
  }

  # Check if all variables exist in the DataFrame
  if (!all(variables %in% colnames(df))) {
    stop("`variables` are not found in the input data frame.")
  }

  # Extract relevant data for training
  df_train <- if ("set" %in% colnames(df)) {
    df %>% dplyr::filter(set == "training") %>% dplyr::select(all_of(c(value, variables)))
  } else {
    df %>% dplyr::select(all_of(c(value, variables)))
  }

  # Default model configuration parameters
  default_model_config <- list(
    max_models = 10,                  # Maximum number of models to train
    nfolds = 5,                       # Number of cross-validation folds
    max_mem_size = '12G',             # Maximum memory for H2O
    include_algos = c('GBM'),         # Algorithms to include (e.g., GBM)
    save_model = TRUE,                # Whether to save the model
    model_name = 'automl',            # Name for the saved model
    model_path = './',                # Path to save the model
    seed = seed,                      # Random seed for reproducibility
    verbose = verbose                 # Verbose output for progress
  )

  # Update default configuration with user-provided config
  if (!is.null(model_config)) {
    default_model_config <- modifyList(default_model_config, model_config)
  }

  # Set up the number of cores for parallel processing
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Function to initialize H2O and train the model
  train_model <- function() {
    nm_init_h2o(n_cores, max_mem_size = default_model_config$max_mem_size)
    df_h2o <- h2o::as.h2o(df_train)
    response <- value
    predictors <- setdiff(colnames(df_h2o), response)

    if (verbose) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Training AutoML...\n")
    }

    # Train the AutoML model
    auto_ml <- h2o::h2o.automl(
      x = predictors,
      y = response,
      training_frame = df_h2o,
      include_algos = default_model_config$include_algos,
      max_models = default_model_config$max_models,
      nfolds = default_model_config$nfolds,
      seed = default_model_config$seed
    )

    if (verbose) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Best model obtained - ", auto_ml@leader@model_id, "\n")
    }

    return(auto_ml)
  }

  # Initialize retry count and train the model with retry mechanism
  retry_count <- 0
  auto_ml <- NULL

  # Loop to retry training if an error occurs
  while (is.null(auto_ml) && retry_count < max_retries) {
    retry_count <- retry_count + 1
    tryCatch({
      auto_ml <- train_model()
    }, error = function(e) {
      if (verbose) {
        cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Error occurred - ", e$message, "\n")
        cat("Retrying... (Attempt ", retry_count, " of ", max_retries, ")\n")
      }
      # Shut down the H2O instance and retry after a short delay
      h2o::h2o.shutdown(prompt = FALSE)
      Sys.sleep(5)
    })
  }

  # If all retries fail, stop the function and report failure
  if (is.null(auto_ml)) {
    stop("Failed to train the model after ", max_retries, " attempts.")
  }

  # Save the model if configured to do so
  if (default_model_config$save_model) {
    nm_save_h2o(auto_ml@leader, default_model_config$model_path, default_model_config$model_name)
  }

  model <- auto_ml@leader

  return(model)
}
