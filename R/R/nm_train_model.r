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
nm_train_model <- function(df, value = "value", variables = NULL, model_config = NULL, seed = 7654321, n_cores = NULL, verbose = TRUE, max_retries = 3) {

  # Check for duplicate variables
  if (length(unique(variables)) != length(variables)) {
    stop("`variables` contains duplicate elements.")
  }

  # Check if all variables are in the DataFrame
  if (!all(variables %in% colnames(df))) {
    stop("`variables` given are not within input data frame.")
  }

  # Extract relevant data for training
  if ("set" %in% colnames(df)) {
    df_train <- df %>%
      filter(set == "training") %>%
      select(all_of(c(value, variables)))
  } else {
    df_train <- df %>%
      select(all_of(c(value, variables)))
  }

  # Default configuration for model training
  default_model_config <- list(
    time_budget = NULL,                  # Total running time in seconds
    nfolds = 5,                          # Number of folds for cross-validation
    max_models = 10,                     # Maximum number of models to train
    max_mem_size = "12g",                # Maximum memory size
    estimator_list = c("GBM"),           # List of algorithms to use in AutoML
    verbose = verbose                    # Print progress messages
  )

  # Update default configuration with user-provided config
  if (!is.null(model_config)) {
    default_model_config <- modifyList(default_model_config, model_config)
  }

  # Set up parallel processing
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Function to initialize H2O and train the model
  train_model <- function() {
    # Initialize H2O
    nm_init_h2o(n_cores, max_mem_size = default_model_config$max_mem_size)

    # Convert the training data frame to H2O object
    df_h2o <- h2o::as.h2o(df_train)
    response <- value
    predictors <- setdiff(colnames(df_h2o), response)

    # Print progress message if verbose is TRUE
    if (verbose) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Training AutoML...", "\n")
    }

    # Train AutoML model
    auto_ml <- h2o::h2o.automl(
      x = predictors,
      y = response,
      training_frame = df_h2o,
      max_models = default_model_config$max_models,
      max_runtime_secs = default_model_config$time_budget,
      include_algos = default_model_config$estimator_list,
      seed = seed
    )

    # Print progress message if verbose is TRUE
    if (verbose) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Best model obtained! - ", auto_ml@leader@model_id, "\n", sep = "")
    }

    return(auto_ml)
  }

  # Try to train the model with retries
  retry_count <- 0
  auto_ml <- NULL

  # Loop to retry training in case of an error
  while (is.null(auto_ml) && retry_count < max_retries) {
    retry_count <- retry_count + 1
    tryCatch({
      auto_ml <- train_model()  # Attempt to train the model
    }, error = function(e) {
      # If an error occurs, print error message and retry
      if (verbose) {
        cat(format(Sys.time(), "%Y-%m-%d %H:%M:%OS"), ": Error occurred - ", e$message, "\n", sep = "")
        cat("Retrying... (attempt ", retry_count, " of ", max_retries, ")\n", sep = "")
      }
      h2o::h2o.shutdown(prompt = FALSE) # Shut down the current H2O instance
      Sys.sleep(5)                      # Wait for a few seconds before retrying
    })
  }

  # If all retries fail, stop the function and report failure
  if (is.null(auto_ml)) {
    stop("Failed to train the model after ", max_retries, " attempts.")
  }

  return(auto_ml)  # Return the trained model
}
