#' Perform All Steps for Meteorological normalisation
#'
#' \code{nm_do_all} performs the entire process of training a model, normalising the data, and collecting model statistics.
#'
#' @param df Data frame containing the input data.
#' @param model Pre-trained model for normalisation. If not provided, a model will be trained.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for training and normalisation.
#' @param variables_resample The names of the variables to be resampled for normalisation. Default is NULL (all feature names except date_unix).
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param model_config A list containing configuration parameters for model training.
#' @param n_samples Number of samples to generate for normalisation. Default is 300.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param aggregate Logical indicating whether to aggregate the results. Default is TRUE.
#' @param weather_df Optional data frame containing weather data for resampling.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the normalised data frame and model statistics.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_do_all(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, seed = 12345)
#' }
#' @export
nm_do_all <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, variables_resample = NULL, split_method = 'random', fraction = 0.75,
                   model_config = NULL, n_samples = 300, seed = 7654321, n_cores = NULL, aggregate = TRUE, weather_df = NULL, verbose = TRUE) {
  set.seed(seed)
  # Default logic for CPU cores
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Initialize H2O
  nm_init_h2o(n_cores)

  # Train model if not provided
  if (is.null(model)) {
    res <- nm_prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)
    df <- res$df
    model <- res$model
  }

  # Collect model statistics
  mod_stats <- nm_modStats(df, model)

  # normalise the data using weather_df if provided
  df_dew <- nm_normalise(df, model, feature_names = feature_names, variables_resample = variables_resample, n_samples = n_samples,
                      aggregate = aggregate, n_cores = n_cores, seed = seed, weather_df = weather_df, verbose = verbose)

  return(list(df_dew = df_dew, mod_stats = mod_stats))
}
