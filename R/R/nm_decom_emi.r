#' Decompose Emissions Influences
#'
#' \code{nm_decom_emi} performs decomposition of emissions influences on a time series using a trained model.
#'
#' @param df Data frame containing the input data.
#' @param model Pre-trained model for decomposition. If not provided, a model will be trained.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for training and decomposition.
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param model_config A list containing configuration parameters for model training.
#' @param n_samples Number of samples to generate for normalization. Default is 300.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the decomposed data frame and model statistics.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_decom_emi(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, seed = 12345)
#' }
#' @export
nm_decom_emi <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, split_method = 'random', fraction = 0.75,
                      model_config = NULL, n_samples = 300, seed = 7654321, n_cores = NULL, verbose = TRUE) {
  library(dplyr)
  library(lubridate)
  # Check if h2o is already initialized
  nm_init_h2o(n_cores)

  set.seed(seed)

  # Train model if not provided
  if (is.null(model)) {
    res <- nm_prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)
    df <- res$df
    model <- res$model
  }

  # Gather model statistics for testing, training, and all data
  mod_stats <- nm_modStats(df, model)

  # Initialize the dataframe for decomposed components
  df_dew <- df %>% select(date, value) %>% rename(observed = value)

  # Default logic for CPU cores
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Initialize progress bar
  pb <- progress_bar$new(
    format = "  Iterative subtracting :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(c('base', 'date_unix', 'day_julian', 'weekday', 'hour')),
    clear = FALSE,
    width = 60
  )

  # Decompose the time series by excluding different features
  var_names <- feature_names
  start_time <- Sys.time()  # Initialize start time before the loop

  for (i in seq_along(c('base', 'date_unix', 'day_julian', 'weekday', 'hour'))) {
    pb$tick()  # Update progress bar

    var_to_exclude <- c('base', 'date_unix', 'day_julian', 'weekday', 'hour')[i]

    var_names <- setdiff(var_names, var_to_exclude)

    df_dew_temp <- nm_normalise(df, model, feature_names = feature_names, variables_resample = var_names,
                             n_samples = n_samples, n_cores = n_cores, seed = seed, verbose = FALSE)

    df_dew[[var_to_exclude]] <- df_dew_temp$normalised
  }

  # Adjust the decomposed components to create deweathered values
  df_dew <- df_dew %>%
    mutate(
      deweathered = hour,
      hour = hour - weekday,
      weekday = weekday - day_julian,
      day_julian = day_julian - date_unix,
      date_unix = date_unix - base + mean(base, na.rm = TRUE),
      emi_noise = base - mean(base, na.rm = TRUE)
    )

  return(list(df_dew = df_dew, mod_stats = mod_stats))
}
