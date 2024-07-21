#' Decompose Meteorological Influences
#'
#' \code{nm_decom_met} performs decomposition of meteorological influences on a time series using a trained model.
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
#' @param importance_ascending Logical indicating whether to sort feature importances in ascending order. Default is FALSE.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the decomposed data frame and model statistics.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_decom_met(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, seed = 12345)
#' }
#' @export
nm_decom_met <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, split_method = 'random', fraction = 0.75,
                      model_config = NULL, n_samples = 300, seed = 7654321, importance_ascending = FALSE, n_cores = NULL, verbose = TRUE) {
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

  # Determine feature importances and sort them
  feature_importances <- as.data.frame(h2o::h2o.varimp(model)) %>%
    select(variable, relative_importance) %>%
    arrange(ifelse(importance_ascending, relative_importance, -relative_importance))

  # Initialize the dataframe for decomposed components
  df_dew <- df %>% select(date, value) %>% rename(observed = value)

  # Create a list of features to be excluded
  met_list <- c('deweathered', feature_importances$variable[!feature_importances$variable %in% c('hour', 'weekday', 'day_julian', 'date_unix')])
  var_names <- feature_importances$variable[!feature_importances$variable %in% c('hour', 'weekday', 'day_julian', 'date_unix')]

  # Default logic for CPU cores
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Initialize progress bar
  pb <- progress_bar$new(
    format = "  Iterative subtracting :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(met_list),
    clear = FALSE,
    width = 80
  )

  # Decompose the time series by excluding different features based on their importance
  start_time <- Sys.time()  # Initialize start time before the loop

  for (i in seq_along(met_list)) {
    var_to_exclude <- met_list[i]
    pb$tick()  # Update progress bar

    var_names <- setdiff(var_names, var_to_exclude)

    df_dew_temp <- nm_normalise(df, model, feature_names = feature_names, variables_resample = var_names,
                             n_samples = n_samples, n_cores = n_cores, seed = seed, verbose = FALSE)

    if (nrow(df_dew_temp) == 0) {
      stop(paste("Normalization failed for variable:", var_to_exclude))
    }

    df_dew[[var_to_exclude]] <- df_dew_temp$normalised
  }

  # Initialize df_dewwc with the original data frame
  df_dewwc <- df_dew

  # Extract the relevant feature names excluding specified ones
  relevant_features <- feature_importances$variable[!feature_importances$variable %in% c('hour', 'weekday', 'day_julian', 'date_unix')]

  # Loop through the relevant features and compute the adjusted values
  for (i in seq_along(relevant_features)) {
    param <- relevant_features[i]
    if (i > 1) {
      df_dewwc[[param]] <- df_dew[[param]] - df_dew[[met_list[i - 1]]]
    } else {
      df_dewwc[[param]] <- df_dew[[param]] - df_dew[['deweathered']]
    }
  }

  # Compute 'met_noise' column
  df_dewwc[['met_noise']] <- df_dew[['observed']] - df_dew[[met_list[length(met_list)]]]

  return(list(df_dewwc = df_dewwc, mod_stats = mod_stats))
}
