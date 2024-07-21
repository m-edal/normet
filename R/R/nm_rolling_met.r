#' Apply rolling window meteorological normalization
#'
#' \code{nm_rolling_met} performs meteorological normalization on data using a rolling window approach.
#'
#' @param df Data frame containing the input data.
#' @param model Pre-trained model for normalization. If not provided, a model will be trained.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for normalization.
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param model_config A list containing configuration parameters for model training.
#' @param n_samples Number of times to sample the data for normalization. Default is 300.
#' @param window_days The size of the rolling window in days. Default is 14.
#' @param rollingevery The interval at which the rolling window is applied in days. Default is 7.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the normalized data frame and model statistics.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(h2o)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   target = rnorm(100)
#' )
#' result <- nm_rolling_met(df, value = "target", feature_names = c("feature1", "feature2"), window_size = 10, model_type = "h2o.gbm", seed = 12345)
#' }
#' @export
nm_rolling_met <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, split_method = 'random', fraction = 0.75,
                        model_config = NULL, n_samples = 300, window_days = 14, rollingevery = 7, seed = 7654321, n_cores = NULL, verbose = TRUE) {
  set.seed(seed)

  # Default logic for CPU cores
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Initialize H2O
  nm_init_h2o(n_cores)

  # Train model if not provided
  if (is.null(model)) {
    res <- prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)
    df <- res$df
    model <- res$model
  }

  # Gather model statistics for testing, training, and all data
  mod_stats <- nm_modStats(df, model)

  # Variables to be used in resampling
  variables_resample <- setdiff(feature_names, c('hour', 'weekday', 'day_julian', 'date_unix'))

  # Normalize the entire data for mean calculation
  df_dew <- nm_normalise(df, model, feature_names, variables_resample, n_samples, replace=TRUE, aggregate=TRUE, seed=seed, n_cores=n_cores, verbose=FALSE)

  # Initialize the dataframe for rolling window results
  dfr <- data.frame(date = df_dew$date)

  df <- df %>% mutate(date_d = as.Date(date))

  # Define the rolling window range
  date_max <- max(df$date_d, na.rm = TRUE) - days(window_days - 1)
  date_min <- min(df$date_d, na.rm = TRUE) + days(window_days - 1)

  rolling_dates <- unique(df$date_d[df$date_d <= date_max])[seq(1, length(unique(df$date_d[df$date_d <= date_max])), by = rollingevery)]

  # Initialize the progress bar
  pb <- progress_bar$new(
    format = "  Rolling window :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(rolling_dates),
    clear = FALSE,
    width = 80
  )

  # Apply the rolling window approach
  start_time <- Sys.time()  # Initialize start time before the loop
  for (i in seq_along(rolling_dates)) {
    ds <- rolling_dates[i]

    dfa <- df %>%
      filter(date_d >= ds & date_d <= (ds + days(window_days - 1)))

    success <- FALSE
    tryCatch({
      # Normalize the data within the rolling window
      dfar <- nm_normalise(dfa, model, feature_names, variables_resample, n_samples, replace=TRUE, aggregate=TRUE, seed=seed, n_cores=n_cores, verbose=FALSE) %>%
        rename_with(~ paste0("rolling_", i), .cols = "normalised")

      # Concatenate the results
      dfr <- left_join(dfr, dfar %>% select(date, paste0("rolling_", i)), by = "date")

      # Update the progress bar
      pb$tick()

      success <- TRUE
    }, error = function(e) {
      cat(sprintf("%s: Error during normalization for rolling window %d from %s to %s: %s",
                 format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                 i, min(dfa$date), max(dfa$date), e$message))
    })

    if (!success) {
      Sys.sleep(10)  # Wait for 10 seconds before retrying
    }
  }

  # Calculate the mean and standard deviation for the rolling window
  df_dew <- df_dew %>%
    mutate(emi_mean = rowMeans(dfr[, -1], na.rm = TRUE),
           emi_std = apply(dfr[, -1], 1, sd, na.rm = TRUE))


  # Calculate the short-term and seasonal components
  df_dew <- df_dew %>%
    mutate(met_short = observed - emi_mean,
           met_season = emi_mean - normalised)

  return(list(df_dew = df_dew, mod_stats = mod_stats))
}
