#' Apply rolling window normalization with H2O
#'
#' \code{nm_rolling_dew} performs normalization on data using a rolling window approach with H2O models.
#'
#' @param df Data frame containing the input data.
#' @param model Pre-trained model for normalization. If not provided, a model will be trained.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for normalization.
#' @param variables_resample The names of the variables to be resampled for normalization.
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
#' h2o.init()
#' result <- nm_rolling_dew(df, value = "target", feature_names = c("feature1", "feature2"), window_size = 10, model_type = "h2o.gbm", seed = 12345)
#' }
#' @export
nm_rolling_dew <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, variables_resample = NULL, split_method = 'random',
                        fraction = 0.75, model_config = NULL, n_samples = 300, window_days = 14, rollingevery = 7, seed = 7654321, n_cores = NULL, verbose = TRUE) {
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

  # Create an initial dataframe to store observed values
  dfr <- df %>%
    select(date, value) %>%
    rename(observed = value)

  df <- df %>%
    mutate(date_d = as.Date(date))

  # Define the rolling window range
  date_max <- max(df$date_d) - days(window_days - 1)
  date_min <- min(df$date_d) + days(window_days - 1)

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
      dfar <- nm_normalise(dfa, model, feature_names = feature_names, variables_resample = variables_resample,
                          n_samples = n_samples, n_cores = n_cores, seed = seed, verbose = FALSE) %>%
        rename_with(~ paste0("rolling_", i), .cols = "normalised")

      # Concatenate the results
      dfr <- dfr %>% left_join(dfar %>% select(date, paste0("rolling_", i)), by = "date")

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
  dfr <- dfr %>%
    mutate(emi_mean = rowMeans(select(., starts_with("rolling_")), na.rm = TRUE),
           emi_std = apply(select(., starts_with("rolling_")), 1, sd, na.rm = TRUE))

  # Calculate the short-term and seasonal components
  dfr <- dfr %>%
    mutate(met_short = observed - emi_mean,
           met_season = emi_mean - observed)  # Assuming 'normalised' is the normalized column in dfr

  # Shutdown H2O after use
  # h2o.shutdown(prompt = FALSE)
  # loginfo('H2O shutdown complete')

  return(list(dfr = dfr, mod_stats = mod_stats))
}
