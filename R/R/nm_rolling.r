#' Apply rolling window meteorological normalisation
#'
#' \code{nm_rolling} performs meteorological normalisation on data using a rolling window approach.
#'
#' @param df Data frame containing the input data.
#' @param model Pre-trained model for normalisation. If not provided, a model will be trained.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for normalisation.
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param model_config A list containing configuration parameters for model training.
#' @param n_samples Number of times to sample the data for normalisation. Default is 300.
#' @param window_days The size of the rolling window in days. Default is 14.
#' @param rolling_every The interval at which the rolling window is applied in days. Default is 7.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the normalised data frame and model statistics.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(h2o)
#' library(purrr)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   target = rnorm(100)
#' )
#' result <- nm_rolling(df, value = "target", feature_names = c("feature1", "feature2"), variables_resample = c("feature1"), window_size = 10, model_type = "h2o.gbm", seed = 12345)
#' }
#' @export
nm_rolling <- function(df = NULL, model = NULL, value = NULL, feature_names = NULL, variables_resample = NULL, split_method = 'random', fraction = 0.75,
                       model_config = NULL, n_samples = 300, window_days = 14, rolling_every = 7, seed = 7654321, n_cores = NULL, verbose = TRUE) {
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

  df <- df %>% mutate(date_d = as.Date(date))

  # Define the rolling window range
  date_max <- max(df$date_d, na.rm = TRUE) - days(window_days - 1)
  rolling_dates <- unique(df$date_d[df$date_d <= date_max])[seq(1, length(unique(df$date_d[df$date_d <= date_max])), by = rolling_every)]

  # Initialize the progress bar
  pb <- progress_bar$new(
    format = "  Rolling window :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(rolling_dates),
    clear = FALSE,
    width = 80
  )

  # Initialize a list to store the results of each rolling window
  rolling_results <- list()

  # Apply the rolling window approach directly on df_dew
  start_time <- Sys.time()
  for (i in seq_along(rolling_dates)) {
    ds <- rolling_dates[i]

    dfa <- df %>%
      filter(date_d >= ds & date_d <= (ds + days(window_days - 1)))

    success <- FALSE
    tryCatch({
      # Normalize the data within the rolling window
      dfar <- nm_normalise(dfa, model, feature_names, variables_resample, n_samples, replace=TRUE, aggregate=TRUE, seed=seed, n_cores=n_cores, verbose=FALSE)

      # Rename the 'normalised' column to include the rolling window index
      dfar <- dfar %>%
        select(date, normalised) %>%
        rename(!!paste0('rolling_', i) := normalised)

      # Store the results of the current rolling window
      rolling_results[[i]] <- dfar

      # Update the progress bar
      pb$tick()

      success <- TRUE
    }, error = function(e) {
      cat(sprintf("%s: Error during normalization for rolling window %d from %s to %s: %s",
                  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                  i, min(dfa$date), max(dfa$date), e$message))
      rolling_results[[i]] <- NULL  # Ensure the list has the same length
    })

    if (!success) {
      Sys.sleep(10)
    }
  }

  # Filter out NULL results
  rolling_results <- rolling_results[!sapply(rolling_results, is.null)]

  # Merge all rolling window results by 'date'
  combined_results <- rolling_results %>%
   reduce(left_join, by = "date")

  return(list(df_dew = combined_results, mod_stats = mod_stats))
}
