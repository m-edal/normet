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
#' @param n_samples Number of samples to generate for normalisation. Default is 300.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param importance_ascending Logical indicating whether to sort feature importances in ascending order. Default is FALSE.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param memory_save Logical indicating whether to save memory by processing each sample independently.
#'   If \code{TRUE}, resampling and prediction are done in memory-efficient batches. If \code{FALSE}, all samples
#'   are generated and processed at once, which uses more memory. Default is FALSE.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A data frame with decomposed components.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(lubridate)
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_decom_met(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, seed = 12345)
#' }
#' @export
nm_decom_met <- function(df = NULL, model = NULL, value = 'value', feature_names = NULL, split_method = 'random', fraction = 0.75,
                      model_config = NULL, n_samples = 300, seed = 7654321, importance_ascending = FALSE, n_cores = NULL, memory_save = FALSE, verbose = TRUE) {

  # Check if h2o is already initialized
  nm_init_h2o(n_cores)

  set.seed(seed)

  # Train model if not provided
  if (is.null(model)) {
    df_model <- prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)
    df <- df_model$df
    model <- df_model$model
  } else if (!"value" %in% colnames(df)) {
    vars <- setdiff(feature_names, c('date_unix', 'day_julian', 'weekday', 'hour'))
    df <- nm_prepare_data(df, value, feature_names = vars, split_method = split_method, fraction = fraction, seed = seed)
  }

  # Extract and sort feature importance
  feature_names_sorted <- nm_extract_feature_names(model, importance_ascending = importance_ascending)

  # Initialize the dataframe for decomposed components
  df_dew <- df %>% select(date, value) %>% rename(observed = value)

  # Create feature exclusion list
  met_list <- c('deweathered', feature_names_sorted[!feature_names_sorted %in% c('hour', 'weekday', 'day_julian', 'date_unix')])

  var_names <- feature_names_sorted[!feature_names_sorted %in% c('hour', 'weekday', 'day_julian', 'date_unix')]

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

    var_to_exclude <- met_list[i]  # Get the current variable to exclude
    pb$tick()  # Update progress bar

    var_names <- setdiff(var_names, var_to_exclude)  # Exclude the current variable from the variable names

    success <- FALSE
    retries <- 3  # Set the number of retries

    while (!success && retries > 0) {
      tryCatch({
        # Normalize the data, excluding the current variable
        df_dew_temp <- nm_normalise(df, model, feature_names = feature_names,
                                    variables_resample = var_names, n_samples = n_samples,
                                    n_cores = n_cores, seed = seed, memory_save = memory_save, verbose = FALSE)

        # Check if the normalization produced any results
        if (nrow(df_dew_temp) == 0) {
          stop(paste("Normalization failed for variable:", var_to_exclude))  # Stop with an error message if no rows
        }

        # Store the normalized results
        df_dew[[var_to_exclude]] <- df_dew_temp$normalised

        success <- TRUE  # Set success to TRUE to exit the retry loop

      }, error = function(e) {
        # Print an error message if normalization fails
        cat(sprintf("%s: Error during normalization for variable '%s': %s\n",
                    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                    var_to_exclude, e$message))

        retries <- retries - 1  # Decrease the retry count
        if (retries > 0) {
          cat(sprintf("Retrying... %d attempts left.\n", retries))  # Indicate remaining attempts
          Sys.sleep(10)  # Wait for 10 seconds before retrying
        } else {
          cat("Failed after 3 attempts. Moving to the next variable.\n")  # Indicate failure after retries
          df_dew[[var_to_exclude]] <- NULL  # Optionally set to NULL if all attempts fail
        }
      })
    }
  }

  # Initialize result with the original data frame
  result <- df_dew

  # Loop through the relevant features and compute the adjusted values
  for (i in seq_along(feature_names_sorted)) {
    param <- feature_names_sorted[i]
    if (!param %in% c('hour', 'weekday', 'day_julian', 'date_unix')) {
      if (i > 1) {
        result[[param]] <- df_dew[[param]] - df_dew[[met_list[i - 1]]]
      } else {
        result[[param]] <- df_dew[[param]] - df_dew[['deweathered']]
      }
    }
  }

  # Compute 'met_noise' column
  result[['met_noise']] <- df_dew[['observed']] - df_dew[[met_list[length(met_list)]]]

  return(result)
}
