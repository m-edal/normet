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
#' @param n_samples Number of samples to generate for normalisation. Default is 300.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param memory_save Logical indicating whether to save memory by processing each sample independently.
#'   If \code{TRUE}, resampling and prediction are done in memory-efficient batches. If \code{FALSE}, all samples
#'   are generated and processed at once, which uses more memory. Default is FALSE.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return The decomposed data frame.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(lubridate)
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_decom_emi(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, seed = 12345)
#' }
#' @export
nm_decom_emi <- function(df = NULL, model = NULL, value = 'value', feature_names = NULL, split_method = 'random', fraction = 0.75,
                      model_config = NULL, n_samples = 300, seed = 7654321, n_cores = NULL, memory_save = FALSE, verbose = TRUE) {

  # Check if h2o is already initialized
  nm_init_h2o(n_cores)

  set.seed(seed)

  # Train model if not provided
  if (is.null(model)) {
    df_model <- nm_prepare_train_model(df, value, feature_names, split_method, fraction, model_config, seed, verbose)
    df <- df_model$df
    model <- df_model$model
  } else if (!"value" %in% colnames(df)) {
    vars <- setdiff(feature_names, c('date_unix', 'day_julian', 'weekday', 'hour'))
    df <- nm_prepare_data(df, value, feature_names = vars, split_method = split_method, fraction = fraction, seed = seed)
  }

  # Initialize the dataframe for decomposed components
  df_dew <- df %>% select(date, value) %>% rename(observed = value)

  # Default logic for CPU cores
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  # Initialize progress bar
  pb <- progress_bar$new(
    format = "  Iterative subtracting :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(c('base', 'date_unix', 'day_julian', 'weekday', 'hour')),
    clear = FALSE,
    width = 80
  )

  # Decompose the time series by excluding different features
  var_names <- feature_names
  start_time <- Sys.time()  # Initialize start time before the loop

  # Define the variables to exclude
  vars_to_exclude <- c('date_unix', 'day_julian', 'weekday', 'hour')

  # Find the intersection between var_names and vars_to_exclude
  vars_intersect <- intersect(var_names, vars_to_exclude)

  for (var_to_exclude in c('base', vars_intersect)) {

    pb$tick()  # Update progress bar

    # Remove the current variable from var_names
    var_names <- setdiff(var_names, var_to_exclude)

    success <- FALSE
    retries <- 3  # Set the number of retries

    while (!success && retries > 0) {
      tryCatch({
        # Normalize the data, excluding the current variable
        df_dew_temp <- nm_normalise(df, model, feature_names = feature_names,
                                    variables_resample = var_names, n_samples = n_samples,
                                    n_cores = n_cores, seed = seed, memory_save = memory_save, verbose = FALSE)

        # Store the normalized data for the excluded variable
        df_dew[[var_to_exclude]] <- df_dew_temp$normalised

        success <- TRUE  # If successful, break the loop
      }, error = function(e) {
        cat(sprintf("%s: Error during normalization for variable '%s': %s\n",
                    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                    var_to_exclude, e$message))

        retries <- retries - 1  # Decrease the retry count
        if (retries > 0) {
          cat(sprintf("Retrying... %d attempts left.\n", retries))
          Sys.sleep(10)  # Wait for 10 seconds before retrying
        } else {
          cat("Failed after 3 attempts. Moving to the next variable.\n")
          df_dew[[var_to_exclude]] <- NULL  # Optionally, set to NULL if failure persists
        }
      })
    }
  }

  # Adjust the decomposed components to create deweathered values
  result <- df_dew

  # Loop over the sorted time variables and adjust the components
  for (i in seq_along(vars_intersect)) {
    var_current <- vars_intersect[i]

    if (i == 1) {
      # For the first variable, subtract 'base' and add the mean of 'base'
      result[[var_current]] <- df_dew[[var_current]] - df_dew$base + mean(df_dew$base, na.rm = TRUE)
    } else {
      # For the rest, subtract the previous variable in the list
      var_previous <- vars_intersect[i - 1]
      result[[var_current]] <- df_dew[[var_current]] - df_dew[[var_previous]]
    }
  }

  # Set the last variable in vars_intersect as 'deweathered'
  if (length(vars_intersect) > 0) {
    result$deweathered <- df_dew[[vars_intersect[length(vars_intersect)]]]
  }

  # Calculate 'emi_noise' based on 'base'
  result$emi_noise <- df_dew$base - mean(df_dew$base, na.rm = TRUE)

  # Return the result with adjusted decomposed components
  return(result)
}
