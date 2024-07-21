#' Perform All Steps for Meteorological Normalization with Uncertainty Estimation
#'
#' \code{nm_do_all_unc} performs the entire process of training multiple models, normalizing the data, and collecting model statistics with uncertainty estimation.
#'
#' @param df Data frame containing the input data.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for training and normalization.
#' @param variables_resample The names of the variables to be resampled for normalization. Default is NULL (all feature names except date_unix).
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param model_config A list containing configuration parameters for model training.
#' @param n_samples Number of samples to generate for normalization. Default is 300.
#' @param n_models Number of models to train for uncertainty estimation. Default is 10.
#' @param confidence_level The confidence level for uncertainty estimation. Default is 0.95.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param weather_df Optional data frame containing weather data for resampling.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A list containing the normalized data frame with uncertainty estimation and model statistics.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(lubridate)
#' library(progress)
#' df <- data.frame(date = Sys.time() + seq(1, 100, by = 1),
#'                  pollutant = rnorm(100), temp = rnorm(100), humidity = rnorm(100))
#' result <- nm_do_all_unc(df, value = "pollutant", feature_names = c("temp", "humidity"), n_samples = 300, n_models = 10, seed = 12345)
#' }
#' @export
nm_do_all_unc <- function(df = NULL, value = NULL, feature_names = NULL, variables_resample = NULL, split_method = 'random', fraction = 0.75,
                       model_config = NULL, n_samples = 300, n_models = 10, confidence_level = 0.95, seed = 7654321, n_cores = NULL, weather_df = NULL, verbose = TRUE) {

  # Check if h2o is already initialized
  nm_init_h2o(n_cores)

  set.seed(seed)
  random_seeds <- sample(1:1000000, n_models, replace = FALSE)

  df_dew_list <- list()
  mod_stats_list <- list()

  # Determine number of CPU cores to use
  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  start_time <- Sys.time()

  # Initialize the progress bar
  pb <- progress_bar$new(
    format = "  Model :current/:total [:bar] :percent :elapsedfull ETA: :eta",
    total = length(random_seeds),
    clear = FALSE,
    width = 80
  )

  for (i in seq_along(random_seeds)) {
    seed <- random_seeds[i]
    success <- FALSE

    tryCatch({
      res <- nm_do_all(df, value = value, feature_names = feature_names,
                    variables_resample = variables_resample,
                    split_method = split_method, fraction = fraction,
                    model_config = model_config,
                    n_samples = n_samples, seed = seed, n_cores = n_cores,
                    weather_df = weather_df, verbose = FALSE)

      df_dew0 <- res$df_dew %>%
        select(date, normalised = starts_with("normalised")) %>%
        rename_with(~ paste0(., "_", seed), starts_with("normalised"))
      df_dew_list[[i]] <- df_dew0
      mod_stats0 <- res$mod_stats %>% mutate(seed = seed)
      mod_stats_list[[i]] <- mod_stats0

      success <- TRUE
    }, error = function(e) {
      cat(sprintf("%s: Error during model %d/%d: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), i, length(random_seeds), e$message))
    })

    if (success) {
      pb$tick()
    } else {
      Sys.sleep(10)  # Wait for 10 seconds before retrying
    }
  }

  df_dew <- df_dew_list %>% reduce(left_join, by = "date")
  mod_stats <- bind_rows(mod_stats_list)

  df_dew <- df_dew %>% mutate(
    mean = rowMeans(select(., starts_with("normalised_")), na.rm = TRUE),
    std = apply(select(., starts_with("normalised_")), 1, sd, na.rm = TRUE),
    median = apply(select(., starts_with("normalised_")), 1, median, na.rm = TRUE),
    lower_bound = apply(select(., starts_with("normalised_")), 1, quantile, probs = (1 - confidence_level) / 2, na.rm = TRUE),
    upper_bound = apply(select(., starts_with("normalised_")), 1, quantile, probs = 1 - (1 - confidence_level) / 2, na.rm = TRUE)
  )

  weighted_R2 <- mod_stats %>%
    filter(set == 'testing') %>%
    mutate(R2 = as.numeric(as.character(R2))) %>%
    filter(!is.na(R2) & !is.infinite(R2)) %>%
    reframe(
      min_R2 = min(R2, na.rm = TRUE),
      max_R2 = max(R2, na.rm = TRUE),
      normalised_R2 = (R2 - min(R2, na.rm = TRUE)) / (max(R2, na.rm = TRUE) - min(R2, na.rm = TRUE)),
      weighted_R2 = normalised_R2 / sum(normalised_R2, na.rm = TRUE)
    ) %>%
    pull(weighted_R2)

  df_dew_weighted <- df_dew %>% select(starts_with("normalised_")) %>% mutate(across(everything(), ~ . * weighted_R2))
  df_dew <- df_dew %>% mutate(weighted = rowSums(df_dew_weighted, na.rm = TRUE))

  # Shutdown H2O after use
  #h2o.shutdown(prompt = FALSE)
  #loginfo('H2O shutdown complete')

  return(list(df_dew = df_dew, mod_stats = mod_stats))
}
