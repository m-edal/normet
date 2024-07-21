#' Generate Resampled Data
#'
#' \code{nm_generate_resampled} resamples specified variables in the input DataFrame.
#'
#' @param df_batch Input data frame batch.
#' @param variables_resample A vector of variables to be resampled.
#' @param replace Logical indicating whether to sample with replacement.
#' @param seed A random seed for reproducibility.
#' @param weather_df Optional data frame containing weather data for resampling.
#'
#' @return A data frame with resampled variables.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(temp = rnorm(100), humidity = rnorm(100))
#' weather_data <- data.frame(temp = rnorm(100), humidity = rnorm(100))
#' resampled_df <- nm_generate_resampled(df, variables_resample = c("temp", "humidity"), replace = TRUE, seed = 12345, weather_df = weather_data)
#' }
#'
#' @export
nm_generate_resampled <- function(df_batch, variables_resample, replace, seed, weather_df=NULL) {
    # Set the random seed for reproducibility
    library(dplyr)
    set.seed(seed)

    # Resample variables
    if (!is.null(weather_df) && nrow(weather_df) == nrow(df_batch)) {
        # Randomly sample indices from the input DataFrame
        index_rows <- sample(nrow(df_batch), size=nrow(df_batch), replace=replace)
        # Resample the specified variables using the sampled indices
        df_batch[variables_resample] <- df_batch[variables_resample][index_rows, ]
    } else {
        # Sample meteorological parameters from the provided weather DataFrame
        sampled_meteorological_params <- weather_df[variables_resample] %>%
            dplyr::sample_n(nrow(weather_df), replace=replace)
        # Use the sampled parameters to resample the specified variables in the input DataFrame
        df_batch[variables_resample] <- sampled_meteorological_params %>%
            dplyr::sample_n(nrow(df_batch), replace=replace)
    }

    # Add seed column to the batch
    df_batch$seed <- seed

    return(df_batch)
}

#' Normalize the Dataset Using the Trained Model
#'
#' \code{nm_normalise} normalizes the dataset using the trained model and generates resampled data in parallel.
#'
#' @param df Input data frame.
#' @param model The trained model object.
#' @param feature_names The names of the features used for normalization.
#' @param variables_resample The names of the variables to be resampled for normalization.
#' @param n_samples Number of samples to generate. Default is 300.
#' @param replace Logical indicating whether to sample with replacement. Default is TRUE.
#' @param aggregate Logical indicating whether to aggregate the results. Default is TRUE.
#' @param seed A random seed for reproducibility. Default is 7654321.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#' @param weather_df Optional data frame containing weather data for resampling.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return A data frame containing the normalized data.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(lubridate)
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   temp = rnorm(100),
#'   humidity = rnorm(100)
#' )
#' model <- lm(temp ~ humidity, data = df)
#' normalized_df <- nm_normalise(df, model, feature_names = c("temp", "humidity"), n_samples = 300, replace = TRUE, seed = 12345)
#' }
#' @export
nm_normalise <- function(df=NULL, model=NULL, feature_names=NULL, variables_resample=NULL, n_samples=300, replace=TRUE,
                         aggregate=TRUE, seed=7654321, n_cores=NULL, weather_df=NULL, verbose=TRUE) {
    # Process input DataFrames
    df <- nm_process_df(df, variables_col=c(feature_names, 'value'))

    # If no weather_df is provided, use df as the weather data
    if (is.null(weather_df)) {
        weather_df <- df
    }

    # Use all variables except the trend term
    if (is.null(variables_resample)) {
        variables_resample <- feature_names[feature_names != 'date_unix']
    }

    # Check if all variables are in the DataFrame
    if (!all(variables_resample %in% colnames(weather_df))) {
        stop("The input weather_df does not contain all variables within `variables_resample`.")
    }

    # Generate random seeds for parallel processing
    set.seed(seed)
    random_seeds <- sample(1:1000000, n_samples, replace=FALSE)

    # Determine number of CPU cores to use
    n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

    # Initialize progress bar only if verbose is TRUE
    if (verbose) {
        pb <- progress_bar$new(
            format = "  Parallel normalisation :current/:total [:bar] :percent :elapsedfull ETA: :eta",
            total = n_samples,
            clear = FALSE,
            width = 80
        )
    }

    # Perform data generation using parallel processing
    cluster <- makeCluster(n_cores)
    registerDoSNOW(cluster)

    # Define a progress function to be called in the main thread
    progress <- function(n) {
        if (verbose) pb$tick()
    }

    # Define a function to update the progress bar
    opts <- list(progress = progress)

    # Ensure model is available in each worker
    clusterEvalQ(cluster, {
        library(h2o)
        nm_init_h2o <- function(n_cores = NULL, min_mem_size = "4G", max_mem_size = "16G") {
          if (is.null(n_cores)) {
            n_cores <- parallel::detectCores() - 1
          }

          tryCatch({
            conn <- h2o.getConnection()
            if (!h2o.clusterIsUp()) {
              stop("H2O cluster is not up")
            }
          }, error = function(e) {
            message("H2O is not running. Starting H2O...")
            h2o::h2o.init(nthreads = n_cores, min_mem_size = min_mem_size, max_mem_size = max_mem_size)
            h2o::h2o.no_progress()
          })
        }
        nm_init_h2o()
    })

    generated_dfs <- foreach(i = 1:n_samples, .packages = c('dplyr', 'tidyr', 'h2o'),
                           .export = c('nm_generate_resampled','nm_predict', 'progress'), .options.snow = opts) %dopar% {
    result <- tryCatch({
      nm_init_h2o()
      df_batch <- nm_generate_resampled(df=df, variables_resample=variables_resample, replace=replace,
                                          seed=random_seeds[i], weather_df=weather_df)
      value_predict <- nm_predict(model, df_batch)
      data.frame(
        date = df_batch$date,
        observed = df_batch$value,
        normalised = value_predict,
        seed = df_batch$seed
      )
    }, error = function(e) {
      message("Error during parallel execution: ", e$message)
      NULL
    })
    result
    }

    stopCluster(cluster)

    generated_dfs <- generated_dfs[!sapply(generated_dfs, is.null)]

    # Aggregate results if needed
    if (aggregate) {
        if (verbose) {
            cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ": Aggregating", n_samples, "predictions...\n")
        }
        df_result <- bind_rows(generated_dfs) %>%
            group_by(date) %>%
            summarise(observed = mean(observed), normalised = mean(normalised))
    } else {
        df_result <- bind_rows(generated_dfs) %>%
          pivot_wider(names_from = seed, values_from = normalised)

        if (verbose) {
            cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ": Concatenated", n_samples, "predictions...\n")
        }
    }

    return(df_result)
}
