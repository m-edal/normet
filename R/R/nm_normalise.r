#' Generate Resampled Data
#'
#' \code{nm_generate_resampled} resamples specified variables in the input DataFrame.
#'
#' @param df Input data frame batch.
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
nm_generate_resampled <- function(df, variables_resample, replace, seed, weather_df) {

    # Set the random seed for reproducibility
    set.seed(seed)

    # Directly sample from the weather DataFrame
    df[variables_resample] <- weather_df[variables_resample] %>%
            slice(sample(nrow(weather_df), size = nrow(df), replace = replace))

    # Add seed column to the batch
    df$seed <- seed
    return(df)
}


#' Normalise the Dataset Using the Trained Model
#'
#' \code{nm_normalise} normalises the dataset using the trained model and generates resampled data in parallel.
#' It supports memory-efficient processing with the \code{memory_save} option.
#'
#' @param df Input data frame containing the data to be normalised.
#' @param model The trained model object (e.g., H2O model).
#' @param feature_names A character vector of feature names used for normalisation.
#' @param variables_resample A character vector of variable names to be resampled for normalisation.
#'   Defaults to all features except 'date_unix'.
#' @param n_samples Integer specifying the number of samples to generate. Default is 300.
#' @param replace Logical indicating whether to sample with replacement during resampling. Default is TRUE.
#' @param aggregate Logical indicating whether to aggregate the results across resamples. Default is TRUE.
#' @param seed An integer seed for random number generation to ensure reproducibility. Default is 7654321.
#' @param n_cores Integer specifying the number of CPU cores to use for parallel processing. Default is all cores minus one.
#' @param weather_df Optional data frame containing weather data for resampling. If \code{NULL}, the input \code{df} is used.
#' @param memory_save Logical indicating whether to save memory by processing each sample independently.
#'   If \code{TRUE}, resampling and prediction are done in memory-efficient batches. If \code{FALSE}, all samples
#'   are generated and processed at once, which uses more memory. Default is FALSE.
#' @param verbose Logical indicating whether to print progress messages. Default is TRUE.
#'
#' @return A data frame containing normalised predictions. If \code{aggregate} is TRUE, the output includes the mean
#'   of observed and normalised values for each date. Otherwise, the output includes a wide data frame with
#'   normalised predictions for each resample.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(lubridate)
#' library(h2o)
#'
#' df <- data.frame(
#'   date = Sys.time() + seq(1, 100, by = 1),
#'   temp = rnorm(100),
#'   humidity = rnorm(100)
#' )
#'
#' # Example: Train a simple model using H2O
#' h2o.init()
#' h2o_df <- as.h2o(df)
#' model <- h2o.glm(x = c("humidity"), y = "temp", training_frame = h2o_df)
#'
#' # Normalise the dataset using the trained model
#' normalised_df <- nm_normalise(
#'   df, model, feature_names = c("temp", "humidity"),
#'   n_samples = 300, replace = TRUE, seed = 12345, memory_save = TRUE
#' )
#' }
#' @export
nm_normalise <- function(df = NULL, model = NULL, feature_names = NULL, variables_resample = NULL, n_samples = 300,
                         replace = TRUE, aggregate = TRUE, seed = 7654321, n_cores = NULL,
                         weather_df = NULL, memory_save = FALSE, verbose = TRUE) {
    # Process input DataFrames
    df <- df %>%
        nm_process_date() %>%
        nm_check_data(feature_names, 'value')

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
    random_seeds <- sample(1:1000000, n_samples, replace = FALSE)

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
    cl <- snow::makeCluster(n_cores, type = "SOCK")
    doSNOW::registerDoSNOW(cl)

    # Define a progress function to be called in the main thread
    progress <- function(n) {
        if (verbose) pb$tick()
    }

    # Define a function to update the progress bar
    opts <- list(progress = progress)

    # Ensure model is available in each worker
    clusterEvalQ(cl, {
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

    # Memory saving approach
    if (memory_save) {
        # Process data in batches using parallel processing
        generated_dfs <- foreach(i = 1:n_samples, .packages = c('dplyr', 'tidyr', 'h2o'),
                               .export = c('nm_generate_resampled', 'nm_predict', 'progress'),
                               .options.snow = opts) %dopar% {
            result <- tryCatch({
                nm_init_h2o()
                df_batch <- nm_generate_resampled(df = df, variables_resample = variables_resample, replace = replace,
                                                  seed = random_seeds[i], weather_df = weather_df)
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
    } else {
        # Process all samples at once (non-memory-saving method)
        df_resampled_list <- foreach(i = 1:n_samples, .packages = c('dplyr', 'tidyr', 'h2o'),
                                   .export = c('nm_generate_resampled', 'progress'), .options.snow = opts) %dopar% {
            tryCatch({
                nm_generate_resampled(df = df, variables_resample = variables_resample, replace = replace,
                                      seed = random_seeds[i], weather_df = weather_df)
            }, error = function(e) {
                message("Error during resampling: ", e$message)
                NULL
            })
        }
        # Filter out NULL results
        df_resampled_list <- df_resampled_list[!sapply(df_resampled_list, is.null)]

        # Make predictions on the entire resampled data
        df_all_resampled <- bind_rows(df_resampled_list)
        predictions <- nm_predict(model, df_all_resampled)
        generated_dfs <- list(data.frame(
            date = df_all_resampled$date,
            observed = df_all_resampled$value,
            normalised = predictions,
            seed = df_all_resampled$seed
        ))
    }

    snow::stopCluster(cl)

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
