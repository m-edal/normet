#' Prepare and Train Model
#'
#' \code{nm_prepare_train_model} prepares the data and trains a model using AutoML.
#'
#' @param df Data frame containing the input data.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for training.
#' @param split_method The method for splitting data into training and testing sets.
#' @param fraction The proportion of the data to be used for training.
#' @param model_config A list containing configuration parameters for model training.
#' @param seed A random seed for reproducibility.
#' @param verbose Should the function print progress messages? Default is TRUE.
#'
#' @return The trained leader model from AutoML.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   target = rnorm(100)
#' )
#' res <- nm_prepare_train_model(df, value = "target", feature_names = c("feature1", "feature2"))
#' }
#' @export
nm_prepare_train_model <- function(df, value, feature_names, split_method, fraction, model_config, seed, verbose=TRUE) {

    vars <- setdiff(feature_names, c('date_unix', 'day_julian', 'weekday', 'hour'))

    # Prepare the data
    df <- nm_prepare_data(df, value=value, feature_names=vars, split_method=split_method, fraction=fraction, seed=seed)

    # Default model configuration parameters
    default_model_config <- list(
      max_models = 10,                  # Maximum number of models to train
      nfolds = 5,                       # Number of cross-validation folds
      max_mem_size = '16G',             # Maximum memory for H2O
      include_algos = c('GBM'),         # Algorithms to include (e.g., "GBM", "GLM", "DeepLearning", "DRF", "StackedEnsemble")
      save_model = TRUE,                # Whether to save the model
      model_name = 'automl',            # Name for the saved model
      model_path = './',                # Path to save the model
      predata_name = 'data_prepared',
      seed = seed,                      # Random seed for reproducibility
      verbose = verbose                 # Verbose output for progress
    )

    # Update default configuration with user-provided config
    if (!is.null(model_config)) {
      default_model_config <- modifyList(default_model_config, model_config)
    }

    # Train the model using AutoML
    auto_ml <- nm_train_model(df, value='value', variables=feature_names, model_config=default_model_config, seed=seed, verbose=verbose)

    if (default_model_config$save_model == TRUE) {
      write.csv(df, paste0(default_model_config$predata_name, '.csv'),row.names = FALSE)
    }

    return(list(df = df, model = auto_ml))
}
