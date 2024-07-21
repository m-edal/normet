#' Generate Partial Dependence Plots (PDP)
#'
#' \code{nm_pdp} generates partial dependence plots for specified features using a trained model.
#'
#' @param df Data frame containing the input data.
#' @param model The trained model object.
#' @param feature_names The names of the features used for training the model.
#' @param variables A vector of feature names for which PDPs are to be generated. Default is NULL (all feature names will be used).
#' @param training_only Logical indicating whether to use only training data for generating PDPs. Default is TRUE.
#' @param grid.resolution The number of points to evaluate on the grid for each feature. Default is 20.
#' @param n_cores Number of CPU cores to use for parallel processing. Default is system's total minus one.
#'
#' @return A data frame containing the partial dependence values.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   set = rep(c("train", "test"), each = 50),
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100)
#' )
#' model <- lm(feature1 ~ feature2, data = df)
#' pdp_results <- nm_pdp(df, model, feature_names = c("feature1", "feature2"))
#' }
#' @export
nm_pdp <- function(df, model, feature_names, variables = NULL, training_only = TRUE, grid.resolution = 20, n_cores = NULL) {
  if (is.null(variables)) {
    variables <- feature_names
  }

  if (training_only) {
    df <- df[df$set == "training", ]
  }

  X_train <- df[, feature_names, drop = FALSE]

  if (is.null(n_cores)) {
    n_cores <- parallel::detectCores() - 1
  }

  cl <- parallel::makeCluster(n_cores)
  doParallel::registerDoParallel(cl)

  results <- foreach(var = variables, .packages = c('pdp', 'h2o'), .export = c("nm_pdp_worker", "nm_predict", "nm_init_h2o")) %dopar% {
    nm_init_h2o()
    nm_pdp_worker(model, X_train, var, grid.resolution)
  }

  parallel::stopCluster(cl)

  df_predict <- dplyr::bind_rows(results)
  return(df_predict)
}

#' Worker function for generating PDP
#'
#' \code{nm_pdp_worker} is a worker function that generates partial dependence values for a specific feature.
#'
#' @param model The trained model object.
#' @param df Data frame containing the input data.
#' @param variable The feature name for which PDP is to be generated.
#' @param grid.resolution The number of points to evaluate on the grid for the feature.
#'
#' @return A data frame containing the partial dependence values for the specified feature.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(pdp)
#' df <- data.frame(
#'   set = rep(c("train", "test"), each = 50),
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100)
#' )
#' model <- lm(feature1 ~ feature2, data = df)
#' pdp_result <- nm_pdp_worker(df, model, feature = "feature2")
#' }
#' @export
nm_pdp_worker <- function(model, df, variable, grid.resolution) {
  nm_init_h2o()
  pd_results <- pdp::partial(object = model, pred.var = variable, train = df, pred.fun = nm_predict, grid.resolution = grid.resolution)
  pd_results$var <- variable

  df_predict <- data.frame(
    var = variable,
    id = pd_results$yhat.id,
    var_value = pd_results[[variable]],
    pdp_value = pd_results$yhat
  )

  return(df_predict)
}
