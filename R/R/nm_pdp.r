#' Generate Partial Dependence Plots (PDP)
#'
#' \code{nm_pdp} generates partial dependence plots for specified features using a trained model.
#'
#' @param df Data frame containing the input data.
#' @param model The trained model object.
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
#' pdp_results <- nm_pdp(df, model, varibales = c("feature1", "feature2"))
#' }
#' @export
nm_pdp <- function(df, model, variables = NULL, training_only = TRUE, grid.resolution = 20, n_cores = NULL) {

  feature_names <- nm_extract_feature_names(model)

  if (is.null(variables)) {
    variables <- feature_names
  }

  if (training_only) {
    df <- df %>% filter(set == "training")
  }

  n_cores <- ifelse(is.null(n_cores), parallel::detectCores() - 1, n_cores)

  cl <- makeCluster(n_cores, type = "SOCK")
  clusterEvalQ(cl, {
    library(h2o)
    library(pdp)
    library(dplyr)
    nm_init_h2o <- function(n_cores = NULL, max_mem_size = "16G") {
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
        h2o::h2o.init(nthreads = n_cores, max_mem_size = max_mem_size)
        h2o::h2o.no_progress()
      })
    }
    nm_init_h2o()
  })
  doSNOW::registerDoSNOW(cl)

  # Create a progress bar
  pb <- progress_bar$new(
    format = "  Processing [:bar] :percent eta: :eta",
    total = length(variables),
    width = 80
  )

  # Progress function
  progress <- function(n) pb$tick()
  opts <- list(progress = progress)

  results <- foreach(var = variables, .packages = c('pdp', 'h2o', 'dplyr'), .export = c("nm_pdp_worker", "nm_predict", "nm_init_h2o"), .options.snow = opts) %dopar% {
    nm_init_h2o()
    nm_pdp_worker(model, df, var, grid.resolution)
  }

  snow::stopCluster(cl)

  df_predict <- bind_rows(results)
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
