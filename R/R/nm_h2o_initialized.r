#' Initialize H2O
#'
#' \code{nm_init_h2o} initializes the H2O cluster with specified settings.
#'
#' @param n_cores The number of CPU cores to use. Default is system's total minus one.
#' @param min_mem_size The minimum memory size for H2O. Default is "4G".
#' @param max_mem_size The maximum memory size for H2O. Default is "16G".
#'
#' @examples
#' \dontrun{
#' library(h2o)
#' nm_init_h2o(n_cores = 4, min_mem_size = "4G", max_mem_size = "16G")
#' }
#' @export
nm_init_h2o <- function(n_cores = NULL, min_mem_size = "4G", max_mem_size = "16G") {
  if (is.null(n_cores)) {
    n_cores <- parallel::detectCores() - 1
  }

  tryCatch({
    conn <- h2o::h2o.getConnection()
    if (!h2o::h2o.clusterIsUp()) {
      stop("H2O cluster is not up")
    }
  }, error = function(e) {
    message("H2O is not running. Starting H2O...")
    h2o::h2o.init(nthreads = n_cores, min_mem_size = min_mem_size, max_mem_size = max_mem_size)
    h2o::h2o.no_progress()
  })
}
