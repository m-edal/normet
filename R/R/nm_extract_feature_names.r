#' Extract and Sort Feature Names by Importance
#'
#' \code{extract_feature_names} extracts feature names from an H2O model and sorts them by their importance.
#'
#' This function uses H2O's \code{varimp} function to extract feature importance from the model and
#' returns the feature names sorted by their relative importance. You can control whether the
#' sorting is in ascending or descending order using the \code{importance_ascending} argument.
#'
#' @param model The trained H2O model object.
#' @param importance_ascending A logical value indicating whether to sort feature names in ascending order
#' of importance. Default is \code{FALSE} (descending order).
#'
#' @return A vector of sorted feature names based on their importance.
#'
#' @examples
#' \dontrun{
#' library(h2o)
#' h2o.init()
#' df <- as.h2o(iris)
#' model <- h2o.gbm(x = 1:4, y = 5, training_frame = df)
#' feature_names <- extract_feature_names(model, importance_ascending = TRUE)
#' print(feature_names)
#' }
#' @export
nm_extract_feature_names <- function(model, importance_ascending = FALSE) {
  # Extract variable importance using H2O's varimp function
  varimp_df <- as.data.frame(h2o.varimp(model))

  # Check if variable importance data is available
  if (nrow(varimp_df) == 0) {
    stop("The H2O model does not have variable importance information.")
  }

  # Extract feature names and sort by relative importance
  feature_names <- varimp_df$variable
  feature_names <- feature_names[order(varimp_df$relative_importance, decreasing = !importance_ascending)]

  # Return the sorted feature names
  return(feature_names)
}
