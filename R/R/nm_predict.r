#' Predict Using Trained Model
#'
#' \code{nm_predict} generates predictions using a trained model on new data.
#'
#' @param object The trained model object.
#' @param newdata A data frame containing the new data for prediction.
#'
#' @return A vector of predicted values.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   outcome = rnorm(100)
#' )
#' trained_model <- lm(outcome ~ feature1 + feature2, data = df)
#' new_data <- data.frame(
#'   feature1 = rnorm(10),
#'   feature2 = rnorm(10)
#' )
#' predictions <- nm_predict(trained_model, df = new_data)
#' }
#' @export
nm_predict <- function(model, df) {
  # Predict values using the model
  value_predict <- as.vector(h2o::h2o.predict(model, h2o::as.h2o(df))$predict)

  return(value_predict)
}
