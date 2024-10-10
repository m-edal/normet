#' Save Trained H2O Model
#'
#' \code{nm_save_h2o} saves a trained H2O model to a specified directory.
#'
#' This function ensures the output directory exists, saves the H2O model, and renames
#' the saved model file to the desired filename.
#'
#' @param model The trained H2O model object.
#' @param path A string specifying the directory path where the model will be saved. Default is './'.
#' @param filename A string specifying the desired filename for the saved model. Default is 'automl'.
#'
#' @return A string indicating the path of the saved model.
#'
#' @examples
#' \dontrun{
#' h2o.init()
#' model <- h2o.gbm(x = c('feature1', 'feature2'), y = 'outcome', training_frame = df)
#' nm_save_h2o(model, path = './models', filename = 'my_model')
#' }
#' @export
nm_save_h2o <- function(model, path = './', filename = 'automl') {
  # Ensure the output directory exists
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)  # Create directory if it doesn't exist
  }

  # Save the model to the specified directory
  model_path <- h2o::h2o.saveModel(model, path = path, force = TRUE)
  new_model_path <- file.path(path, filename)

  # Rename the model file to the desired filename
  file.rename(model_path, new_model_path)

  return(new_model_path)  # Return the new model path
}

#' Load Saved H2O Model
#'
#' \code{nm_load_h2o} loads a previously saved H2O model from disk.
#'
#' This function loads the H2O model from the specified directory and filename.
#'
#' @param path A string specifying the directory path where the model is saved. Default is './'.
#' @param filename A string specifying the name of the saved model file. Default is 'automl'.
#'
#' @return The loaded H2O model object.
#'
#' @examples
#' \dontrun{
#' h2o.init()
#' model <- nm_load_h2o(path = './models', filename = 'my_model')
#' }
#' @export
nm_load_h2o <- function(path = './', filename = 'automl') {
  # Construct the full path to the model file
  model_path <- file.path(path, filename)

  # Load the H2O model
  model <- h2o.loadModel(model_path)

  return(model)  # Return the loaded model
}
