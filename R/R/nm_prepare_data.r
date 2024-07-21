#' Process DataFrame for Model Training
#'
#' \code{nm_process_df} processes the input DataFrame by checking for date in row names, identifying
#' the date column, and selecting relevant features.
#'
#' @param df Input data frame.
#' @param variables_col A vector of column names to be selected along with the date column.
#'
#' @return Processed data frame with selected columns.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   date = Sys.time() + seq(1, 100, by = 1)
#' )
#' processed_df <- nm_process_df(df, variables_col = c("feature1", "feature2"))
#' }
#' @export
nm_process_df <- function(df, variables_col) {

  # Check if the date is in the row names
  if (inherits(row.names(df), "POSIXct")) {
    df <- df %>% tibble::rownames_to_column(var = "date")
  }

  # Identify POSIXct date columns
  time_columns <- names(df)[sapply(df, inherits, what = "POSIXct")]

  if (length(time_columns) == 0) {
    stop("No datetime information found in index or columns.")
  } else if (length(time_columns) > 1) {
    stop("More than one datetime column found.")
  }

  # Select features and the date column
  selected_columns <- intersect(variables_col, names(df))
  selected_columns <- c(selected_columns, time_columns[1])

  df <- df %>%
    select(all_of(selected_columns)) %>%
    rename(date = !!sym(time_columns[1]))

  return(df)
}

#' Check Data Validity
#'
#' \code{nm_check_data} checks the validity of the input data frame and ensures the target variable exists.
#'
#' @param df Input data frame.
#' @param value The target variable name as a string.
#'
#' @return Data frame with renamed target variable.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   target_variable = rnorm(100),
#'   other_variable = rnorm(100),
#'   date = Sys.time() + seq(1, 1000, by = 10)
#' )
#' checked_df <- nm_check_data(df, value = "target_variable")
#' }
#'
#' @export
nm_check_data <- function(df, value) {
  if (!value %in% colnames(df)) {
    stop(paste("Target variable", value, "not found in the data frame"))
  }
  df <- df %>% rename(value = !!sym(value))
  if (!inherits(df$date, "POSIXct")) {
    stop("`date` variable needs to be a parsed date (Date class).")
  }
  if (any(is.na(df$date))) {
    stop("`date` must not contain missing (NA) values.")
  }
  return(df)
}

#' Impute Missing Values
#'
#' \code{nm_impute_values} imputes missing values in the data frame. Numeric columns are filled with median,
#' character columns with mode, and factor columns with the most frequent level.
#'
#' @param df Input data frame.
#' @param na_rm Logical indicating whether to remove rows with missing values.
#'
#' @return Data frame with imputed values.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(tidyr)
#' df <- data.frame(
#'   numeric_col = c(1, 2, NA, 4, 5),
#'   char_col = c("a", "b", NA, "a", "b"),
#'   factor_col = factor(c("low", "medium", "high", NA, "low"))
#' )
#' # Remove rows with missing values
#' cleaned_df <- nm_impute_values(df, na_rm = TRUE)
#' # Impute missing values
#' imputed_df <- nm_impute_values(df, na_rm = FALSE)
#' }
#' @export
nm_impute_values <- function(df, na_rm) {
  if (na_rm) {
    df <- df %>% na.omit()
  } else {
    df <- df %>% mutate(across(where(is.numeric), ~replace_na(., median(., na.rm = TRUE))))
    df <- df %>% mutate(across(where(is.character), ~replace_na(., nm_getmode(.))))
    df <- df %>% mutate(across(where(is.factor), ~replace_na(., levels(.)[which.max(table(.))])))
  }
  return(df)
}

#' Add Date Variables
#'
#' \code{nm_add_date_variables} adds date-related variables to the data frame.
#'
#' @param df Input data frame.
#' @param replace Logical indicating whether to replace existing columns.
#'
#' @return Data frame with added date variables.
#'
#' @examples
#' \donttest{
#' library(dplyr)
#' library(lubridate)
#' df <- data.frame(date = Sys.time() + seq(1, 1000, by = 100))
#' date_added_df <- nm_add_date_variables(df, replace = FALSE)
#' }
#'
#' @export
nm_add_date_variables <- function(df, replace) {
  # Load dplyr and magrittr functions

  if (replace) {
    df <- df %>%
      mutate(
        date_unix = as.numeric(as.POSIXct(date)),
        day_julian = yday(date),
        weekday = as.factor(wday(date, label = TRUE)),
        hour = hour(date)
      )
  } else {
    if (!"date_unix" %in% colnames(df)) {
      df <- df %>% mutate(date_unix = as.numeric(as.POSIXct(date)))
    }
    if (!"day_julian" %in% colnames(df)) {
      df <- df %>% mutate(day_julian = yday(date))
    }
    if (!"weekday" %in% colnames(df)) {
      df <- df %>% mutate(weekday = as.factor(wday(date, label = TRUE)))
    }
    if (!"hour" %in% colnames(df)) {
      df <- df %>% mutate(hour = lubridate::hour(date))
    }
  }
  return(df)
}

#' Convert Ordered Factors to Factors
#'
#' \code{nm_convert_ordered_to_factor} converts ordered factors in the data frame to regular factors.
#'
#' @param df Input data frame.
#'
#' @return Data frame with converted factors.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   ordered_factor = ordered(rep(c("low", "medium", "high"), length.out = 100)),
#'   other_variable = rnorm(100),
#'   date = Sys.time() + seq(1, 100, by = 1)
#' )
#' converted_df <- nm_convert_ordered_to_factor(df)
#' }
#'
#' @export
nm_convert_ordered_to_factor <- function(df) {
  df <- df %>%
    mutate(across(where(is.ordered), ~factor(as.character(.))))
  return(df)
}

#' Split Data into Training and Testing Sets
#'
#' \code{nm_split_into_sets} splits the data frame into training and testing sets based on the specified method.
#'
#' @param df Input data frame.
#' @param split_method The method for splitting data ('random', 'time_series', 'season', 'month').
#' @param fraction The proportion of data to be used for training. Default is 0.75.
#' @param seed Random seed for reproducibility. Default is 7654321.
#'
#' @return Data frame with split sets labeled.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   target = rnorm(100)
#' )
#' df_split <- nm_split_into_sets(df, split_method = 'random', fraction = 0.75, seed = 12345)
#' }
#' @export
nm_split_into_sets <- function(df, split_method, fraction = 0.75, seed = 7654321) {
  set.seed(seed)
  df <- df %>% mutate(rowid = row_number())
  if (split_method == 'random') {
    train_indices <- sample(seq_len(nrow(df)), size = floor(fraction * nrow(df)))
    df_training <- df[train_indices, ] %>% mutate(set = "training")
    df_testing <- df[-train_indices, ] %>% mutate(set = "testing")
  } else if (split_method == 'time_series') {
    split_index <- floor(fraction * nrow(df))
    df_training <- df %>% slice(1:split_index) %>% mutate(set = "training")
    df_testing <- df %>% slice((split_index + 1):n()) %>% mutate(set = "testing")
  } else if (split_method == 'season') {
    nm_get_season <- function(month) {
      if (month %in% c(12, 1, 2)) {
        return('DJF')
      } else if (month %in% c(3, 4, 5)) {
        return('MAM')
      } else if (month %in% c(6, 7, 8)) {
        return('JJA')
      } else {
        return('SON')
      }
    }
    df <- df %>% mutate(season = sapply(month(date), nm_get_season))
    df_training_list <- list()
    df_testing_list <- list()
    for (season in c('DJF', 'MAM', 'JJA', 'SON')) {
      season_df <- df %>% filter(season == !!season)
      split_index <- floor(fraction * nrow(season_df))
      season_training <- season_df %>% slice(1:split_index) %>% mutate(set = "training")
      season_testing <- season_df %>% slice((split_index + 1):n()) %>% mutate(set = "testing")
      df_training_list <- append(df_training_list, list(season_training))
      df_testing_list <- append(df_testing_list, list(season_testing))
    }
    df_training <- bind_rows(df_training_list)
    df_testing <- bind_rows(df_testing_list)
  } else if (split_method == 'month') {
    df <- df %>% mutate(month = month(date))
    df_training_list <- list()
    df_testing_list <- list()
    for (m in 1:12) {
      month_df <- df %>% filter(month == !!m)
      split_index <- floor(fraction * nrow(month_df))
      month_training <- month_df %>% slice(1:split_index) %>% mutate(set = "training")
      month_testing <- month_df %>% slice((split_index + 1):n()) %>% mutate(set = "testing")
      df_training_list <- append(df_training_list, list(month_training))
      df_testing_list <- append(df_testing_list, list(month_testing))
    }
    df_training <- bind_rows(df_training_list)
    df_testing <- bind_rows(df_testing_list)
  } else {
    stop("Unknown split method")
  }
  df_split <- bind_rows(df_training, df_testing) %>% arrange(date)
  return(df_split)
}

#' Prepare Data for Model Training
#'
#' \code{nm_prepare_data} performs a series of data preparation steps including processing,
#' checking, imputing, adding date variables, and splitting into sets.
#'
#' @param df Input data frame.
#' @param value The target variable name as a string.
#' @param feature_names The names of the features used for training.
#' @param na_rm Logical indicating whether to remove rows with missing values.
#' @param split_method The method for splitting data into training and testing sets. Default is 'random'.
#' @param replace Logical indicating whether to replace existing date variables.
#' @param fraction The proportion of the data to be used for training. Default is 0.75.
#' @param seed Random seed for reproducibility. Default is 7654321.
#'
#' @return Prepared data frame.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100),
#'   target = rnorm(100)
#' )
#' prepared_df <- nm_prepare_data(df, value = "target", feature_names = c("feature1", "feature2"))
#' }
#' @export
nm_prepare_data <- function(df, value, feature_names, na_rm = TRUE, split_method = 'random', replace = FALSE, fraction = 0.75, seed = 7654321) {

  # Perform data preparation steps
  df <- df %>%
    nm_process_df(variables_col = c(feature_names, value)) %>%
    nm_check_data(value = value) %>%
    nm_impute_values(na_rm = na_rm) %>%
    nm_add_date_variables(replace = replace) %>%
    nm_convert_ordered_to_factor() %>%
    nm_split_into_sets(split_method = split_method, fraction = fraction, seed = seed) %>%
    arrange(date)

  return(df)
}

# Helper function to get mode
nm_getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
