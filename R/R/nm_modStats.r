#' Calculate Model Statistics
#'
#' \code{nm_modStats} calculates various statistical measures for a given model and dataset.
#'
#' @param df Data frame containing the input data.
#' @param model The trained model object.
#' @param obs The name of the observed values column. Default is "value".
#' @param set The name of the set for which to calculate statistics. If NULL, statistics for all sets will be calculated. Default is NULL.
#' @param statistic A vector of statistics to calculate. Default is c("n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2").
#'
#' @return A data frame containing the calculated statistics.
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' library(magrittr)
#' df <- data.frame(
#'   set = rep(c("train", "test"), each = 50),
#'   value = rnorm(100)
#' )
#' model <- lm(value ~ set, data = df)
#' stats <- nm_modStats(df, model, obs = "value")
#' }
#' @export
nm_modStats <- function(df, model, obs = "value", set = NULL, statistic = NULL) {
  # Set default statistics if not provided
  if (is.null(statistic)) {
    statistic <- c("n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2")
  }

  # Function to calculate statistics for a given set
  calculate_stats <- function(df, set_name) {
    if (!is.null(set_name)) {
      df <- df %>% filter(set == !!set_name)
    }

    # Predict values using the model
    value_predict <- nm_predict(model, df)

    # Add predicted values to the dataframe
    df <- df %>% mutate(value_predict = value_predict)

    # Calculate statistics
    df_stats <- nm_Stats(df, mod = "value_predict", obs = obs, statistic = statistic)
    df_stats$set <- set_name

    return(df_stats)
  }

  # If 'set' is NULL, calculate statistics for all possible sets and combine
  if (is.null(set)) {
    if ("set" %in% names(df)) {
      sets <- unique(df$set)
      stats_list <- lapply(sets, function(s) calculate_stats(df, s))

      # Add statistics for the whole dataset with 'set' as "all"
      df_all <- df
      df_all$set <- "all"
      stats_list <- c(stats_list, list(calculate_stats(df_all, "all")))

      df_stats <- do.call(rbind, stats_list)
    } else {
      stop("The DataFrame does not contain the 'set' column and 'set' parameter was not provided.")
    }
  } else {
    df_stats <- calculate_stats(df, set)
  }

  return(df_stats)
}

#' Calculate Various Statistics
#'
#' \code{nm_Stats} calculates various statistical measures for model evaluation.
#'
#' @param df Data frame containing the input data.
#' @param mod The name of the model predictions column.
#' @param obs The name of the observed values column.
#' @param statistic A vector of statistics to calculate. Default is c("n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2").
#'
#' @return A data frame containing the calculated statistics.
#'
#' @examples
#' \donttest{
#' library(dplyr)
#' # Example usage
#' df <- data.frame(value_predict = rnorm(100), value = rnorm(100))
#' stats <- nm_Stats(df, mod = "value_predict", obs = "value")
#' }
#'
#' @export
nm_Stats <- function(df, mod, obs, statistic = NULL) {
  library(dplyr)
  if (is.null(statistic)) {
    statistic <- c("n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA", "R2")
  }

  res <- list()

  if ("n" %in% statistic) {
    res$n <- nm_nstat(df, mod, obs)
  }
  if ("FAC2" %in% statistic) {
    res$FAC2 <- nm_FAC2(df, mod, obs)
  }
  if ("MB" %in% statistic) {
    res$MB <- nm_MB(df, mod, obs)
  }
  if ("MGE" %in% statistic) {
    res$MGE <- nm_MGE(df, mod, obs)
  }
  if ("NMB" %in% statistic) {
    res$NMB <- nm_NMB(df, mod, obs)
  }
  if ("NMGE" %in% statistic) {
    res$NMGE <- nm_NMGE(df, mod, obs)
  }
  if ("RMSE" %in% statistic) {
    res$RMSE <- nm_RMSE(df, mod, obs)
  }
  if ("r" %in% statistic) {
    r_result <- nm_r_stat(df, mod, obs)
    res$r <- r_result[1]
    p_value <- r_result[2]
    res$p_level <- if (p_value >= 0.1) {
      ""
    } else if (p_value < 0.1 & p_value >= 0.05) {
      "+"
    } else if (p_value < 0.05 & p_value >= 0.01) {
      "*"
    } else if (p_value < 0.01 & p_value >= 0.001) {
      "**"
    } else {
      "***"
    }
  }
  if ("COE" %in% statistic) {
    res$COE <- nm_COE(df, mod, obs)
  }
  if ("IOA" %in% statistic) {
    res$IOA <- nm_IOA(df, mod, obs)
  }
  if ("R2" %in% statistic) {
    res$R2 <- nm_R2(df, mod, obs)
  }

  results <- data.frame(t(unlist(res)))

  return(results)
}

## Number of valid readings
nm_nstat <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(nrow(df))
}

## Fraction within a factor of two
nm_FAC2 <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  ratio <- df[[mod]] / df[[obs]]
  ratio <- na.omit(ratio)
  len <- length(ratio)
  if (len > 0) {
    return(sum(ratio >= 0.5 & ratio <= 2) / len)
  } else {
    return(NA)
  }
}

## Mean bias
nm_MB <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(mean(df[[mod]] - df[[obs]]))
}

## Mean gross error
nm_MGE <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(mean(abs(df[[mod]] - df[[obs]])))
}

## Normalised mean bias
nm_NMB <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(sum(df[[mod]] - df[[obs]]) / sum(df[[obs]]))
}

## Normalised mean gross error
nm_NMGE <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(sum(abs(df[[mod]] - df[[obs]])) / sum(df[[obs]]))
}

## Root mean square error
nm_RMSE <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(sqrt(mean((df[[mod]] - df[[obs]])^2)))
}

## Correlation coefficient
nm_r_stat <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  cor_test <- cor.test(df[[mod]], df[[obs]])
  return(c(cor_test$estimate, cor_test$p.value))
}

## Coefficient of Efficiency
nm_COE <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  return(1 - sum(abs(df[[mod]] - df[[obs]])) / sum(abs(df[[obs]] - mean(df[[obs]]))))
}

## Index of Agreement
nm_IOA <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  LHS <- sum(abs(df[[mod]] - df[[obs]]))
  RHS <- 2 * sum(abs(df[[obs]] - mean(df[[obs]])))
  if (LHS <= RHS) {
    return(1 - LHS / RHS)
  } else {
    return(RHS / LHS - 1)
  }
}

## Determination of coefficient
nm_R2 <- function(df, mod, obs) {
  df <- df %>% dplyr::select(all_of(c(mod, obs))) %>% na.omit()
  fit <- lm(df[[mod]] ~ df[[obs]])
  return(summary(fit)$r.squared)
}
