library(tidyverse)
library(baseballr)

##### Load Data ######

deception_matrix <- read_csv("deception_matrix_df.csv") %>%
  group_by(pitcher, p_right, cluster, pitch_type, prev_pitch, b_right) %>%
  slice_head(n = 1) %>% # Remove pitches that were predicted twice
  ungroup()

pitch_keys <- deception_matrix %>%
  group_by(pitcher, cluster, pitch_type, b_right) %>%
  summarise(
    mean_deception = mean(deception_plus, na.rm = TRUE),
    pred_key = case_when(
      any(predicted == 0) ~ 0,
      any(predicted == 2) ~ 2,
      TRUE ~ 1)
  ) %>% ungroup() 

players <- baseballr::mlb_sports_players(sport_id = 1, season = 2023) %>%
  select(player_id, full_name, primary_number, current_age, birth_city, birth_country, height, weight, current_team_name, pitch_hand_description) %>%
  filter(player_id %in% deception_matrix$pitcher) %>% 
  mutate(headshot_url = paste0("https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/", player_id, "/headshot/67/current"))

marginal_effects <- read_csv("marginal_effects.csv")
 
calculate_arsenal_efficacy <- function(pitch_keys, deception_matrix, marginal_effects, player, selected_pitches, cluster, hand) {
  fastballs = c("FF", "SI", "FC")
  breaking = c("SL", "CU", "ST")
  offspeed = c("CH", "FS")
  
  # matrix filtered down to repertoire
  standard_arsenal <- pitch_keys %>%
    filter(pitcher == player &
           b_right == 1 &
           pred_key == 0) %>%
    distinct() %>%
    pull(pitch_type)
  
  current_arsenal <- deception_matrix %>% 
    filter(
      pitcher == player & # select pitcher
        pitch_type %in% selected_pitches &
        prev_pitch %in% selected_pitches &
        cluster == cluster &
        b_right == hand # select against r/l
    )
  
  n_pitches_original = length(unique(standard_arsenal))
  n_pitches_new = length(unique(current_arsenal$pitch_type))
  
  diff <- n_pitches_new - n_pitches_original
  
  addt_effects <- if (diff < 0) {
    me <- marginal_effects %>%
      filter(Pitches %in% (n_pitches_new):(n_pitches_original - 1))
    -sum(me$Marginal_Effects)  # Double Negative - Subtract effects if the difference is negative
  } else if (diff > 0) {
    me <- marginal_effects %>%
      filter(Pitches %in% (n_pitches_original + 1):(n_pitches_new))
    sum(me$Marginal_Effects)  # Single Negative - Add effects if the difference is positive
  } else {
    0
  }
  
  arsenal_score <- current_arsenal %>%
    select(prev_pitch, deception_plus) %>%
    pivot_wider(names_from = prev_pitch,
                values_from = deception_plus) %>% 
    ungroup() %>%
    rowwise() %>% 
    mutate(overall = mean(c_across(everything()), na.rm = TRUE),
           off_fb = mean(c_across(all_of(intersect(fastballs, names(.)))), na.rm = TRUE),
           off_breaking = mean(c_across(all_of(intersect(breaking, names(.)))), na.rm = TRUE),
           off_offspeed = mean(c_across(all_of(intersect(offspeed, names(.))))), na.rm = TRUE) %>% 
    ungroup() %>% 
    summarise(grade = mean(overall, na.rm = TRUE) + addt_effects,
              off_fb = mean(off_fb, na.rm = TRUE),
              off_br = mean(off_breaking, na.rm = TRUE),
              off_os = mean(off_offspeed, na.rm = TRUE))
  return(arsenal_score)
}

##### Grab Standard Arsenal Numbers ######

all_combinations <- pitch_keys %>%
  select(pitcher, cluster, b_right) %>%
  distinct()

standard_arsenal <- data.frame(
  pitcher = numeric(),
  cluster = numeric(),
  b_right = numeric(),
  grade = numeric(),
  off_fb = numeric(),
  off_br = numeric(),
  off_os = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:nrow(all_combinations)) {
  
  player <- all_combinations$pitcher[i]
  cluster <- all_combinations$cluster[i]
  hand <- all_combinations$b_right[i]
  
  # Get the standard arsenal for this player
  arsenal <- pitch_keys %>%
    filter(pitcher == player & b_right == hand & pred_key == 0) %>%
    distinct() %>%
    pull(pitch_type)
  
  if (length(arsenal) == 0) {
    if (hand == 0) {
      arsenal <- pitch_keys %>%
        filter(pitcher == player & b_right == 1 & pred_key == 0) %>%
        distinct() %>%
        pull(pitch_type)
    } else {
      arsenal <- pitch_keys %>%
        filter(pitcher == player & b_right == 0 & pred_key == 0) %>%
        distinct() %>%
        pull(pitch_type)
    }
  }
  
  # Calculate the grade using the standard arsenal
  grade_info <- calculate_arsenal_efficacy(
    pitch_keys,
    deception_matrix,
    marginal_effects,
    player,
    arsenal,
    cluster,
    hand
  )
  
  # Store the result for this player
  standard_arsenal <- rbind(
    standard_arsenal, 
    data.frame(
      pitcher = player,
      cluster = cluster,
      b_right = hand,
      grade = grade_info$grade,
      off_fb = grade_info$off_fb,
      off_br = grade_info$off_br,
      off_os = grade_info$off_os
    )
  )
}

write_csv(standard_arsenal, "standard_arsenal.csv")

##### Grab Best Arsenal Numbers ######

find_best_arsenal <- function(pitch_keys, deception_matrix, marginal_effects, player, cluster, hand) {
  # Identify the primary fastball
  fastballs <- c("FF", "SI", "FC")
  
  primary_fastball <- deception_matrix %>%
    filter(pitcher == player & b_right == hand & cluster == cluster & is.na(pf_velo_diff)) %>%
    pull(pitch_type) %>%
    first()
  
  # Get all possible pitches excluding the primary fastball
  all_pitches <- deception_matrix %>%
    filter(pitcher == player & b_right == hand & cluster == cluster) %>%
    select(pitch_type) %>%
    distinct() %>%
    pull(pitch_type)
  
  other_pitches <- setdiff(all_pitches, primary_fastball)
  
  # Create all possible arsenals
  all_other <- lapply(1:7, function(i) combn(other_pitches, i, simplify = FALSE))
  all_other <- unlist(all_other, recursive = FALSE)
  all_arsenal <- lapply(all_other, function(comb) c(primary_fastball, comb))
  
  # Function to calculate grade for a given set of pitches
  calc_grade <- function(pitches) {
    selected_pitches <- pitches
    score <- calculate_arsenal_efficacy(pitch_keys, deception_matrix, marginal_effects, player, selected_pitches, cluster, hand)
    return(score$grade)
  }
  
  # Initialize a list to store all results
  results <- list()
  
  # Iterate over each combination in all_arsenal
  for (i in seq_along(all_arsenal)) {
    comb <- all_arsenal[[i]]
    grade <- calc_grade(comb)
    
    # Store the result in the results list
    results[[i]] <- data.frame(
      pitcher = player,
      cluster = cluster,
      hand = hand,
      combination = paste(comb, collapse = "_"),
      grade = grade,
      pitches = length(comb)
    )
  }
  
  # Combine all results into a single data frame
  results_df <- do.call(rbind, results)
  
  return(results_df)
}

best_arsenal <- data.frame(
  pitcher = numeric(),
  cluster = numeric(),
  b_right = numeric(),
  combination = numeric(),
  grade = numeric(),
  pitches = numeric(),
  stringsAsFactors = FALSE
)

# Loop through all combinations
for (i in 1:nrow(all_combinations)) {
  
  player <- all_combinations$pitcher[i]
  cluster <- all_combinations$cluster[i]
  hand <- all_combinations$b_right[i]
  
  arsenal <- find_best_arsenal(pitch_keys, deception_matrix, marginal_effects, player, cluster, hand)
  
  best_comb <- arsenal %>%
    group_by(pitches) %>%
    slice_max(grade)
  
  best_arsenal <- rbind(best_arsenal, best_comb)
  
  # Print the iteration number
  print(paste("Iteration:", i))
}

write_csv(best_arsenal, "best_arsenal.csv")

##### Grab Pitch Plot Data ######

pbp <- readRDS("updated_arm_angles (1).RDS") %>%
  filter(game_year == 2023) %>%
  mutate(
    pfx_x = pfx_x * 12,
    pfx_z = pfx_z * 12,
    p_right = ifelse(p_throws == "R", 1, 0),
    b_right = ifelse(stand == "R", 1, 0),
    # make negative x movement representative of moving away from batter
    pfx_x = ifelse(b_right == 1, -pfx_x, pfx_x),
    # make lefties with negative arm angle
    arm_angle = ifelse(p_right == 1, arm_angle, -arm_angle)
  )

fastballs = c("FF", "SI", "FC")
breaking = c("SL", "CU", "ST")
offspeed = c("CH", "FS")

# lg avg variances
pbp %>% 
  mutate(
    pitch_type == case_when(
      pitch_type == "KC" ~ "CU",
      pitch_type == "SV" ~ "SL",
      .default = pitch_type
    )
  ) %>% 
  filter(
    pitch_type %in% c(fastballs, breaking, offspeed)
    # replace this line to filter down to specific pitches
  ) %>% 
  group_by(pitcher, p_right, pitch_type) %>% 
  # mutate(pfx_x = ifelse(b_right == 0, pfx_x, -pfx_x)) %>% # reapply data transformation
  select(pfx_x, pfx_z) %>% 
  summarise(sd_x = sd(pfx_x, na.rm = T),
            sd_y = sd(pfx_z, na.rm = T),
            cor_xy = cor(pfx_x, pfx_z),
            n = n()) %>% 
  drop_na() %>% 
  mutate(
    se_x = sd_x * 1.96,
    se_y = sd_y * 1.96
  ) %>% 
  ungroup() %>%
  group_by(p_right, pitch_type) %>% 
  summarise(se_x = weighted.mean(se_x, w = n),
            se_y = weighted.mean(se_y, w = n),
            cor_xy = weighted.mean(cor_xy, w = n)) -> lg_avgs

graph_dat <- data.frame()
for (i in 1:nrow(all_combinations)) {
  
  player <- all_combinations$pitcher[i]
  cluster <- all_combinations$cluster[i]
  hand <- all_combinations$b_right[i]
  
  pbp %>% 
    mutate(
      pitch_type == case_when(
        pitch_type == "KC" ~ "CU",
        pitch_type == "SV" ~ "SL",
        .default = pitch_type
      )
    ) %>% 
    filter(
      pitch_type %in% c(fastballs, breaking, offspeed) &
        pitcher == player &
        cluster == cluster &
        b_right == hand
    ) %>% 
    group_by(pitcher, cluster, pitch_type, b_right, p_right) %>% 
    # mutate(pfx_x = ifelse(b_right == 0, pfx_x, -pfx_x)) %>% # reapply data transformation
    select(pfx_x, pfx_z, p_throws) %>% 
    summarise(mean_x = mean(pfx_x),
              mean_y = mean(pfx_z),
              sd_x = sd(pfx_x),
              sd_y = sd(pfx_z),
              cor_xy = cor(pfx_x, pfx_z),
              n = n()) %>% 
    mutate(
      se_x = sd_x * 1.96,
      se_y = sd_y * 1.96,
      cluster = as.factor(cluster)
    ) %>% 
    ungroup() %>%
    mutate(calc = "original") -> plot_data
  
  p_right <- plot_data %>%
    select(p_right) %>%
    distinct() %>%
    pull()
  
  # add on predicted/previously thrown pitches
  deception_matrix %>% 
    filter(pitcher == player &
             cluster == cluster &
             b_right == hand &
             !pitch_type %in% plot_data$pitch_type) %>% 
    group_by(pitcher, cluster, pitch_type, p_right, b_right) %>% 
    summarise(mean_x = mean(pfx_x),
              mean_y = mean(pfx_z)) %>% 
    left_join(lg_avgs %>% 
              filter(!pitch_type %in% plot_data$pitch_type &
                     p_right == p_right),
              by = c("pitch_type", "p_right")) %>%
    left_join(pitch_keys %>% select(pitcher, pitch_type, pred_key)) %>%
    distinct() %>%
    mutate(calc = case_when(
              pred_key == 0 ~ "original",
              pred_key == 1 ~ "predicted",
              pred_key == 2 ~ "previous",
              is.na(pred_key) ~ "predicted"
            ),
           cluster = as.factor(cluster)) -> pred_pit
  
  all_player_graph <- plot_data %>%
    bind_rows(pred_pit) %>%
    group_by(pitcher, pitch_type, cluster) %>%
    arrange(pitcher, pitch_type, cluster, 
            desc(is.na(pred_key)), # Prioritize NA values
            desc(pred_key == 0),    # Then prioritize pred_key == 0
            desc(pred_key == 2),    # Then prioritize pred_key == 2
            pred_key) %>%           # Finally prioritize pred_key == 1 or other values
    slice(1) %>%  # Select the first observation per group based on the order
    ungroup()
  
  graph_dat <- rbind(graph_dat, all_player_graph)
}

create_matrix <- function(current_pitcher_id, selected_pitches, current_pitcher_cluster, current_pitcher_name, hand) {
  
  handedness <- ifelse(hand == 1, "Righties", "Lefties")
  
  # Filter the data for the current pitcher
  temp <- deception_matrix %>%
    filter(pitcher == current_pitcher_id &
             cluster == current_pitcher_cluster) %>%
    mutate(deception_plus = round(deception_plus, 0),
           pitch_percentile = round(pitch_percentile, 0)) %>%
    filter(b_right == hand,
           pitch_type %in% selected_pitches,
           prev_pitch %in% selected_pitches)
  
  # Plot the data
  ggplot(temp, aes(x = pitch_type, y = prev_pitch)) +
    # Tile layers with different colors based on `predicted`
    geom_tile(data = temp %>% filter(predicted == 2), aes(fill = pitch_percentile, color = "grey"), size = 1.25) +
    geom_tile(data = temp %>% filter(predicted == 1), aes(fill = pitch_percentile, color = "darkgrey"), size = 1.25) +
    geom_tile(data = temp %>% filter(predicted == 0), aes(fill = pitch_percentile, color = "black"), size = 1.25) +
    # Add labels on top of tiles
    geom_text(aes(label = ifelse(is.na(deception_plus), "NA", deception_plus)), color = "black", size = 5) +
    # Remove fill legend
    scale_fill_gradient2(low = "royalblue", mid = "white", high = "red", midpoint = 50, na.value = "white", guide = "none") +
    # Define color and labels for the outline legend
    scale_color_manual(values = c("black" = "black", "darkgrey" = "darkgrey", "grey" = "grey"), 
                       labels = c("black" = "Currently Thrown", 
                                  "darkgrey" = "Predicted Pitch", 
                                  "grey" = "Previously Thrown"),
                       guide = guide_legend(title = "Pitch Status", override.aes = list(fill = "white"))) +
    theme_minimal() +
    labs(x = "Pitch Type", y = "Previous Pitch", title = paste0(current_pitcher_name, " Deception Matrix vs ", handedness)) +
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(hjust = 0.5, size = 18, face = "bold"),
          plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
          legend.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.text = element_text(size = 12))
}

current_pitcher_id <- 518886
selected_pitches <- c("FF", "CU")
current_pitcher_cluster <- 1
current_pitcher_name <- 'Craig Kimbrel'
hand <- 1

create_matrix(current_pitcher_id, selected_pitches, current_pitcher_cluster, current_pitcher_name, hand)

marginal_effects$Marginal_Effects <- ifelse(marginal_effects$Marginal_Effects == 0, -0.01, marginal_effects$Marginal_Effects)

ggplot(marginal_effects, aes(y = Pitches, x = Marginal_Effects)) +
  geom_bar(stat = "identity", orientation = "y", fill = "#FB4D09") +
  theme_bw() +
  xlim(0, -1.6) +
  ylim(1.5, 8.5) + 
  scale_y_continuous(breaks = 2:8, labels = 2:8) +
  labs(x = "Marginal Effects", y = "Number of Pitches", title = "Marginal Effects of Adding a Pitch") +
  geom_text(aes(label = sprintf("%.2f", Marginal_Effects)), 
            color = "white", 
            size = 4.75, 
            hjust = 1.2) +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(hjust = 0.5, size = 18, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        legend.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.text = element_text(size = 12))
  
