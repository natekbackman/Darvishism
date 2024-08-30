library(tidyverse)
library(baseballr)
library(dbscan)
library(mousetrap)

get_seed <- function() {
  str_extract_all(string = Sys.time(), 
                  pattern = "\\d+")[[1]][4:6] %>% 
    paste(collapse = "") %>% 
    as.numeric()
}

data <- read.csv("savantpbp_all.csv") %>% 
  filter(game_year %in% c(2020:2023)) %>% 
  code_barrel()
players <- read.csv("MLB_Players_Since_1876.csv")

# data %>% 
#   distinct(pitcher) %>% 
#   .[[1]] -> pitchers

players %>% 
  filter(player_id %in% pitchers &
           primary_position_code %in% c("1", "Y")) %>% 
  distinct(player_id, height, strike_zone_top) -> players

pitchers <- players$player_id

map(pitchers, function(x) {
  data.frame(
    pitcher = x,
    data %>% 
      filter(pitcher == x) %>% 
      distinct(game_year)
  )
}, .progress = T) %>% 
  bind_rows() -> all_pitchers

# pitch total distribution
data %>% 
  group_by(pitcher, game_year) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(x = n)) + 
  geom_histogram()

# determine outliers for arm angle calcs

get_outliers <- function(player, year) {
  data %>% 
    filter(pitcher == player &
             game_year == year) %>% 
    select(game_pk, at_bat_number, pitch_number,
           release_pos_x, release_pos_z) %>% 
    drop_na() -> cluster_data

  if (nrow(cluster_data) > 500) {
    minpts = 50
    eps = 0.1
  } else {
    minpts = ceiling(nrow(cluster_data) * 0.1)
    eps = 0.2
  }
  
  set.seed(get_seed())

  fpc::dbscan(cluster_data %>% 
                select(release_pos_x, release_pos_z), 
              eps = eps,
              MinPts = minpts, 
              method = "raw") -> outlier_det
  
  n_clusters = unique(outlier_det$cluster) %>% length() - 1
  
  # print(n_clusters)
  
  if (n_clusters == 0) {
    minpts = ceiling(nrow(cluster_data) * 0.05)
    eps = 0.25

    while (n_clusters == 0 & minpts > nrow(cluster_data) * 0.01) {
      minpts = minpts - 1
      
      set.seed(get_seed())
      
      fpc::dbscan(cluster_data %>% 
                    select(release_pos_x, release_pos_z), 
                  eps = eps,
                  MinPts = minpts, 
                  method = "raw") -> outlier_det
      
      n_clusters = unique(outlier_det$cluster) %>% length() - 1
    }
  } else if (n_clusters > 2) {
    while (n_clusters > 2 & n_clusters != 0 & eps < 0.5) {
      eps = eps + 0.01

      set.seed(get_seed())
      
      fpc::dbscan(cluster_data %>% 
                    select(release_pos_x, release_pos_z), 
                  eps = eps,
                  MinPts = minpts, 
                  method = "raw") -> outlier_det
      
      n_clusters = unique(outlier_det$cluster) %>% length() - 1
    }
  } else {
    invisible()
  }
  
  cluster_data %>% 
    mutate(outlier = outlier_det$cluster %>% 
             as.factor()) -> cluster_data
  
  bix = bimodality_coefficient(
    cluster_data %>% 
      pull(release_pos_x)
  )
  
  biz = bimodality_coefficient(
    cluster_data %>% 
      pull(release_pos_z)
  )
  
  # print(glue::glue("bix: {bix}; biz: {biz}"))
  
  if (bix > 5/9 & biz < 5/9 & n_clusters == 1) {
    while (n_clusters != 2 & eps >= 0.051) {
      eps = eps - 0.01

      set.seed(get_seed())
      
      fpc::dbscan(cluster_data %>% 
                    select(release_pos_x, release_pos_z), 
                  eps = eps,
                  MinPts = minpts, 
                  method = "raw") -> outlier_det
      
      n_clusters = unique(outlier_det$cluster) %>% length() - 1
    }
  } else {
    invisible()
  }
  
  if (bix > 5/9 & biz < 5/9 & n_clusters == 0) {
    if (nrow(cluster_data) > 500) {
      minpts = 50
      eps = 0.1
    } else {
      minpts = ceiling(nrow(cluster_data) * 0.1)
      eps = 0.2
    }
    
    set.seed(get_seed())
    
    fpc::dbscan(cluster_data %>% 
                  select(release_pos_x, release_pos_z), 
                eps = eps,
                MinPts = minpts, 
                method = "raw") -> outlier_det
    
    n_clusters = unique(outlier_det$cluster) %>% length() - 1
  } else {
    invisible()
  }
  
  # add cluster tags to the data
  cluster_data %>% 
    mutate(outlier = outlier_det$cluster %>% 
             as.factor()) -> cluster_data
  
  bimodal_tag = ifelse(bix > 5/9 & 
                         biz < 5/9 & 
                         n_clusters == 2,
                       1, 0)

  # print(glue::glue("Minpts: {minpts}; Eps: {eps}"))
  # print(glue::glue("Pitcher: {player}; Year: {year}"))
  
  cluster_data %>% 
    mutate(bimodal = bimodal_tag)
}

pmap(list(player = all_pitchers$pitcher,
          year = all_pitchers$game_year), 
     get_outliers,
     .progress = T) %>% 
  bind_rows() -> outlier_data

data %>% 
  left_join(outlier_data %>% 
              select(game_pk, at_bat_number, pitch_number, outlier, bimodal),
            by = c("game_pk" = "game_pk",
                   "at_bat_number" = "at_bat_number",
                   "pitch_number" = "pitch_number")) -> data

players %>% 
  mutate(ft = parse_number(height),
         inches = gsub("\\D", "", height) %>% substring(2) %>% as.numeric(),
         height_ft = (ft * 12 + inches) / 12,
         arm_length = height_ft / 2) -> players

data %>% 
  drop_na(release_pos_x, release_pos_z) %>% 
  filter(outlier != "0") %>% 
  select(pitcher, release_pos_x, release_pos_z, game_pk, p_throws, game_year,
         release_extension, at_bat_number, pitch_number, outlier, bimodal) %>% 
  left_join(players, by = c("pitcher" = "player_id")) %>% 
  group_by(pitcher) %>% 
  mutate(adj = release_pos_z - strike_zone_top,
         opp = abs(release_pos_x),
         hyp = sqrt(adj^2 + opp^2),
         num = adj^2 + hyp^2 - opp^2,
         den = 2 * (adj * hyp),
         arm_angle = acos(num / den) * (180 / pi),
         arm_angle = ifelse(adj == 0, 90, arm_angle)) %>% 
  rename("cluster" = "outlier") %>% 
  ungroup() -> arm_angle_data

unique(arm_angle_data$cluster)

arm_angle_data %>% 
  group_by(pitcher, game_year, cluster, bimodal) %>% 
  summarise(n = n(),
            avg_arm_length = mean(hyp)) %>% 
  left_join(players %>% 
              select(player_id, arm_length), 
            by = c("pitcher" = "player_id")) %>% 
  ungroup() -> arm_angle_clusters

arm_angle_clusters %>% 
  filter(bimodal == 1) %>% 
  group_by(pitcher, game_year) %>% 
  mutate(length_diff = abs(avg_arm_length - arm_length)) %>% 
  arrange(length_diff) %>% 
  slice_min(length_diff) -> filtered_clusters

clean_data <- function(player, year, tag) {
  if (tag == 1) {
    clus = filtered_clusters %>% 
      filter(pitcher == player &
               game_year == year) %>% 
      pull(cluster)
    
    arm_angle_data %>% 
      filter(pitcher == player &
               cluster == clus &
               game_year == year)
  } else {
    arm_angle_data %>% 
      filter(pitcher == player &
               game_year == year)
  }
}

pmap(list(player = arm_angle_clusters$pitcher,
          year = arm_angle_clusters$game_year,
          tag = arm_angle_clusters$bimodal),
     clean_data,
     .progress = T) %>% 
  bind_rows() -> arm_angle_data

data <- read.csv("savantpbp_all.csv") %>% 
  filter(game_year %in% c(2020:2023)) %>% 
  code_barrel() %>% 
  left_join(arm_angle_data %>% 
              select(pitcher, game_pk, at_bat_number, pitch_number,
                     cluster, arm_angle),
            by = c("pitcher" = "pitcher",
                   "game_pk" = "game_pk",
                   "at_bat_number" = "at_bat_number",
                   "pitch_number" = "pitch_number")) -> data

saveRDS(data, "updated_arm_angles.RDS")

### SCRATCH BELOW ###

# get bimodal data

get_bimodal_pitchers <- function(x) {
  bix <- bimodality_coefficient(
    arm_angle_data %>% 
      filter(pitcher == x) %>% 
      pull(release_pos_x)
  )
  
  biz <- bimodality_coefficient(
    arm_angle_data %>% 
      filter(pitcher == x) %>% 
      pull(release_pos_z)
  )
  
  data %>% 
    mutate(outlier = outlier %>% as.character()) %>% 
    filter(pitcher == x,
           outlier != "0") %>% 
    pull(outlier) %>% 
    unique() %>% 
    length() -> n_clusters
  
  data.frame(pitcher = x,
             bix = bix,
             biz = biz,
             n_clusters = n_clusters)
}

map(pitchers, get_bimodal_pitchers, .progress = T) %>% 
  bind_rows() %>% 
  filter(bix > 5/9 & biz < 5/9) -> bimodal_pitchers

# all single cluster bimodal observations will be rerun thru dbscan with smaller epsilon value
eps = 0.05

map(bimodals, get_outliers, .progress = T) %>% 
  bind_rows() %>% 
  rename("updated_outlier" = "outlier") -> bimodal_data

outlier_data %>% 
  left_join(bimodal_data,
            by = c("game_pk" = "game_pk",
                   "at_bat_number" = "at_bat_number",
                   "pitch_number" = "pitch_number",
                   "release_pos_x" = "release_pos_x",
                   "release_pos_z" = "release_pos_z")) %>% 
  mutate(outlier = coalesce(updated_outlier, outlier), 
         .keep = "unused") -> outlier_data

### testing

arm_angle_clusters %>% 
  filter(cluster %in% c("3", "4", "5")) -> anomalies

data %>% 
  filter(pitcher == 676083 &
           game_year == 2023) %>% 
  ggplot(aes(release_pos_x, release_pos_z, color = cluster)) +
  geom_point()


write.csv(arm_angle_data, "arm_angles.csv")

data %>% 
  filter(!outlier %in% c("0", "1", "2")) %>% 
  group_by(pitcher) %>% 
  summarise(n = n()) %>% 
  ggplot(aes(x = n)) +
  geom_histogram()

data %>% 
  filter(!outlier %in% c("0", "1", "2")) %>% 
  group_by(pitcher) %>% 
  summarise(n = n()) %>% 
  slice_max(n) %>% 
  select(pitcher) %>% .[[1]] -> test_pitcher

data %>% 
  filter(pitcher == 518886 &
           game_year == 2023) %>% 
  ggplot(aes(release_pos_x, release_pos_z, color = cluster)) +
  geom_point() +
  theme_bw(base_family = "serif") +
  labs(title = "Craig Kimbrel Release Points (2023)") +
  xlab("Release Position X") +
  ylab("Release Position Z")

data %>% 
  filter(pitcher == 448179 &
           game_year == 2022) %>% 
  ggplot(aes(release_pos_x, release_pos_z, color = cluster)) +
  geom_point() +
  theme_bw(base_family = "serif") +
  labs(title = "Rich Hill Release Points (2022)") +
  xlab("Release Position X") +
  ylab("Release Position Z")

# fine tune arm angle calculations with different x spots

# if release_point_x is bimodal but release_point_z is not, the pitcher uses multiple sides of the rubber

# best params (50 minpts and 0.1 eps) have 25 pitchers with 1 bimodal cluster

# check 1 cluster bimodal obs
bimodal_pitchers %>% 
  filter(n_clusters == 1) %>% 
  arrange(desc(bix)) %>% 
  pull(pitcher) -> bimodals

data %>% 
  filter(pitcher == 518516) %>% 
  ggplot(aes(release_pos_x, release_pos_z, color = outlier)) +
  geom_point()

# re-read in data
data <- read.csv("savantpbp_all.csv") %>% 
  filter(game_year %in% c(2020:2023)) %>% 
  code_barrel()

data %>% 
  filter(!pitcher %in% bimodals) %>% 
  left_join(outlier_data %>% 
              select(game_pk, at_bat_number, pitch_number, outlier),
            by = c("game_pk" = "game_pk",
                   "at_bat_number" = "at_bat_number",
                   "pitch_number" = "pitch_number")) -> base

data %>% 
  filter(pitcher %in% bimodals) %>% 
  left_join(bimodal_data %>% 
              select(game_pk, at_bat_number, pitch_number, outlier),
            by = c("game_pk" = "game_pk",
                   "at_bat_number" = "at_bat_number",
                   "pitch_number" = "pitch_number")) -> edited

base %>% 
  filter(pitcher == 623437) %>% 
  ggplot(aes(release_pos_x, release_pos_z, color = outlier)) +
  geom_point()

base %>% 
  bind_rows(edited) %>% 
  drop_na(release_pos_x, release_pos_z) %>% 
  filter(outlier != "0") %>% 
  select(pitcher, release_pos_x, release_pos_z, game_pk, p_throws, 
         release_extension, at_bat_number, pitch_number, outlier) %>% 
  left_join(players, by = c("pitcher" = "player_id")) %>% 
  group_by(pitcher) %>% 
  mutate(adj = release_pos_z - strike_zone_top,
         opp = abs(release_pos_x),
         hyp = sqrt(adj^2 + opp^2),
         num = adj^2 + hyp^2 - opp^2,
         den = 2 * (adj * hyp),
         arm_angle = acos(num / den) * (180 / pi),
         arm_angle = ifelse(adj == 0, 90, arm_angle)) %>% 
  ungroup() -> arm_angle_data

rm(base, edited)

# build some logic that determines "true" cluster based on hyp length relative to arm length (derived from height)

#####

arm_angle_data %>% 
  mutate(ft = parse_number(height),
         inches = gsub("\\D", "", height) %>% substring(2) %>% as.numeric(),
         height_ft = (ft * 12 + inches) / 12) -> arm_angle_data
