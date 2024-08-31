library(tidyverse)

# data transformation ideas
# make lefty arm angles negative (remove p_throws)
# make horizontal movement negative for movement away from batter (how does it affect x_diffs?)
data <- readRDS("updated_arm_angles (1).RDS")

pitches = c("4-Seam Fastball", "Changeup", "Curveball", "Cutter",
            "Knuckle Curve", "Sinker", "Slider", "Split-Finger",
            "Sweeper")

fastballs = c("4-Seam Fastball", "Sinker", "Cutter")

data %>% 
  mutate(
    pfx_x = pfx_x * 12,
    pfx_z = pfx_z * 12,
    p_right = ifelse(p_throws == "R", 1, 0),
    b_right = ifelse(stand == "R", 1, 0),
    # make negative x movement representative of moving away from batter
    pfx_x = ifelse(b_right == 1, -pfx_x, pfx_x),
    # make lefties with negative arm angle
    arm_angle = ifelse(p_right == 1, arm_angle, -arm_angle),
  ) %>%
  filter(pitch_name %in% fastballs) %>% 
  group_by(game_year, pitcher, pitch_name, p_right, b_right) %>%
  summarise(
    velo = mean(release_speed, na.rm = TRUE),
    h_break = mean(pfx_x, na.rm = TRUE),
    v_break = mean(pfx_z, na.rm = TRUE),
    aa = median(arm_angle, na.rm = TRUE),
    n_obs = n()
  ) %>%
  ungroup() %>%
  group_by(game_year, pitcher, b_right) %>%
  arrange(desc(n_obs)) %>% 
  slice_head() %>% 
  drop_na(aa) %>% 
  rename("fb" = "pitch_name") %>% 
  ungroup() -> pf_data

data %>% 
  filter(game_type %in% "R") %>% 
  group_by(game_pk, pitcher, at_bat_number) %>% 
  arrange(at_bat_number, pitch_number) %>% 
  mutate(prev_pitch = lag(pitch_type, 1),
         pfx_x = pfx_x * 12,
         pfx_z = pfx_z * 12,
         p_right = ifelse(p_throws == "R", 1, 0),
         b_right = ifelse(stand == "R", 1, 0),
         # make negative x movement representative of moving away from batter
         pfx_x = ifelse(b_right == 1, -pfx_x, pfx_x),
         # make lefties with negative arm angle
         arm_angle = ifelse(p_right == 1, arm_angle, -arm_angle),
         prev_pitch = lag(pitch_name),
         diff_pitch = ifelse(pitch_name != prev_pitch, 1, 0),
         velo_diff = release_speed - lag(release_speed),
         x_diff = pfx_x - lag(pfx_x),
         y_diff = pfx_z - lag(pfx_z),
         # dependent variable
         b_ss = case_when(
           launch_speed_angle == 6 ~ 2,
           launch_angle <= 32 & launch_angle >= 8 ~ 2,
           !is.na(launch_speed_angle) & !is.na(launch_angle) ~ 1,
           description == "foul" ~ 1, # place foul balls in same class as weak contact
           .default = 0
         ),
         swing = case_when(
           type == "X" ~ 1,
           type == "S" & 
             description %in% c("foul",
                                "foul_tip",
                                "swinging_strike",
                                "swinging_strike_blocked") ~ 1,
           .default = 0
         )) %>% 
  mutate(pitch_name = case_when(
    pitch_name %in% c("Slurve") ~ "Slider",
    pitch_name %in% c("Knuckle Curve") ~ "Curveball",
    TRUE ~ pitch_name
  )) %>%
  filter(!grepl("bunt", des, ignore.case = T) &
           !description %in% c("pitchout", "hit_by_pitch") &
           swing == 1) -> stuff

rm(data)

# add whatever variables you need to this vector:
predictors = c("release_speed", "pfx_x", "pfx_z", "arm_angle",
               # "b_right", "p_right",
               "primary_fb", "release_pos_z")

stuff %>% 
  left_join(pf_data %>% 
              select(game_year, pitcher, p_right, b_right, velo, h_break, v_break, aa, fb), 
            by = c("game_year" = "game_year",
                   "pitcher" = "pitcher",
                   "p_right" = "p_right",
                   "b_right" = "b_right"), 
            relationship = "many-to-one") %>% 
  mutate(pf_velo_diff = release_speed - velo,
         pf_hbreak_diff = pfx_x - h_break,
         pf_vbreak_diff = pfx_z - v_break,
         pf_aa_diff = ifelse(is.na(arm_angle) == F,
                             (arm_angle - aa) %>% as.character(),
                             NA_character_),
         primary_fb = ifelse(pitch_name == fb, 1, 0)) %>% 
  ungroup() -> stuff

# model data
stuff %>% 
  filter(pitch_name %in% pitches) %>% 
  filter(game_year %in% 2020:2022) %>% 
  select(b_ss, all_of(predictors), ends_with("diff"), primary_fb,
         # keep merge keys to evaluate stuff scores in test set
         game_pk, game_year, pitcher, player_name, at_bat_number, pitch_number, 
         pitch_type, prev_pitch, cluster, b_right, p_right) %>% 
  drop_na(release_speed, pfx_x, pfx_z, 
          pf_velo_diff, pf_hbreak_diff, pf_vbreak_diff) %>% 
  mutate(pf_velo_diff = ifelse(primary_fb == 1, NA_character_, pf_velo_diff),
         pf_hbreak_diff = ifelse(primary_fb == 1, NA_character_, pf_hbreak_diff),
         pf_vbreak_diff = ifelse(primary_fb == 1, NA_character_, pf_vbreak_diff),
         pf_aa_diff = ifelse(primary_fb == 1, NA_character_, pf_aa_diff)) %>% 
  select(-primary_fb) -> stuff_model

write.csv(stuff_model, "stuff_model.csv")

stuff %>% 
  filter(pitch_name %in% pitches) %>% 
  filter(game_year %in% 2023) %>% 
  select(b_ss, all_of(predictors), ends_with("diff"), primary_fb,
         # keep merge keys to evaluate stuff scores in test set
         game_pk, game_year, pitcher, player_name, at_bat_number, pitch_number, 
         pitch_type, prev_pitch, cluster, b_right, p_right) %>% 
  drop_na(release_speed, pfx_x, pfx_z, 
          pf_velo_diff, pf_hbreak_diff, pf_vbreak_diff) %>% 
  mutate(pf_velo_diff = ifelse(primary_fb == 1, NA_character_, pf_velo_diff),
         pf_hbreak_diff = ifelse(primary_fb == 1, NA_character_, pf_hbreak_diff),
         pf_vbreak_diff = ifelse(primary_fb == 1, NA_character_, pf_vbreak_diff),
         pf_aa_diff = ifelse(primary_fb == 1, NA_character_, pf_aa_diff)) %>% 
  select(-primary_fb) -> stuff_model_23

write.csv(stuff_model_23, "stuff_model_test.csv")

#### Predicted Pitches ####
ch_proj = read_csv("ch_proj.csv") %>% mutate(pitch_type = "CH")
ct_proj = read_csv("ct_proj.csv") %>% mutate(pitch_type = "FC")
cu_proj = read_csv("cu_proj.csv") %>% mutate(pitch_type = "CU")
ff_proj = read_csv("ff_proj.csv") %>% mutate(pitch_type = "FF")
si_proj = read_csv("si_proj.csv") %>% mutate(pitch_type = "SI")
sl_proj = read_csv("sl_proj.csv") %>% mutate(pitch_type = "SL")
sp_proj = read_csv("sp_proj.csv") %>% mutate(pitch_type = "FS")
st_proj = read_csv("st_proj.csv") %>% mutate(pitch_type = "ST")
proj_pitches <- rbind(ch_proj, ct_proj, cu_proj, ff_proj, si_proj, sl_proj, sp_proj, st_proj)

pitches_to_fix <- proj_pitches %>% filter(is.na(proj_velo | proj_x | proj_y))
write_csv(pitches_to_fix, "pitches_to_fix.csv")
fixed_pitch_projections <- read_csv("fixed_pitch_projections.csv") %>% select(-1) %>% mutate(cluster = as.factor(cluster))
fixed_pitch_projections_2 <- read_csv("fixed_pitch_projections_2.csv") %>% filter(game_year == 2023 | is.na(game_year)) %>% select(-1, -game_year) %>% mutate(cluster = as.factor(cluster))
fixed_pitch_projections_3 <- read_csv("fixed_pitch_projections_3.csv") %>% mutate(cluster = as.factor(cluster))

proj_pitches <- rbind(proj_pitches, 
                      fixed_pitch_projections, 
                      fixed_pitch_projections_2,
                      fixed_pitch_projections_3
                      ) %>%
  filter(!is.na(proj_velo)) %>%
  group_by(pitcher, cluster, pitch_type) %>%
  slice_head(n = 1) # Remove pitches that were predicted twice

stuff %>%
  filter(game_year == 2023) %>%
  group_by(pitcher, cluster, b_right) %>%
  summarise(
    arm_angle = mean(arm_angle, na.rm = TRUE),
    release_pos_z = mean(release_pos_z, na.rm = TRUE),
  ) %>% filter(!is.na(cluster)) -> aa_rel

proj_pitches %>%
  crossing(b_right = c(0, 1)) %>%
  left_join(pf_data %>% filter(game_year == 2023) %>%
            select(game_year, pitcher, p_right, b_right, velo, h_break, v_break, aa, fb), 
            by = c("pitcher" = "pitcher",
                   "b_right" = "b_right")) %>%
  left_join(aa_rel, by = c("pitcher" = "pitcher",
                           "cluster" = "cluster",
                           "b_right" = "b_right")) %>%
  mutate(
    # make negative x movement representative of moving away from batter
    proj_x = ifelse(b_right == 1, -proj_x, proj_x),
    # # make lefties with negative arm angle
    # arm_angle = ifelse(p_right == 1, arm_angle, -arm_angle)
    pf_velo_diff = velo - proj_velo,
    pf_hbreak_diff = h_break - proj_x,
    pf_vbreak_diff = v_break - proj_y,
    pf_aa_diff = aa - arm_angle,
  ) %>% select(
    pitcher, pitch_type, p_right, b_right, cluster,
    release_speed = proj_velo,
    pfx_x = proj_x,
    pfx_z = proj_y,
    arm_angle, release_pos_z, pf_velo_diff, pf_hbreak_diff,
    pf_vbreak_diff, pf_aa_diff
  ) %>% distinct() -> pitch_projections
write_csv(pitch_projections, "pitch_projections.csv")

#### Validation Pitches ####
velo_val <- read_csv("velo_val.csv") %>% select(pitcher, cluster, pitch_type, proj_velo, accurate_velo = accurate)
x_val <- read_csv("x_val.csv") %>% select(pitcher, cluster, pitch_type, proj_x, accurate_x = accurate)
y_val <- read_csv("y_val.csv") %>% select(pitcher, cluster, pitch_type, proj_y, accurate_y = accurate)
val_set <- velo_val %>% full_join(x_val) %>% full_join(y_val) %>%
  mutate(cluster = as.factor(cluster)) %>%
  crossing(b_right = c(0, 1)) %>%
  left_join(pf_data %>% filter(game_year == 2023) %>%
              select(game_year, pitcher, p_right, b_right, velo, h_break, v_break, aa, fb), 
            by = c("pitcher" = "pitcher",
                   "b_right" = "b_right")) %>%
  left_join(aa_rel, by = c("pitcher" = "pitcher",
                           "cluster" = "cluster",
                           "b_right" = "b_right")) %>%
  mutate(
    # make negative x movement representative of moving away from batter
    proj_x = ifelse(b_right == 1, -proj_x, proj_x),
    # # make lefties with negative arm angle
    #arm_angle = ifelse(p_right == 1, arm_angle, -arm_angle),
    pf_velo_diff = velo - proj_velo,
    pf_hbreak_diff = h_break - proj_x,
    pf_vbreak_diff = v_break - proj_y,
    pf_aa_diff = aa - arm_angle,
  ) %>% select(
    pitcher, pitch_type, p_right, b_right, cluster,
    release_speed = proj_velo,
    pfx_x = proj_x,
    pfx_z = proj_y,
    arm_angle, release_pos_z, pf_velo_diff, pf_hbreak_diff,
    pf_vbreak_diff, pf_aa_diff
  ) %>% distinct()
write_csv(val_set, "val_set.csv")
