using System;
using System.IO;
using System.Reflection;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Thresholds are serialized as flat fields. Each JSON value has a reason beside it.</summary>
    [Serializable]
    public sealed class BrainConfig
    {
        public float full_snapshot_s = 10;
        public float peer_claim_s = 15;
        public float peer_claim_radius_m = 1;
        public float grid_size_m = 44.0f;
        public bool fixed_grid_origin = false;
        public float grid_origin_x_m = -2.0f;
        public float grid_origin_z_m = -2.0f;
        public float cell_m = 0.1f;
        public int depth_stride = 4;
        public float obstacle_min_height_m = 0.15f;
        public float obstacle_max_height_m = 1.9f;
        public float free_log_odds = -0.45f;
        public float occupied_log_odds = 0.9f;
        public float log_odds_cap = 5.0f;
        public float free_threshold = -0.4f;
        public float occupied_threshold = 0.7f;
        public float robot_radius_m = 0.4f;
        public float body_half_length_m = 0.375f, body_radius_m = 0.18f;
        public float obstacle_cost_margin_m = 0.15f;
        public float obstacle_cost_weight = 2.0f;
        public float stuck_command_mps = 0.15f;
        public float stuck_measured_mps = 0.03f;
        public float stuck_window_s = 1.2f;
        public float stuck_strip_depth_m = 0.2f;
        public float stuck_log_odds = 0.9f;
        public float stuck_reverse_m = 0.2f;
        public float stuck_reverse_mps = 0.15f;
        public float stuck_reverse_timeout_s = 2.5f;
        public float stuck_turn_deg = 45.0f;
        public float stuck_turn_timeout_s = 2.5f;
        public float stuck_repeat_radius_m = 0.75f;
        public int stuck_repeat_count = 3;
        public float stuck_blacklist_s = 30.0f;
        public float ambient_c = 20.0f;
        public float heat_project_min_c = 60.0f;
        public float fire_c = 200.0f;
        public float fire_clearance_m = 1.2f;
        public float heat_cost_radius_m = 2.5f;
        public float heat_cost_weight = 8.0f;
        public float heat_decay_c_per_s = 0.2f;
        public float heat_prediction_s = 1.0f;
        public float heat_rise_memory_s = 2.0f;
        public float frame_pair_tolerance_s = 0.08f;
        public float frame_max_age_s = 0.5f;
        public float motion_stale_s = 0.25f;
        public int health_stride = 4;
        public float corner_gradient = 0.08f;
        public int corner_target = 40;
        public float contrast_target = 0.12f;
        public float health_threshold = 0.45f;
        public float health_corner_weight = 0.4f;
        public float health_contrast_weight = 0.35f;
        public float health_depth_weight = 0.25f;
        public float switch_to_thermal_s = 0.5f;
        public float switch_to_rgb_s = 2.0f;
        public float health_max_gap_s = 0.2f;
        public float person_min_c = 28.0f;
        public float person_max_c = 40.0f;
        public float person_min_area_m2 = 0.25f;
        public float person_max_area_m2 = 2.5f;
        public int person_min_pixels = 6;
        public float blob_depth_jump_m = 0.5f;
        public int candidate_frames = 3;
        public float candidate_window_s = 1.0f;
        public float survivor_merge_m = 1.0f;
        public float survivor_standoff_m = 1.2f;
        public float survivor_stop_min_m = 1.0f;
        public float survivor_stop_max_m = 1.5f;
        public float survivor_face_tolerance_deg = 10.0f;
        public float agent_arrival_hold_s = 5.0f;
        public float escape_time_factor = 1.5f, escape_rate_window_s = 4.0f;
        public float survivor_dwell_s = 3.0f;
        public float survivor_fresh_s = 0.5f;
        public float survivor_lost_s = 3.0f;
        public float survivor_stop_speed_mps = 0.03f;
        public int standoff_samples = 32;
        public float replan_s = 1.0f;
        public float frontier_min_distance_m = 0.5f;
        public float frontier_gain_weight = 0.04f;
        public int frontier_gain_cap_cells = 150, frontier_min_information_cells = 10;
        public float visited_radius_m = 0.8f, visited_hold_s = 90f;
        public float sweep_max_s = 12f;
        public float collision_escape_s = 3f;
        public float collision_margin_m = 0.05f;
        public float lead_max_m = 1.0f;
        public float scan_yaw_gain = 0.05f;
        public float agent_avoid_min_radius_m = 0.3f, agent_avoid_max_radius_m = 4f;
        public float frontier_cluster_max_m = 2.5f;
        public bool search_requires_ceiling = false, ceiling_sight_lines = true;
        public bool frontier_local_evidence = true;
        public bool reentry_requires_interior_link = true;
        public bool frontier_doorway_looks = true;
        public float ceiling_min_height_m = 2.5f, ceiling_radius_m = 0.6f, ceiling_wall_margin_m = 0.4f;
        public int ceiling_min_hits = 2;
        public int enclosure_rays = 36;
        public float enclosure_range_m = 10f, enclosure_min_closed_share = 0.6f, enclosure_max_open_share = 0.1f;
        public float lidar_open_span_deg = 30f;
        public float fire_cut_blacklist_s = 20f;
        public float reenter_after_heat_s = 30f;
        public float candidate_wall_extent_m = 2f, candidate_wall_disc_m = .3f, candidate_wall_window_m = 3f;
        public float last_resort_fire_clearance_m = .25f;
        public float air_warm_c = 30f, air_rise_c_per_min = 8.3f, air_rise_window_s = 10f, air_rise_min_delta_c = 5f, air_rise_baseline_s = 60f;
        public float frontier_peek_m = 1.2f, see_over_clearance_m = 0.5f;
        public float search_low_m = .3f, frontier_beyond_m = 12f, reject_new_view_deg = 45f, frontier_wide_min_m2 = 1f, reject_share_s = 60f;
        public int frontier_wide_options = 3;
        public bool thermal_lidar_range_fallback = true;
        public float carried_person_radius_m = 0.85f, search_max_range_m = 4.0f;
        public bool frames_shared_at_start = true;
        public bool stay_inside = false;
        public float keyframe_spacing_m = 1.0f, keyframe_turn_deg = 45f;
        public int keyframe_beams = 360, keyframe_max = 400, keyframes_per_message = 2;
        public bool visual_place_linking = true;
        public int visual_lidar_keyframes_per_message = 1, visual_keyframe_max_payload_bytes = 3653;
        public int visual_keyframe_max = 200, visual_keyframes_per_message = 1, visual_link_min_inliers = 16;
        public float visual_keyframe_period_s = 2f;
        public int link_shortlist = 5, link_max_candidates = 20, link_turn_candidates = 3, link_verifications_per_step = 4;
        public float link_max_descriptor_distance = 0.3f, link_window_m = 0.8f, link_step_m = 0.1f, link_window_deg = 12f, link_step_deg = 2f;
        public float link_min_fit = 0.6f, link_agree_m = 0.3f, link_agree_deg = 4f, link_min_separation_m = 1.0f;
        public float keyframe_min_return_share = 0.6f, link_unique_shift_m = 0.3f, link_unique_ratio = 0.85f;
        public float link_coarse_min_fit = 0.35f;
        public bool link_diagnostics = false;
        public float person_rgb_min_confidence = 0.5f, person_rgb_max_range_m = 6f, person_rgb_window_s = 3f, person_rgb_fresh_s = 1.5f, person_rgb_scan_period_s = 1f;
        public int person_rgb_sightings = 2;
        public float person_floor_clearance_m = 0.12f, person_top_max_m = 2.0f;
        public int scan_beam_stride = 3, scan_min_beams = 40;
        public float scan_match_range_m = 8f, scan_sigma_m = 0.05f, scan_window_m = 0.1f, scan_step_m = 0.025f;
        public float scan_window_deg = 1.5f, scan_step_deg = 0.5f, scan_min_fit = 0.3f, scan_softness = 0.004f;
        public float localization_report_s = 5f;
        public float sweep_spacing_m = 3.0f, sweep_repeat_s = 120f, sweep_hold_s = 0.6f, sweep_yaw_dps = 45f, sweep_tolerance_deg = 6f;
        public float area_report_radius_m = 4.0f;
        public int progress_min_cells = 40;
        public float progress_window_s = 15f, progress_skip_radius_m = 2.5f;
        public float progress_min_travel_m = 1f;
        public float burning_radius_m = 2f; public int body_track_poses = 600; public float follow_partner_standoff_m = 2f, teammate_track_max_m = 2.5f, threshold_exit_clear_m = 1.5f, teammate_predict_max_s = 4f, teammate_mask_margin_m = .15f, wait_behind_partner_s = 30f, survivor_view_clear_m = .5f, survivor_keepout_m = 1.05f; public int threshold_look_options = 8;
        public float rendezvous_after_s = 45f, rendezvous_hold_s = 10f; public int rendezvous_min_unshared_cells = 1500;
        public float exterior_dwell_s = 15f, facade_band_m = 2f; public float exit_open_share = 0.9f; public int exit_min_scans = 30, exit_max_count = 4;
        public int frontier_information_radius_cells = 3;
        public float lookahead_m = 0.45f;
        public float goal_tolerance_m = 0.18f, return_arrive_m = 0.5f;
        public float escape_check_s = 2f, escape_fire_margin_m = 2f, escape_ignore_m = 2f, escape_recheck_s = 10f;
        // Off unless brain.json turns it on, so the existing reflex tests keep their exact behaviour.
        public bool route_watch = false;
        public float route_watch_stale_s = 15f, route_watch_max_stale_share = 0.5f, route_watch_min_interval_s = 90f,
            route_watch_dwell_s = 4.5f, route_watch_max_s = 12f;
        public float heat_emergency_fraction = 0.15f, emergency_fire_clearance_m = 0.5f;
        public float heat_cool_after_s = 2f;
        public float open_speed_mps = 0.6f;
        public float near_speed_mps = 0.3f;
        public float near_obstacle_m = 1.0f;
        public float near_cone_deg = 70.0f;
        public float goal_slow_distance_m = 0.6f;
        public float thermal_speed_mps = 0.3f;
        public float turn_in_place_deg = 35.0f;
        public float arc_max_heading_deg = 0f;
        public float heading_gain = 2.0f;
        public float collision_stop_m = 0.5f;
        public float collision_cone_deg = 30.0f;
        public float scan_yaw_dps = 30.0f;
        public float no_frontier_scan_s = 12.0f;
        public float heat_drain_start_c = 50.0f;
        public float heat_drain_per_degree_s = 0.002f;
        public float heat_recover_below_c = 35.0f;
        public float heat_recover_per_s = 0.04f;
        public float heat_retreat_c = 70.0f;
        public float heat_retreat_fraction = 0.25f;
        public float heat_resume_fraction = 0.75f;
        public float retreat_min_distance_m = 0.75f;
        public float mission_time_limit_s = 240.0f;

        public float survivor_stop_yaw_dps = 5.0f;
        public float retreat_start_weight = 0.1f;
        public float retreat_path_weight = 0.01f;

        // Load reads only the limits block, so the brain never depends on the simulator driver.
        [Serializable] public sealed class MotionLimits
        { public float max_forward_mps, max_reverse_mps, max_yaw_rate_dps, max_accel_mps2; }
        [Serializable] sealed class LimitDocument { public MotionLimits limits = null; }
        [NonSerialized] public MotionLimits limits;

        public static BrainConfig Load(string path = null, string sensorPath = null)
        {
            var result = JsonUtility.FromJson<BrainConfig>(File.ReadAllText(path ?? Path.Combine(SensorConfig.ConfigDir, "brain.json")));
            if (result == null) throw new InvalidDataException("Empty brain configuration.");
            result.limits = JsonUtility.FromJson<LimitDocument>(File.ReadAllText(sensorPath ?? Path.Combine(SensorConfig.ConfigDir, "sensors.json")))?.limits;
            result.Validate();
            return result;
        }
        public void Validate()
        {
            foreach (var field in typeof(BrainConfig).GetFields(BindingFlags.Instance | BindingFlags.Public))
            {
                if (field.FieldType == typeof(float))
                {
                    float value = (float)field.GetValue(this);
                    if (float.IsNaN(value) || float.IsInfinity(value)) throw new ArgumentException("Nonfinite brain parameter " + field.Name);
                }
            }
            if (obstacle_cost_margin_m <= 0 || obstacle_cost_weight < 0)
                throw new ArgumentException("Invalid obstacle clearance cost configuration.");
            if (stuck_command_mps <= 0 || stuck_measured_mps < 0 || stuck_measured_mps >= stuck_command_mps ||
                stuck_window_s <= 0 || stuck_strip_depth_m < cell_m ||
                stuck_log_odds < occupied_threshold || stuck_log_odds > log_odds_cap ||
                stuck_reverse_m <= 0 || stuck_reverse_mps <= 0 || stuck_reverse_timeout_s <= 0 ||
                stuck_turn_deg <= 0 || stuck_turn_deg > 90 || stuck_turn_timeout_s <= 0 ||
                stuck_repeat_radius_m < cell_m || stuck_repeat_count < 2 || stuck_blacklist_s <= stuck_window_s)
                throw new ArgumentException("Invalid stuck detection or recovery configuration.");
            if (route_watch_stale_s <= 0 || route_watch_max_stale_share <= 0 || route_watch_max_stale_share >= 1 ||
                route_watch_min_interval_s < 0 || route_watch_dwell_s < 0 || route_watch_max_s <= route_watch_dwell_s)
                throw new ArgumentException("Invalid route watch configuration.");
            if (scan_yaw_gain < 0 || scan_yaw_gain > 1 || collision_margin_m < 0 || lead_max_m < cell_m || agent_avoid_min_radius_m < cell_m || agent_avoid_max_radius_m < agent_avoid_min_radius_m || frontier_cluster_max_m < cell_m * 2 || ceiling_min_height_m <= obstacle_max_height_m || ceiling_radius_m < 0 || ceiling_min_hits < 1 || ceiling_wall_margin_m < 0 ||
                enclosure_rays < 8 || enclosure_range_m <= cell_m || enclosure_min_closed_share <= 0 || enclosure_min_closed_share > 1 || enclosure_max_open_share < 0 || enclosure_max_open_share >= 1 || lidar_open_span_deg <= 0 || fire_cut_blacklist_s < 0 || reenter_after_heat_s < 0 || candidate_wall_extent_m <= 0 || candidate_wall_disc_m <= 0 || candidate_wall_window_m < candidate_wall_disc_m || last_resort_fire_clearance_m <= 0 || last_resort_fire_clearance_m > emergency_fire_clearance_m || air_warm_c <= ambient_c || air_rise_c_per_min <= 0 || air_rise_window_s <= 0 || air_rise_min_delta_c < 0 || air_rise_baseline_s < air_rise_window_s || frontier_peek_m < 0 || search_low_m < 0 || frontier_beyond_m < 0 || reject_new_view_deg < 0 || reject_new_view_deg > 180 || reject_share_s < 0 || frontier_wide_min_m2 < 0 || progress_min_travel_m < 0 || exterior_dwell_s < 0 || facade_band_m <= 0 || exit_open_share <= 0.5f || exit_open_share > 1 || exit_min_scans < 1 || exit_max_count < 0 || frontier_wide_options < 0 || see_over_clearance_m < 0 || carried_person_radius_m <= 0 || search_max_range_m <= 0 ||
                keyframe_spacing_m <= 0 || keyframe_turn_deg <= 0 || keyframe_beams < 36 || keyframe_max < 2 || keyframes_per_message < 1 || keyframes_per_message > 16 || visual_keyframe_max_payload_bytes < 1 || visual_keyframe_max_payload_bytes > BrainMessage.MaxVisualPayloadBytes || visual_lidar_keyframes_per_message < 1 || visual_lidar_keyframes_per_message > keyframes_per_message || visual_keyframe_max < 1 || visual_keyframes_per_message < 0 || visual_keyframes_per_message > 16 || visual_link_min_inliers < 3 || !(visual_keyframe_period_s > 0) ||
                link_shortlist < 1 || link_max_candidates < 2 || link_turn_candidates < 1 || link_verifications_per_step < 1 || link_max_descriptor_distance <= 0 || link_window_m < 0 || link_step_m <= 0 ||
                link_window_deg < 0 || link_step_deg <= 0 || link_min_fit <= 0 || link_min_fit > 1 || link_agree_m <= 0 || link_agree_deg <= 0 || link_min_separation_m < 0 ||
                keyframe_min_return_share < 0 || keyframe_min_return_share > 1 || link_unique_shift_m <= 0 || link_unique_ratio <= 0 || link_unique_ratio >= 1 || link_coarse_min_fit <= 0 || link_coarse_min_fit > link_min_fit ||
                person_rgb_min_confidence <= 0 || person_rgb_min_confidence >= 1 || person_rgb_max_range_m <= 0 || person_rgb_window_s <= 0 ||
                person_rgb_fresh_s <= 0 || person_rgb_scan_period_s <= 0 || person_rgb_sightings < 1 ||
                person_floor_clearance_m < 0 || person_top_max_m <= person_floor_clearance_m || scan_beam_stride < 1 || scan_min_beams < 3 || scan_match_range_m <= 0 ||
                scan_sigma_m <= 0 || scan_window_m < 0 || scan_step_m <= 0 || scan_window_deg < 0 || scan_step_deg <= 0 ||
                scan_min_fit < 0 || scan_min_fit > 1 || scan_softness <= 0 || localization_report_s <= 0 || collision_escape_s <= 0)
                throw new ArgumentException("Invalid LiDAR localization or collision margin configuration.");
            if (full_snapshot_s <= 0 || peer_claim_s <= 0 || peer_claim_radius_m <= 0 ||
                cell_m <= 0 || cell_m > BrainMessage.ResolutionM ||
                Math.Abs(BrainMessage.ResolutionM / cell_m - Math.Round(BrainMessage.ResolutionM / cell_m)) > .0001 || grid_size_m < cell_m * 5 || grid_size_m / cell_m > 1000 || depth_stride < 1 || health_stride < 1 ||
                robot_radius_m <= 0 || free_log_odds >= 0 || occupied_log_odds <= 0 || free_threshold >= 0 || occupied_threshold <= 0 ||
                log_odds_cap < Math.Max(-free_threshold, occupied_threshold) || obstacle_max_height_m <= obstacle_min_height_m ||
                fire_clearance_m < robot_radius_m || heat_cost_radius_m <= fire_clearance_m || heat_cost_weight < 0 ||
                heat_decay_c_per_s < 0 || fire_c <= heat_project_min_c || heat_project_min_c <= ambient_c || heat_prediction_s < 0 || heat_rise_memory_s <= 0 ||
                frame_pair_tolerance_s < 0 || frame_max_age_s <= 0 || motion_stale_s <= 0 || health_max_gap_s <= 0 ||
                corner_target < 1 || contrast_target <= 0 || corner_gradient <= 0 || health_threshold <= 0 || health_threshold >= 1 ||
                health_corner_weight < 0 || health_contrast_weight < 0 || health_depth_weight < 0 ||
                Math.Abs(health_corner_weight + health_contrast_weight + health_depth_weight - 1) > .001 ||
                switch_to_thermal_s <= 0 || switch_to_rgb_s <= 0 || candidate_frames < 1 || candidate_window_s <= 0 ||
                person_min_pixels < 3 || person_max_c <= person_min_c || person_min_area_m2 <= 0 || person_max_area_m2 < person_min_area_m2 || blob_depth_jump_m <= 0 ||
                survivor_merge_m <= 0 || survivor_stop_min_m <= 0 || survivor_stop_min_m > survivor_standoff_m || survivor_standoff_m > survivor_stop_max_m ||
                survivor_dwell_s <= 0 || survivor_fresh_s <= 0 || survivor_lost_s < survivor_fresh_s || survivor_stop_speed_mps < 0 || standoff_samples < 4 ||
                survivor_face_tolerance_deg <= 0 || agent_arrival_hold_s < 0 || escape_time_factor < 0 || escape_rate_window_s <= 0 || survivor_face_tolerance_deg > 90 || replan_s <= 0 || frontier_min_distance_m <= 0 ||
                frontier_gain_weight < 0 || frontier_information_radius_cells < 1 || frontier_information_radius_cells > 20 ||
                lookahead_m <= 0 || goal_tolerance_m <= 0 || goal_slow_distance_m <= 0 || open_speed_mps <= 0 || near_speed_mps <= 0 || thermal_speed_mps <= 0 ||
                near_obstacle_m < collision_stop_m || collision_stop_m <= robot_radius_m || collision_cone_deg <= 0 || collision_cone_deg > 90 ||
                near_cone_deg < collision_cone_deg || near_cone_deg > 90 || turn_in_place_deg <= 0 || arc_max_heading_deg < 0 || arc_max_heading_deg > 90 || heading_gain <= 0 || scan_yaw_dps <= 0 || no_frontier_scan_s <= 0 ||
                survivor_stop_yaw_dps <= 0 || retreat_start_weight < 0 || retreat_path_weight < 0 ||
                heat_drain_per_degree_s < 0 || heat_recover_per_s < 0 || heat_recover_below_c >= heat_drain_start_c || heat_retreat_c <= heat_drain_start_c ||
                heat_retreat_fraction <= 0 || heat_resume_fraction <= heat_retreat_fraction || heat_resume_fraction > 1 || retreat_min_distance_m <= 0 || mission_time_limit_s <= 0 ||
                limits == null || !(limits.max_forward_mps > 0) || !(limits.max_reverse_mps >= 0) || !(limits.max_yaw_rate_dps > 0) || !(limits.max_accel_mps2 > 0) ||
                float.IsInfinity(limits.max_forward_mps + limits.max_reverse_mps + limits.max_yaw_rate_dps + limits.max_accel_mps2))
                throw new ArgumentException("Invalid brain configuration or missing sensor motion limits.");
        }
    }
}
