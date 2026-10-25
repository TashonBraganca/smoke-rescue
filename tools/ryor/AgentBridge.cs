        // Four decision points per robot leave time for answers before the 140 s exit limit.
        // Poll at replay cadence. Files are atomically renamed by the Python worker.
        void AgentTick(StreamWriter log, List<Body> robots, double t, int k)
        {
            if (k % 30 != 0) return;
            foreach (var r in robots)
            {
                string file = Path.Combine(agentDir, r.id + ".order.json");
                if (File.Exists(file))
                {
                    var answer = JObject.Parse(File.ReadAllText(file)); File.Delete(file);
                    double asked = (double)answer["asked_t"];
                    bool fresh = t - asked <= 30 && t < 140;
                    string text = (string)answer["reason"] ?? "Terra left the brain searching.";
                    if (fresh && (string)answer["type"] == "explore_frontier")
                    {
                        // AgentExplore rechecks indoor evidence, heat, return state and partner claims.
                        var at = new Vector2((float)answer["x"], (float)answer["y"]);
                        r.brain.AgentExplore(at, t, 30, text);
                    }
                    string kind = (string)answer["type"] == "unavailable" ? "agent_unavailable" : "agent_decision";
                    if (!fresh) text = "Terra answered after this decision expired. The brain kept searching.";
                    log.WriteLine(new JObject { ["t"] = t, ["kind"] = kind, ["robot"] = r.id,
                        ["reason"] = text, ["asked_t"] = asked, ["fresh"] = fresh,
                        ["order"] = answer, ["model"] = answer["model"] }.ToString(Newtonsoft.Json.Formatting.None));
                }
                if (t < 15 || t > 105 || ((int)Math.Round(t) - 15) % 30 != 0) continue;
                var s = r.brain.Snapshot();
                if (s.state == "return" || s.state == "retreat" || s.state == "complete") continue;
                var options = new JArray(r.brain.FrontierOptions(t, 8).Where(f => f.provenIndoors).Select(f =>
                    new JObject { ["id"] = f.id, ["x"] = f.position.x, ["y"] = f.position.y,
                        ["path_m"] = f.pathM, ["unsearched_m2"] = f.unsearchedM2,
                        ["mapped_unsearched_m2"] = f.mappedUnsearchedM2, ["room_id"] = f.roomId }));
                var rooms = new JArray(r.brain.RoomOptions(t).Take(24).Select(q => new JObject {
                    ["id"] = q.id, ["covered_share"] = q.coveredShare, ["floor_m2"] = q.indoorFloorM2,
                    ["entered"] = q.entered, ["corridor"] = q.corridor, ["partner_inside"] = q.partnerInside }));
                log.WriteLine(new JObject { ["t"] = t, ["kind"] = "agent_observation", ["robot"] = r.id,
                    ["pose"] = new JObject { ["x"] = s.pose.position.x, ["y"] = s.pose.position.y,
                        ["heading_deg"] = s.pose.yawRad * Mathf.Rad2Deg },
                    ["state"] = s.state, ["heat_budget"] = s.heatBudget,
                    ["mission_time_left_s"] = r.brain.MissionTimeLeft(t), ["frontiers"] = options, ["rooms"] = rooms,
                    ["people_found"] = new JArray(s.confirmedSurvivors.Select(p => new JArray(p.x, p.y))),
                    ["sensed_fire"] = new JArray(s.fireHotspots.Take(32).Select(p => new JArray(p.x, p.y))),
                    ["radio_peers"] = new JArray(s.peers.Select(p => new JObject {
                        ["id"] = p.robot, ["linked"] = p.linked, ["sent_at"] = p.sentAt,
                        ["x"] = p.pose.position.x, ["y"] = p.pose.position.y }))
                }.ToString(Newtonsoft.Json.Formatting.None));
                log.Flush();
            }
        }
