using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SmokeRescue.Brain;

namespace SmokeRescue.Recording
{
    /// <summary>One bounded HTTP request at a time per robot. No Unity calls run on the HTTP continuation thread.</summary>
    public sealed class VisualPlaceClient : IDisposable
    {
        public sealed class Response { public JObject body; public string error; }
        readonly HttpClient http;
        readonly Uri baseUri;
        readonly float timeoutS;
        public VisualPlaceClient(string baseUrl, float timeoutSeconds, HttpMessageHandler handler = null)
        {
            if (!(timeoutSeconds > 0) || float.IsInfinity(timeoutSeconds)) throw new ArgumentException("Invalid place HTTP timeout.");
            baseUri = new Uri(baseUrl.TrimEnd('/') + "/"); timeoutS = timeoutSeconds;
            http = handler == null ? new HttpClient() : new HttpClient(handler);
            http.Timeout = Timeout.InfiniteTimeSpan;
        }
        public async Task<Response> Post(string path, JObject body)
        {
            try
            {
                using (var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutS)))
                using (var content = new StringContent(body.ToString(Formatting.None), Encoding.UTF8, "application/json"))
                using (var response = await http.PostAsync(new Uri(baseUri, path.TrimStart('/')), content, timeout.Token).ConfigureAwait(false))
                {
                    string text = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                    if (!response.IsSuccessStatusCode) return new Response { error = "HTTP " + (int)response.StatusCode };
                    return new Response { body = JObject.Parse(text) };
                }
            }
            catch (Exception e) when (e is HttpRequestException || e is OperationCanceledException || e is JsonException || e is ObjectDisposedException)
            { return new Response { error = e.GetBaseException().Message }; }
        }
        public void Dispose() => http.Dispose();
    }

    /// <summary>The recorder polls completions on its simulation thread. A delayed place service never stalls a tick.</summary>
    public sealed class VisualPlaceWorker : IDisposable
    {
        readonly RobotBrain brain;
        readonly VisualPlaceClient client;
        readonly string robot, directory;
        readonly List<(string sender, BrainVisualKeyframe keyframe)> queue = new List<(string, BrainVisualKeyframe)>();
        readonly Dictionary<int, int> serviceToRadio = new Dictionary<int, int>();
        Task<VisualPlaceClient.Response> pending;
        bool capturing, disposed;
        double lastCapture = double.NegativeInfinity, captureStamp;
        BrainPose capturePose;
        string database;
        int nextRadioId;
        (string sender, BrainVisualKeyframe keyframe) matching;
        // visual_slam_live loop closure: the newest own view, matched against this robot's own earlier views of the place.
        (int radioId, byte[] payload, List<int> serviceIds)? loopJob;
        bool loopMatching; int loopRadioId;
        public bool Busy => pending != null;
        public VisualPlaceWorker(RobotBrain robotBrain, string robotId, string robotDir, VisualPlaceClient placeClient)
        { brain = robotBrain; robot = Uri.EscapeDataString(robotId); directory = robotDir; client = placeClient; }
        public bool CaptureDue(double t) => !disposed && brain.VisualPlacesEnabled && !Busy && t - lastCapture >= brain.VisualKeyframePeriodS;
        JObject Body(double t) => new JObject { ["robot_dir"] = directory, ["t"] = t };
        public bool TryCapture(double t, BrainPose pose)
        {
            if (!CaptureDue(t)) return false;
            lastCapture = t; capturePose = pose;
            // Use the recorder's exact index formatting (six decimals, at most 0.5 microseconds rounding). A nearest
            // older image must not inherit this capture pose, including captures between whole simulation seconds.
            captureStamp = double.Parse(RecordingSession.Fmt(t), CultureInfo.InvariantCulture);
            capturing = true; loopMatching = false; pending = client.Post("/place/" + robot + "/keyframe", Body(captureStamp));
            return true;
        }
        public void Step(double t)
        {
            if (disposed || !brain.VisualPlacesEnabled) return;
            if (pending != null && pending.IsCompleted)
            {
                // Result is read only after completion. HTTP errors are values, including timeout and malformed JSON.
                var result = pending.GetAwaiter().GetResult(); pending = null;
                if (result.error != null) Log(t, result.error);
                else try { Apply(result.body, t); }
                catch (Exception e) when (e is ArgumentException || e is FormatException || e is InvalidCastException || e is InvalidOperationException || e is OverflowException || e is JsonException)
                { Log(t, "Ignored invalid place response: " + e.Message); }
            }
            foreach (var item in brain.DrainPartnerVisualKeyframes())
            {
                if (!brain.VisualMatchWanted(item.sender) || queue.Exists(x => x.sender == item.sender && x.keyframe.id == item.keyframe.id)) continue;
                queue.Add(item); if (queue.Count > brain.VisualKeyframeCapacity) queue.RemoveAt(0);
            }
            if (Busy || CaptureDue(t)) return;  // a due camera capture gets the next slot
            while (queue.Count > 0)
            {
                matching = queue[0]; queue.RemoveAt(0);
                if (!brain.VisualMatchWanted(matching.sender)) continue;
                // With a link (the operator's or a measured one) only own views near where the partner stood are compared.
                var near = brain.VisualCandidates(matching.sender, matching.keyframe.pose);
                var ids = near == null ? null : ServiceIds(near);
                if (ids != null && ids.Count == 0) continue;   // no own view there yet; it is queued again with the next own view
                capturing = false; loopMatching = false;
                var body = Body(t); body["payload_b64"] = Convert.ToBase64String(matching.keyframe.payload);
                if (ids != null) body["candidates"] = new JArray(ids);
                pending = client.Post("/place/" + robot + "/match", body); return;
            }
            if (loopJob.HasValue)
            {
                var job = loopJob.Value; loopJob = null;
                capturing = false; loopMatching = true; loopRadioId = job.radioId;
                var body = Body(t); body["payload_b64"] = Convert.ToBase64String(job.payload); body["candidates"] = new JArray(job.serviceIds);
                pending = client.Post("/place/" + robot + "/match", body);
            }
        }
        List<int> ServiceIds(List<int> radioIds)
        {
            var ids = new List<int>();
            foreach (var pair in serviceToRadio) if (radioIds.Contains(pair.Value)) ids.Add(pair.Key);
            return ids;
        }
        static JToken Required(JObject reply, string field)
        {
            var value = reply?[field];
            if (value == null || value.Type == JTokenType.Null) throw new FormatException("Missing " + field);
            return value;
        }
        void Apply(JObject reply, double t)
        {
            if (capturing)
            {
                if (reply.Value<bool?>("ok") != true) return;
                string identity = Required(reply, "database_id").Value<string>();
                if (string.IsNullOrEmpty(identity)) throw new FormatException("Empty place database identity.");
                if (Required(reply, "t").Value<double>() != captureStamp) throw new FormatException("Place service returned another image timestamp.");
                int serviceId = Required(reply, "id").Value<int>();
                var payload = Convert.FromBase64String(Required(reply, "payload_b64").Value<string>());
                if (database != identity)
                {
                    brain.ResetVisualKeyframes(); serviceToRadio.Clear(); database = identity;
                }
                if (serviceToRadio.ContainsKey(serviceId)) return;
                int radioId = nextRadioId++;
                brain.AddVisualKeyframe(radioId, capturePose, payload, captureStamp); serviceToRadio[serviceId] = radioId;
                // Its own earlier views of this place, if any, are the loop-closure candidates (newest job replaces an unsent one).
                var earlier = ServiceIds(brain.LoopCandidates(radioId));
                if (earlier.Count > 0) loopJob = (radioId, payload, earlier);
                // The service retains 300 views, the brain retains 200. Old ids may still be returned; the brain rejects
                // them if their capture pose has expired. Bound this lookup too by the same configured capacity.
                if (serviceToRadio.Count > brain.VisualKeyframeCapacity)
                {
                    int oldest = -1, radio = int.MaxValue;
                    foreach (var pair in serviceToRadio) if (pair.Value < radio) { oldest = pair.Key; radio = pair.Value; }
                    serviceToRadio.Remove(oldest);
                }
            }
            else if (loopMatching)
            {
                if (reply.Value<bool?>("matched") != true) return;
                if (Required(reply, "database_id").Value<string>() != database) return;
                if (!serviceToRadio.TryGetValue(Required(reply, "mine_id").Value<int>(), out int oldRadio)) return;
                brain.OnVisualLoopClosure(loopRadioId, oldRadio, Required(reply, "dx_m").Value<float>(), Required(reply, "dy_m").Value<float>(),
                    Required(reply, "dyaw_deg").Value<float>(), Required(reply, "inliers").Value<int>(), t);
            }
            else
            {
                if (reply.Value<bool?>("matched") != true || !brain.VisualMatchWanted(matching.sender)) return;
                if (Required(reply, "database_id").Value<string>() != database) { Log(t, "Place worker restarted; waiting for a fresh own keyframe."); return; }
                if (!serviceToRadio.TryGetValue(Required(reply, "mine_id").Value<int>(), out int mineId)) return;
                brain.OnVisualPlaceMatch(matching.sender, matching.keyframe.id, mineId, Required(reply, "dx_m").Value<float>(),
                    Required(reply, "dy_m").Value<float>(), Required(reply, "dyaw_deg").Value<float>(), Required(reply, "inliers").Value<int>(), t);
            }
        }
        void Log(double t, string reason) => brain.Log(t, "place_service", reason);
        public void Dispose()
        {
            disposed = true; client.Dispose(); queue.Clear();
        }
    }
}
