using System;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Recording;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        sealed class PlaceHandler : HttpMessageHandler
        {
            public Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send;
            protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken token) => send(request, token);
        }
        static HttpResponseMessage PlaceResponse(JObject body) => new HttpResponseMessage(HttpStatusCode.OK)
            { Content = new StringContent(body.ToString()) };

        [Test]
        public void PlaceServiceDownOrSlowDoesNotBlockTheRecorderPump()
        {
            foreach (bool slow in new[] { true, false })
            {
                var f = CameraScene();
                var handler = new PlaceHandler { send = async (request, token) =>
                {
                    if (slow) await Task.Delay(Timeout.Infinite, token);
                    throw new HttpRequestException("service unavailable");
                } };
                using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", .05f, handler)))
                {
                    Assert.That(worker.TryCapture(0, new BrainPose()), Is.True);
                    worker.Step(.01);
                    // Completion is applied only by the simulation thread, so poll through Step until the deadline.
                    for (int i = 0; i < 60 && worker.Busy; i++) { Thread.Sleep(5); worker.Step(.02); }
                    Assert.That(worker.Busy, Is.False, "the configured timeout or connection failure ends the request");
                    Assert.That(f.brain.DrainEvents(), Has.Some.Matches<BrainEvent>(e => e.kind == "place_service"));
                    Assert.That(f.brain.TryPartnerLink("A", out _), Is.False);
                }
            }
            var off = CameraScene(enabled: false); int requests = 0;
            var control = new PlaceHandler { send = (request, token) => { requests++; throw new Exception("must not call"); } };
            using (var worker = new VisualPlaceWorker(off.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", .05f, control)))
            {
                Assert.That(worker.TryCapture(0, new BrainPose()), Is.False); worker.Step(1);
                Assert.That(requests, Is.Zero, "control: the disabled feature makes no service requests");
            }
        }

        [Test]
        public void PlaceServiceRepliesFeedTheVerifiedFrameLinkPath()
        {
            var f = CameraScene();
            // A new own view requeues the partner views stored by CameraScene.
            int match = 0;
            var handler = new PlaceHandler { send = async (request, token) =>
            {
                var body = JObject.Parse(await request.Content.ReadAsStringAsync());
                if (request.RequestUri.AbsolutePath.EndsWith("keyframe"))
                    return PlaceResponse(new JObject { ["ok"] = true, ["id"] = 7, ["database_id"] = "db1", ["t"] = body["t"],
                        ["payload_b64"] = Convert.ToBase64String(VisualPayload()) });
                return PlaceResponse(new JObject { ["matched"] = true, ["database_id"] = "db1", ["mine_id"] = 7,
                    ["dx_m"] = 0, ["dy_m"] = match++ == 0 ? 0 : 1.5, ["dyaw_deg"] = 0, ["inliers"] = 30, ["rmse_m"] = .02 });
            } };
            using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f, handler)))
            {
                worker.TryCapture(0, new BrainPose()); worker.Step(.01); worker.Step(.02);
                Assert.That(f.brain.TryPartnerLink("A", out _), Is.False, "one HTTP match still waits for an independent place");
                worker.Step(.03);
                Assert.That(f.brain.TryPartnerLink("A", out var link), Is.True);
                Assert.That(Vector2.Distance(link.position, f.truth.position), Is.LessThan(.01f));
                Assert.That(match, Is.EqualTo(2));
            }
        }

        [Test]
        public void PlaceWorkerRestartAndStaleFrameRepliesCannotReuseOldPoses()
        {
            var f = CameraScene(); int capture = 0;
            var handler = new PlaceHandler { send = async (request, token) =>
            {
                var body = JObject.Parse(await request.Content.ReadAsStringAsync());
                return PlaceResponse(new JObject { ["ok"] = true, ["id"] = 0, ["database_id"] = "db" + capture++,
                    ["t"] = body.Value<double>("t") - (capture == 3 ? .1 : 0), ["payload_b64"] = Convert.ToBase64String(VisualPayload()) });
            } };
            using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f, handler)))
            {
                worker.TryCapture(0, new BrainPose()); worker.Step(2);
                var first = f.brain.BuildMessage(2).VisualKeyframes[0];
                worker.TryCapture(2, new BrainPose { position = Vector2.one }); worker.Step(4);
                var second = f.brain.BuildMessage(4).VisualKeyframes[0];
                Assert.That(second.id, Is.GreaterThan(first.id), "a restarted service id 0 gets a fresh radio id");
                Assert.That(second.pose.position, Is.EqualTo(Vector2.one));
                worker.TryCapture(4, new BrainPose { position = Vector2.one * 2 }); worker.Step(6);
                var unchanged = f.brain.BuildMessage(6).VisualKeyframes[0];
                Assert.That(unchanged.id, Is.EqualTo(second.id), "a nearby but different image stamp cannot take the capture pose");
                Assert.That(f.brain.DrainEvents(), Has.Some.Matches<BrainEvent>(e => e.kind == "place_service"));
            }
        }
        [Test]
        public void MalformedPlaceResponsesAreLoggedWithoutStoppingThePump()
        {
            foreach (bool invalidJson in new[] { true, false })
            {
                var f = CameraScene();
                var handler = new PlaceHandler { send = (request, token) => Task.FromResult(invalidJson
                    ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("truncated {") }
                    : PlaceResponse(new JObject { ["ok"] = true })) };
                using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f, handler)))
                {
                    worker.TryCapture(0, new BrainPose());
                    Assert.DoesNotThrow(() => worker.Step(.01));
                    Assert.That(f.brain.TryPartnerLink("A", out _), Is.False);
                    Assert.That(f.brain.DrainEvents(), Has.Some.Matches<BrainEvent>(e => e.kind == "place_service"));
                }
            }
            var off = CameraScene(enabled: false);
            using (var worker = new VisualPlaceWorker(off.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f)))
                Assert.That(worker.TryCapture(0, new BrainPose()), Is.False, "control: disabled requests leave the old result");
        }
        [Test]
        public void AnUnfinishedPlaceRequestDoesNotBlockCaptureOrPolling()
        {
            var f = CameraScene();
            var gate = new TaskCompletionSource<HttpResponseMessage>();
            var handler = new PlaceHandler { send = (request, token) => gate.Task };
            using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f, handler)))
            {
                Task<bool> capture = null; Task poll = null;
                bool captureReturned = false, pollReturned = false;
                try
                {
                    capture = Task.Run(() => worker.TryCapture(0, new BrainPose()));
                    captureReturned = capture.Wait(1000);
                    if (captureReturned)
                    {
                        poll = Task.Run(() => worker.Step(.01)); pollReturned = poll.Wait(1000);
                    }
                }
                finally { gate.TrySetResult(PlaceResponse(new JObject { ["ok"] = false })); capture?.Wait(); poll?.Wait(); }
                Assert.That(captureReturned, Is.True, "capture returns before the service response exists");
                Assert.That(pollReturned, Is.True, "polling returns before the service response exists");
                Assert.That(capture.Result, Is.True);
                worker.Step(.02);
                Assert.That(worker.Busy, Is.False, "control: a completed response is applied");
            }
        }
        [Test]
        public void ASubsecondCameraReplyUsesTheRecordersExactImageStamp()
        {
            var f = CameraScene(); double captureTime = 2 + 1 / 15.0; const double imageStamp = 2.066667;
            var pose = new BrainPose { position = new Vector2(2, 3), yawRad = .4f };
            var handler = new PlaceHandler { send = (request, token) => Task.FromResult(PlaceResponse(new JObject
            {
                ["ok"] = true, ["id"] = 0, ["database_id"] = "subsecond", ["t"] = imageStamp,
                ["payload_b64"] = Convert.ToBase64String(VisualPayload())
            })) };
            using (var worker = new VisualPlaceWorker(f.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f, handler)))
            {
                Assert.That(worker.TryCapture(captureTime, pose), Is.True); worker.Step(3);
                var sent = f.brain.BuildMessage(3).VisualKeyframes.Single();
                Assert.That(sent.pose.position, Is.EqualTo(pose.position)); Assert.That(sent.pose.yawRad, Is.EqualTo(pose.yawRad));
            }
            var off = CameraScene(enabled: false);
            using (var worker = new VisualPlaceWorker(off.brain, "B", "/tmp/own-camera", new VisualPlaceClient("http://place/", 1f)))
                Assert.That(worker.TryCapture(captureTime, pose), Is.False, "control: disabled capture leaves the old result");
        }
    }
}
