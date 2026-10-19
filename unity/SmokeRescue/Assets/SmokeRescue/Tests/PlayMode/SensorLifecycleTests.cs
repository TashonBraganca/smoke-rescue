using System.Collections;
using NUnit.Framework;
using SmokeRescue.Sensors;
using UnityEngine;
using UnityEngine.TestTools;

namespace SmokeRescue.Tests.PlayMode
{
    public class SensorLifecycleTests
    {
        [UnityTest]
        public IEnumerator SensorCameraNeverRendersWithoutAnExplicitRequest()
        {
            var mount = new GameObject("C4 lifecycle camera");
            try
            {
                var camera = mount.AddComponent<RgbdCamera>();
                yield return null;
                Assert.That(camera.Camera.enabled, Is.False);
                Assert.That(camera.IsCapturing, Is.False);
            }
            finally { Object.Destroy(mount); }
        }
    }
}
