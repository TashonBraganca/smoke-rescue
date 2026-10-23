// Headless simulator outside the editor: tools/headless/run.sh <request.json>
using System;
using System.IO;
static class Program
{
    static int Main(string[] args)
    {
        string root = Environment.GetEnvironmentVariable("SMOKE_REPO") ?? throw new ArgumentException("Set SMOKE_REPO.");
        var request = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(args[0]));
        if ((string)request["kind"] == "sensor_replay") { SensorReplay.Run(request); return 0; }
        foreach (var line in SmokeRescue.Tests.EditMode.HeadlessSim.RunRequest(root, Path.GetFullPath(args[0]))) Console.WriteLine(line);
        return 0;
    }
}
