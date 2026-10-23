// Runs the pure C# NUnit logic tests without starting the Unity editor.
using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
static class LogicTests
{
    static int Main(string[] args)
    {
        int passed = 0, failed = 0;
        var type = typeof(SmokeRescue.Tests.EditMode.BrainTests);
        foreach (var m in type.GetMethods().OrderBy(m => m.Name))
        {
            if (!m.IsDefined(typeof(TestAttribute)) || m.GetParameters().Length != 0 || m.IsDefined(typeof(ExplicitAttribute))) continue;
            if (args.Length > 0 && !args.Any(s => m.Name.Contains(s))) continue;
            var context = NUnit.Framework.Internal.TestExecutionContext.CurrentContext;
            context.CurrentTest = new NUnit.Framework.Internal.TestMethod(new NUnit.Framework.Internal.MethodWrapper(type, m));
            context.CurrentResult = context.CurrentTest.MakeTestResult();
            try { m.Invoke(Activator.CreateInstance(type), null); Console.WriteLine("PASS " + m.Name); passed++; }
            catch (TargetInvocationException e) { Console.WriteLine("FAIL " + m.Name + "\n" + e.InnerException); failed++; }
            if (!string.IsNullOrEmpty(context.CurrentResult.Output)) Console.Write(context.CurrentResult.Output);
        }
        Console.WriteLine($"{passed} passed, {failed} failed");
        return failed == 0 && passed > 0 ? 0 : 1;
    }
}
