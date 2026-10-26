using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;
namespace SmokeRescue.Tests.EditMode
{
 public sealed partial class BrainTests
 {
  // Codex deep-bug report 26 Oct (handoff/codex_deep_bugs_report.txt); passes since the DB01/DB02 fixes.
  [Test] public void DeepAudit_TrailAdmissionChecksDensifiedFire()
  {
   var c=Settings(); c.grid_size_m=12; c.cell_m=.1f; c.robot_radius_m=.3f; c.breadcrumb_spacing_m=1f;
   var b=new RobotBrain(Calibration(),c,"A",0); var g=Map(b);
   var flags=BindingFlags.Instance|BindingFlags.NonPublic;
   for(int i=0;i<g.Count;i++)g.ObserveFree(i);
   var heat=(float[])typeof(OccupancyGrid).GetField("heat",flags).GetValue(g);
   heat[g.Cell(new Vector2(0,-.5f))]=500;
   g.PrepareNavigation();
   var crumbs=(List<Vector2>)typeof(RobotBrain).GetField("breadcrumbs",flags).GetValue(b);
   crumbs.Add(new Vector2(0,-1));crumbs.Add(Vector2.zero);
   typeof(RobotBrain).GetField("home",flags).SetValue(b,new Vector2(0,-1));
   var route=(List<Vector2>)typeof(RobotBrain).GetMethod("BreadcrumbRoute",flags).Invoke(b,null);
   var valid=route.Count==0||(bool)typeof(RobotBrain).GetMethod("TrailPathOk",flags).Invoke(b,new object[]{route,0});
   Console.WriteLine($"trail admitted points={route.Count}, complete fire check={valid}");
   Assert.That(valid,Is.True,"A fallback must be checked before the first command, including fire between sparse crumbs.");
  }

  // Codex deep-bug report 26 Oct (handoff/codex_deep_bugs_report.txt); passes since the DB01/DB02 fixes.
  [Test] public void DeepAudit_ExpiredTrialDoesNotResetWhenInterpolatedFireInvalidatesIt()
  {
   var c=Settings();c.grid_size_m=12;c.cell_m=.1f;c.robot_radius_m=.3f;c.breadcrumb_spacing_m=1f;c.breadcrumb_join_m=1.5f;
   c.stay_inside=true;c.search_requires_ceiling=false;c.return_home_at_end=true;c.mission_time_limit_s=1000;c.end_return_breadcrumbs=true;
   var b=new RobotBrain(Calibration(),c,"A",0);var g=Map(b);var flags=BindingFlags.Instance|BindingFlags.NonPublic;
   for(int i=0;i<g.Count;i++)
   {var p=g.Centre(i); if(Mathf.Abs(p.x)<.4f&&p.y<.5f&&p.y> -2)g.ObserveFree(i);else g.ObserveOccupied(i);}
   var heat=(float[])typeof(OccupancyGrid).GetField("heat",flags).GetValue(g);heat[g.Cell(new Vector2(0,-.5f))]=500;g.PrepareNavigation();
   typeof(RobotBrain).GetField("home",flags).SetValue(b,new Vector2(0,-1));
   ((GridPlanner)typeof(RobotBrain).GetField("planner",flags).GetValue(b)).Home=new Vector2(0,-1);
   var crumbs=(List<Vector2>)typeof(RobotBrain).GetField("breadcrumbs",flags).GetValue(b);crumbs.Add(new Vector2(0,-1));crumbs.Add(Vector2.zero);
   typeof(RobotBrain).GetMethod("BeginReturn",flags).Invoke(b,new object[]{0d,"Mission time is nearly up: leave now to be out by the deadline."});
   var all=new List<BrainEvent>();
   for(int i=0;i<40;i++){DriveTick(b,i);all.AddRange(b.DrainEvents());}
   int installs=all.Count(e=>e.kind=="return_via_trail");Console.WriteLine($"invalid trail installs over 2 s={installs}");
   Assert.That(installs,Is.LessThanOrEqualTo(1),"An unsafe route must enter blocked handling instead of reinstalling every tick.");
  }

  // Codex deep-bug report 26 Oct (handoff/codex_deep_bugs_report.txt); passes since the DB01/DB02 fixes.
  [Test] public void DeepAudit_ClearWayRetriesAfterUnmappedSideTargets()
  {
   var c=Settings();c.grid_size_m=12;c.search_requires_ceiling=false;c.stay_inside=true;c.return_home_at_end=true;c.end_clear_way_out_m=1.5f;c.end_clear_way_side_m=1.5f;
   var b=new RobotBrain(Calibration(),c,"A",0);var g=Map(b);
   for(int i=0;i<g.Count;i++){var p=g.Centre(i);if(Mathf.Abs(p.x)<.5f&&Mathf.Abs(p.y)<.5f)g.ObserveFree(i);}
   var flags=BindingFlags.Instance|BindingFlags.NonPublic;
   typeof(RobotBrain).GetMethod("BeginReturn",flags).Invoke(b,new object[]{0d,"Mission time is nearly up: leave now to be out by the deadline."});
   DriveTick(b,0);bool tried=(bool)typeof(RobotBrain).GetField("clearWayTried",flags).GetValue(b);
   Console.WriteLine($"no lateral route, clearWayTried={tried}, state={b.Snapshot().state}");
   Assert.That(tried&&b.Snapshot().state=="complete",Is.False,"A failed step-aside plan must remain pending so later mapping can permit a retry.");
  }

 }
}

