#!/usr/bin/env python3
"""Precompute the recorded replay of every gallery setup on Fargate, with the same engine image the live runs use.

    python3 tools/ryor/precompute.py <image tag> [--parallel 12] [--only key1,key2] [--dry]

Each setup runs as one Fargate task of the task family smoke-rescue-ryor-precompute (registered here from the live family
smoke-rescue-ryor: same roles, log group and size, this image). The task writes to ryor/missions/pre-<tag>/<setup>/ (the only
prefix the task role may write); a finished, valid replay is then copied to ryor/cache/<setup>/ with this user's credentials.
A setup whose ryor/cache/<setup>/status.json is already "done" with this engine tag is skipped. Writes a summary to
data/runs/ryor/precompute_<tag>.json with wall times per setup.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
REGION, CLUSTER, LIVE_FAMILY, FAMILY = "us-east-1", "smoke-rescue", "smoke-rescue-ryor", "smoke-rescue-ryor-precompute"
BUCKET = "smoke-rescue-ryor-runs-968000946565"
REPO = "968000946565.dkr.ecr.us-east-1.amazonaws.com/smoke-rescue/ryor-engine"
SUBNETS = ["subnet-0cc6300638b000910", "subnet-0dc74c2f3b81a8eb6"]
SG = "sg-0001f7d862e5e78ec"
ecs, s3 = boto3.client("ecs", region_name=REGION), boto3.client("s3", region_name=REGION)


def register(tag):
    live = ecs.describe_task_definition(taskDefinition=LIVE_FAMILY)["taskDefinition"]
    c = dict(live["containerDefinitions"][0])
    c["image"] = f"{REPO}:{tag}"
    c["environment"] = [{"name": "BUCKET", "value": BUCKET}, {"name": "HARD_LIMIT_S", "value": "900"}]
    keep = ("name", "image", "essential", "environment", "logConfiguration", "stopTimeout", "readonlyRootFilesystem", "user")
    c = {k: v for k, v in c.items() if k in keep}
    td = ecs.register_task_definition(family=FAMILY, taskRoleArn=live["taskRoleArn"], executionRoleArn=live["executionRoleArn"],
                                      networkMode="awsvpc", requiresCompatibilities=["FARGATE"], cpu="4096", memory="8192",
                                      runtimePlatform=live["runtimePlatform"], containerDefinitions=[c],
                                      tags=[{"key": "Project", "value": "smoke-rescue"}])
    return td["taskDefinition"]["taskDefinitionArn"]


def get_json(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def copy_to_cache(tag, key):
    src = f"ryor/missions/pre-{tag}/{key}/"
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=src)["Contents"]
    for o in objs:
        name = o["Key"][len(src):]
        s3.copy_object(Bucket=BUCKET, Key=f"ryor/cache/{key}/{name}", CopySource={"Bucket": BUCKET, "Key": o["Key"]},
                       ContentType="application/json", CacheControl="no-store", MetadataDirective="REPLACE")
    return len(objs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    keys = sorted(p.stem for p in (ROOT / "data/runs/ryor/setups").glob("*.json"))
    if args.only:
        keys = [k for k in keys if k in set(args.only.split(","))]
    todo = []
    for k in keys:
        st = get_json(f"ryor/cache/{k}/status.json")
        if not (st and st.get("state") == "done" and st.get("engine", {}).get("image") == args.tag):
            todo.append(k)
    print(f"{len(keys)} setups, {len(todo)} to run with engine {args.tag}", flush=True)
    if args.dry or not todo:
        return
    td = register(args.tag)
    print("task definition", td, flush=True)
    running, results, queue = {}, {}, list(todo)
    summary_path = ROOT / f"data/runs/ryor/precompute_{args.tag}.json"
    while queue or running:
        while queue and len(running) < args.parallel:
            k = queue.pop(0)
            r = ecs.run_task(cluster=CLUSTER, taskDefinition=td, launchType="FARGATE", count=1, startedBy="ryor-precompute",
                             networkConfiguration={"awsvpcConfiguration": {"subnets": SUBNETS, "securityGroups": [SG], "assignPublicIp": "ENABLED"}},
                             overrides={"containerOverrides": [{"name": "engine", "environment": [
                                 {"name": "SETUP", "value": k}, {"name": "MISSION_ID", "value": "cache"},
                                 {"name": "PREFIX", "value": f"ryor/missions/pre-{args.tag}/{k}"}]}]})
            if r.get("failures"):
                print(f"{k}: run_task failure {r['failures']}; retry later", flush=True)
                queue.append(k)
                time.sleep(10)
                break
            running[k] = (r["tasks"][0]["taskArn"], time.time())
        time.sleep(15)
        arns = [v[0] for v in running.values()]
        for i in range(0, len(arns), 100):
            for t in ecs.describe_tasks(cluster=CLUSTER, tasks=arns[i:i + 100])["tasks"]:
                if t["lastStatus"] != "STOPPED":
                    continue
                k = next(k for k, v in running.items() if v[0] == t["taskArn"])
                started = running.pop(k)[1]
                st = get_json(f"ryor/missions/pre-{args.tag}/{k}/status.json") or {}
                code = t["containers"][0].get("exitCode")
                rec = {"state": st.get("state"), "exit": code, "reason": t.get("stoppedReason"), "error": st.get("error"),
                       "wall_from_launch_s": round(time.time() - started), "engine_wall": st.get("wall"), "score": st.get("score")}
                if st.get("state") == "done" and code == 0:
                    rec["copied"] = copy_to_cache(args.tag, k)
                results[k] = rec
                print(f"{k}: {rec['state']} exit {code} in {rec['wall_from_launch_s']} s; engine {st.get('wall')}; "
                      f"{st.get('score') or rec['error'] or rec['reason']}", flush=True)
                summary_path.write_text(json.dumps(results, indent=1))
    ok = sum(1 for r in results.values() if r.get("copied"))
    print(f"done: {ok} of {len(results)} setups cached; summary {summary_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
