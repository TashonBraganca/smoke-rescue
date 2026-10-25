# Stand-in for the ryor engine image (tests only; runs in public.ecr.aws/aws-cli/aws-cli). Writes a contract-shaped
# replay (CONTRACT.txt section 3) for SETUP to s3://$BUCKET/$PREFIX: 18 chunks of 20 frames, STANDIN_SLEEP s apart,
# then a final "done" status. Robots stand still 1 m past the pads. Deployed as the ryor stack's StandInScript.
set -e; export AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}; r=${SETUP#*__r}; r=${r%%__*}; P="s3://$BUCKET/$PREFIX"; L=(A B C)
st() { printf '{"v":1,"mission":"%s","setup":"%s","state":"%s","source":"live","chunks":%s,"sim_t":%s,"updated":"%s","error":null,"score":%s,"engine":{"git":"standin","image":"%s"}}' "$MISSION_ID" "$SETUP" "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" "$IMAGE_TAG" > /tmp/s.json; aws s3 cp --quiet --content-type application/json /tmp/s.json "${P}status.json"; }
echo "standin start $MISSION_ID $SETUP robots=$r"; st starting 0 0 null
for c in $(seq 0 17); do
  f=""; for k in $(seq 0 19); do h=$((c*20+k)); rb=""
    for i in $(seq 0 $((r-1))); do rb="$rb{\"id\":\"robot_${L[$i]}\",\"x\":$i.2,\"y\":1.0,\"yaw\":90,\"state\":\"explore\",\"out\":false},"; done
    f="$f{\"t\":$((h/2)).$((h%2*5)),\"robots\":[${rb%,}],\"seen\":{},\"fire\":[],\"found\":[],\"events\":[]},"; done
  printf '{"v":1,"chunk":%d,"t0":%d,"frames":[%s]}' "$c" "$((c*10))" "${f%,}" > /tmp/c.json
  aws s3 cp --quiet --content-type application/json /tmp/c.json "$(printf '%schunk_%04d.json' "$P" "$c")"
  st running $((c+1)) $((c*10+10)) null; echo "standin chunk $c"; sleep "$STANDIN_SLEEP"
done
st done 18 180 '{"people_found":0,"people_total":0,"time_s":180,"coverage_pct":0,"robots_out":0,"false_confirmations":0}'; echo "standin done"
