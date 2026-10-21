"""Write the architecture and agent-workflow diagrams as plain SVG (no Graphviz needed): diagrams/*.svg.

    python tools/make_diagrams.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / 'diagrams'
INK, MUTED, LINE = '#1d2433', '#5b6475', '#8a93a6'
FILL = {'sim': '#eef4fb', 'brain': '#f1f7ee', 'agent': '#fbf3ea', 'eval': '#f3f0f8', 'aws': '#f4f4f4'}
EDGE = {'sim': '#5b8cc9', 'brain': '#5f9a52', 'agent': '#c98a3e', 'eval': '#8a72b5', 'aws': '#9a9a9a'}


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
        self.text(24, 34, title, 20, INK, weight=700)

    def text(self, x, y, s, size=13, colour=INK, anchor='start', weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
                          f'fill="{colour}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')

    def frame(self, x, y, w, h, title, kind):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{FILL[kind]}" stroke="{EDGE[kind]}" stroke-width="1.5"/>')
        self.text(x + 12, y + 22, title, 14, EDGE[kind], weight=700)

    def box(self, x, y, w, h, title, lines=(), kind='sim'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{EDGE[kind]}" stroke-width="1.2"/>')
        self.text(x + 10, y + 19, title, 13, INK, weight=700)
        for i, line in enumerate(lines):
            self.text(x + 10, y + 37 + 16 * i, line, 12, MUTED)

    def arrow(self, pts, label=None, at=0.5, dx=6, dy=-6, colour=LINE, dashed=False, both=False):
        d = 'M ' + ' L '.join(f'{x} {y}' for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ''
        start = ' marker-start="url(#back)"' if both else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6"{dash} marker-end="url(#head)"{start}/>')
        if label:
            i = min(int(at * (len(pts) - 1)), len(pts) - 2)
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            self.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, 12, MUTED)

    def write(self, path):
        defs = ('<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
                '<marker id="back" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M10,0 L0,5 L10,10 z" fill="{LINE}"/></marker></defs>')
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                        + defs + ''.join(self.parts) + '</svg>\n')


def architecture():
    s = Svg(1600, 900, 'Smoke Rescue: system architecture')
    s.text(24, 56, 'Robots, agents and the commander get no map, outline, entry pose or sectors. Only the evaluator reads ground truth.', 13, MUTED)
    X1, X2, X3, W1, W2, W3 = 24, 564, 1144, 400, 440, 432
    s.frame(X1, 80, W1, 560, 'Unity 6 HDRP simulation', 'sim')
    s.box(X1 + 16, 112, W1 - 32, 118, 'Sensors, per robot, seeded', ['RGB-D 640x480 at 15 Hz, depth noise model', 'Thermal 320x256 (LWIR, saturates 650 C)', 'Planar LiDAR 720 beams 10 Hz; smoke drops pulses', 'IMU 200 Hz, leg odometry, thermocouple 10 Hz'], 'sim')
    s.box(X1 + 16, 244, W1 - 32, 86, 'Burning house (config/floorplan.json)', ['FireSim: spread on 0.25 m cells, heat, gas layer', 'Fluid smoke (stable fluids, two layers)', 'Water absorbs heat; wet fuel resists reignition'], 'sim')
    s.box(X1 + 16, 344, W1 - 32, 112, 'Recorder (RecordingSession)', ['Steps world, sensors, brains and agents on sim time', 'Writes recordings, brain maps, decisions and agent', 'traces; truth goes only to oracle/', 'Checkpoints every 30 sim s'], 'sim')
    s.box(X1 + 16, 470, W1 - 32, 56, 'RadioModel', ['Lossy, range-limited radio: robots and the post'], 'sim')
    s.box(X1 + 16, 540, W1 - 32, 86, 'Robot body (RobotActions)', ['Quadruped legs, 3-joint arm, torch, 12 L nozzle', 'open_door, clear_debris, spray_water,', 'extract_survivor: sled towed to a way out'], 'sim')
    s.frame(X2, 80, W2, 560, 'Robot brain (C#, 60 Hz), one per robot', 'brain')
    s.box(X2 + 16, 112, W2 - 32, 86, 'Mapping and localisation', ['Occupancy grid from LiDAR and depth, LiDAR open ground', 'Gyro heading plus 5 % of each scan-match turn', 'Live 3D top heights from the agent service TSDF'], 'brain')
    s.box(X2 + 16, 212, W2 - 32, 70, 'Indoor rule (own evidence only)', ['Ceiling seen overhead, ceiling sight lines,', 'or walled in on its own map in most directions'], 'brain')
    s.box(X2 + 16, 296, W2 - 32, 70, 'Search', ['Floor counts as searched only when seen within 4 m;', 'frontier claims shared; A* with wall and fire clearance'], 'brain')
    s.box(X2 + 16, 380, W2 - 32, 102, 'Heat and safety reflexes', ['Keep a way out: fire near the route home, leave', 'Heat building (30 C, or 8.3 C/min): no agent override;', 'cut off: emergency exit past the fire, never through it', 'Retreat, heat budget, re-entry when cool'], 'brain')
    s.box(X2 + 16, 496, W2 - 32, 128, 'Distributed context', ['Radio v4: map layers, claims, confirmations,', 'carried-out people, LiDAR keyframes', 'FrameLink: fingerprint shortlist, scan-match check,', 'two agreeing matches 1 m apart before frames link;', 'partner data applies only after a link'], 'brain')
    s.frame(X3, 80, W3, 560, 'Agent layer (Python, FastAPI :8080)', 'agent')
    s.box(X3 + 16, 112, W3 - 32, 102, 'Robot agent x2 (GPT, tool use)', ['Woken by brain events or every 20 s. Context: its', 'own-frame pose, offered frontiers, LiDAR summary,', 'radio inbox. Chooses where to search, decides if a', 'warm shape is a person, orders actions'], 'agent')
    s.box(X3 + 16, 228, W3 - 32, 102, 'OpenCV 5 perception tools (own frames)', ['scan_thermal, inspect_candidate, check_indoors,', 'YOLO11m person detector via cv2.dnn,', 'assess_visibility, track_fire_spread, measure_opening;', 'each result gets an evidence id'], 'agent')
    s.box(X3 + 16, 344, W3 - 32, 86, 'Guardrails in code', ['Actions must cite a matching OpenCV result;', 'confirm only within 4 m, never a person this robot', 'already carried out; invented frontier ids refused'], 'agent')
    s.box(X3 + 16, 444, W3 - 32, 70, 'Incident commander', ['Hears radio reports, relays them, approves risky', 'actions; no map, area or coordinates'], 'agent')
    s.box(X3 + 16, 528, W3 - 32, 96, 'Services', ['Live 3D map: OpenCV 5 HashTSDF per robot (/map3d)', 'Place recognition: ORB + 3D RANSAC (/place)', 'Trace: perception id -> tool call -> action'], 'agent')
    g1, g2 = X1 + W1, X2 + W2
    s.arrow([(g1, 150), (X2, 150)]); s.text(g1 + 70, 142, 'own sensors only', 12, MUTED, anchor='middle')
    s.arrow([(X2, 598), (g1, 598)]); s.text(g1 + 70, 590, 'speed, turn, torch', 12, MUTED, anchor='middle')
    s.arrow([(g1, 505), (X2, 505)], both=True); s.text(g1 + 70, 497, 'radio packets', 12, MUTED, anchor='middle')
    s.arrow([(g2, 140), (X3, 140)]); s.text(g2 + 70, 132, 'events + context', 12, MUTED, anchor='middle')
    s.arrow([(X3, 188), (g2, 188)]); s.text(g2 + 70, 180, 'orders', 12, MUTED, anchor='middle')
    s.text(g2 + 70, 206, 'refusals go back', 12, MUTED, anchor='middle')
    s.arrow([(X3 + 120, 640), (X3 + 120, 662), (X1 + 300, 662), (X1 + 300, 626)], dashed=True)
    s.text(X2 + 220, 656, 'agent actions: door, debris, water, lift (RobotActions, after the guardrails)', 12, MUTED, anchor='middle')
    s.arrow([(X1 + 80, 640), (X1 + 80, 700)], dashed=True); s.text(X1 + 90, 684, 'recordings + oracle/', 12, MUTED)
    s.frame(X1, 700, 1100, 180, 'Evaluation (reads hidden truth; robots never do)', 'eval')
    s.box(X1 + 16, 732, 520, 132, 'oracle/ in every recording', ['True poses, survivors, fire events, collisions, house', 'point cloud, heat exposure per robot. Written by the', 'SmokeRescue.Oracle assembly; assembly definitions and', 'a pytest keep robot code away from it'], 'eval')
    s.box(X1 + 556, 732, 528, 132, 'Scoring and replay', ['nav_eval: pose error, map precision and recall, indoor', 'layer outside the building, stalls, link error;', 'compare_runs: people found, false, carried out,', 'searched floor, heat exposure, agent orders;', 'showcase.py: video from the robots\u2019 own views'], 'eval')
    s.frame(X3, 700, W3, 180, 'AWS (owned by Vishnu)', 'aws')
    s.box(X3 + 16, 732, W3 - 32, 132, 'deploy/aws', ['CloudFormation: agent + OpenCV API on Fargate ARM64,', 'S3 + CloudFront results site, Batch job for the', 'OpenCV 5 SLAM container, budget alarms. No Open3D', 'left in the pipeline, so it runs on ARM64'], 'aws')
    s.write(OUT / 'architecture.svg')


def workflow():
    s = Svg(1400, 760, 'Smoke Rescue: one agent step (OpenCV results change the next action)')
    s.text(24, 56, 'Every arrow is logged in agent_trace.jsonl and the robot’s decisions.jsonl. The model only sees this robot’s own data.', 13, MUTED)
    y = 100
    steps = [
        ('1  Brain event', ['survivor_candidate, no_frontier, area_searched,', 'escape_closing, agent_goal_reached, or 20 s'], 'brain'),
        ('2  Context', ['Recorder flushes frames and logs; sends own pose,', 'offered frontiers, LiDAR sectors, radio inbox'], 'sim'),
        ('3  Perceive (OpenCV 5)', ['inspect_candidate, scan_thermal, YOLO11m,', 'check_indoors on the robot’s own frame'], 'agent'),
        ('4  Decide (GPT)', ['Reads tool results with their evidence ids,', 'chooses an action that cites one'], 'agent'),
        ('5  Guardrail (Python)', ['Evidence type and position match, confirm', 'within 4 m, not a person already carried out'], 'agent'),
        ('6  Brain check (C#)', ['Indoors by its own evidence? Heat building', 'with no frontier left? Already leaving?'], 'brain'),
        ('7  Act', ['go_to / explore_frontier / confirm / reject,', 'torch, keep-out area, door, water, lift'], 'sim'),
        ('8  Talk', ['send_team_message over the lossy radio;', 'commander relays, approves risky actions'], 'agent'),
    ]
    xs = [24, 368, 712, 1056]
    for i, (title, lines, kind) in enumerate(steps):
        x, yy = xs[i % 4], y + (i // 4) * 170
        s.box(x, yy, 320, 96, title, lines, kind)
    for i in range(3):
        s.arrow([(xs[i] + 320, y + 48), (xs[i + 1], y + 48)])
        s.arrow([(xs[i] + 320, y + 218), (xs[i + 1], y + 218)])
    s.arrow([(xs[3] + 160, y + 96), (xs[3] + 160, y + 130), (xs[0] + 160, y + 130), (xs[0] + 160, y + 170)])
    s.arrow([(xs[1] + 160, y + 266), (xs[1] + 160, y + 300)], dashed=True)
    s.text(xs[1] + 170, y + 290, 'refused: the reason goes back to the model in the same step', 12, MUTED)
    s.arrow([(xs[2] + 160, y + 266), (xs[2] + 160, y + 300)], dashed=True)
    s.text(xs[2] + 170, y + 318, 'refused by the brain: logged, the agent is told at its next step', 12, MUTED)
    s.frame(24, 470, 1352, 260, 'Observed in the held-out run (seed 914, robot B); evidence ids from agent_trace.jsonl', 'eval')
    ex = [
        '0.0 s   inspect_candidate (4.8 m) -> uncertain [b63fd941] -> approach_candidate citing it -> brain go_to, facing the shape.',
        '4.0 s   brain: survivor_candidate (thermal blob) -> inspect_candidate (3.6 m) -> person [2d4da82a] -> confirm_survivor citing it: accepted',
        '          (truth: person s1, 0.14 m from the confirmed position).',
        '13.2 s  extract_survivor citing [2d4da82a] -> arm grips, sled -> towed out; set down past the front door at 47.0 s; re-entered at 48.4 s.',
        '18.8 s  brain: avoid_fire -> track_fire_spread [c4627820] -> avoid_area citing it + team message: fire growing ahead of its route out.',
        '43.7 s  explore_frontier while still towing -> refused by the brain: "the robot is leaving (Carrying a person out.)".',
        'Every action cites the OpenCV result it depends on; the same calls without a matching result are refused in code (tests).',
    ]
    for i, line in enumerate(ex):
        s.text(40, 506 + 26 * i, line, 13, INK if not line.startswith('          ') and not line.startswith('Every') else MUTED)
    s.write(OUT / 'agent_workflow.svg')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    architecture(); workflow()
    print('wrote', *sorted(p.name for p in OUT.glob('*.svg')))
