using UnityEngine;

namespace SmokeRescue.Robots
{
    /// <summary>
    /// Manipulator on the robot's back: base yaw, shoulder and elbow pitch, and a rigid gripper. Analytic two-link IK
    /// in the vertical plane after the base yaw, elbow up. Purely kinematic and visual: no colliders, Robots layer, so it
    /// never trips the robot's own collision check. Stowed, it folds back along the body behind both cameras.
    ///   ArmRoot (on Body)  BaseYaw  Shoulder  UpperArm  Elbow  Forearm  Gripper  Tip
    /// </summary>
    public sealed class RobotArm
    {
        public const float ShoulderHeight = 0.08f, UpperLength = 0.45f, ForearmLength = 0.40f, GripperLength = 0.10f;
        const float Lower = ForearmLength + GripperLength;     // the gripper is rigid, so forearm + gripper is one link
        const float StowElevationDeg = 180f, StowFoldDeg = -170f;

        public Transform Root { get; private set; }
        public bool IsStowed { get; private set; }
        public Vector3 GripperPosition => tip.position;
        public float BaseYawDeg { get; private set; }
        public float ShoulderElevationDeg { get; private set; }
        public float ElbowDeg { get; private set; }

        Transform baseYaw, shoulder, elbow, tip;

        public static RobotArm Build(Transform body, Vector3 localRoot, Color band)
        {
            var arm = new RobotArm();
            arm.Root = Node("ArmRoot", body, localRoot);
            arm.baseYaw = Node("BaseYaw", arm.Root, Vector3.zero);
            arm.shoulder = Node("Shoulder", arm.baseYaw, new Vector3(0f, ShoulderHeight, 0f));
            arm.elbow = Node("Elbow", arm.shoulder, new Vector3(0f, 0f, UpperLength));
            var gripper = Node("Gripper", arm.elbow, new Vector3(0f, 0f, ForearmLength));
            arm.tip = Node("Tip", gripper, new Vector3(0f, 0f, GripperLength));

            Color metal = new Color(0.18f, 0.18f, 0.2f), joint = new Color(0.08f, 0.08f, 0.09f);
            Visual(arm.Root, PrimitiveType.Cylinder, "Turret", new Vector3(0f, 0.03f, 0f), new Vector3(0.12f, 0.03f, 0.12f), joint);
            Visual(arm.baseYaw, PrimitiveType.Cube, "ShoulderBlock", new Vector3(0f, ShoulderHeight * 0.6f, 0f), new Vector3(0.09f, 0.08f, 0.09f), metal);
            // Cylinders are 2 units long along y; rotate them to run along the link's +z.
            Visual(arm.shoulder, PrimitiveType.Cylinder, "UpperArm", new Vector3(0f, 0f, UpperLength / 2f), new Vector3(0.055f, UpperLength / 2f, 0.055f), metal, true);
            Visual(arm.shoulder, PrimitiveType.Cube, "UpperBand", new Vector3(0f, 0f, UpperLength * 0.55f), new Vector3(0.062f, 0.062f, 0.06f), band);
            Visual(arm.elbow, PrimitiveType.Sphere, "ElbowJoint", Vector3.zero, Vector3.one * 0.075f, joint);
            Visual(arm.elbow, PrimitiveType.Cylinder, "Forearm", new Vector3(0f, 0f, ForearmLength / 2f), new Vector3(0.045f, ForearmLength / 2f, 0.045f), metal, true);
            Visual(gripper, PrimitiveType.Cube, "Wrist", new Vector3(0f, 0f, 0.02f), new Vector3(0.07f, 0.05f, 0.05f), joint);
            Visual(gripper, PrimitiveType.Cube, "FingerL", new Vector3(-0.025f, 0f, GripperLength * 0.6f), new Vector3(0.012f, 0.03f, GripperLength * 0.8f), joint);
            Visual(gripper, PrimitiveType.Cube, "FingerR", new Vector3(0.025f, 0f, GripperLength * 0.6f), new Vector3(0.012f, 0.03f, GripperLength * 0.8f), joint);
            arm.Stow();
            return arm;
        }

        public void Stow()
        {
            SetJoints(0f, StowElevationDeg, StowFoldDeg);
            IsStowed = true;
        }

        /// <summary>Gripper tip position the stow pose gives, for planning a smooth return before Stow() snaps the joints.</summary>
        public Vector3 StowTipPosition
        {
            get
            {
                float e1 = StowElevationDeg * Mathf.Deg2Rad, e2 = (StowElevationDeg + StowFoldDeg) * Mathf.Deg2Rad;
                var local = new Vector3(0f, ShoulderHeight + UpperLength * Mathf.Sin(e1) + Lower * Mathf.Sin(e2),
                                        UpperLength * Mathf.Cos(e1) + Lower * Mathf.Cos(e2));
                return Root.TransformPoint(local);
            }
        }

        /// <summary>Move the gripper tip to a world point. Returns false, leaving the arm as it was, if out of reach.</summary>
        public bool Reach(Vector3 world)
        {
            var p = Root.InverseTransformPoint(world);
            float yaw = new Vector2(p.x, p.z).sqrMagnitude > 1e-8f ? Mathf.Atan2(p.x, p.z) * Mathf.Rad2Deg : BaseYawDeg;
            float r = Mathf.Sqrt(p.x * p.x + p.z * p.z), h = p.y - ShoulderHeight;
            float d = Mathf.Sqrt(r * r + h * h);
            if (d > UpperLength + Lower - 1e-4f || d < Mathf.Abs(UpperLength - Lower) + 1e-4f) return false;
            float cosBend = Mathf.Clamp((d * d - UpperLength * UpperLength - Lower * Lower) / (2f * UpperLength * Lower), -1f, 1f);
            float bend = Mathf.Acos(cosBend);                                  // 0 = straight
            float e1 = Mathf.Atan2(h, r) + Mathf.Atan2(Lower * Mathf.Sin(bend), UpperLength + Lower * Mathf.Cos(bend));
            SetJoints(yaw, e1 * Mathf.Rad2Deg, -bend * Mathf.Rad2Deg);
            IsStowed = false;
            return true;
        }

        void SetJoints(float yawDeg, float shoulderElevationDeg, float elbowDeg)
        {
            BaseYawDeg = yawDeg; ShoulderElevationDeg = shoulderElevationDeg; ElbowDeg = elbowDeg;
            baseYaw.localRotation = Quaternion.Euler(0f, yawDeg, 0f);
            // Positive elevation lifts the link: Unity's +x rotation turns +z downward, hence the minus sign.
            shoulder.localRotation = Quaternion.Euler(-shoulderElevationDeg, 0f, 0f);
            elbow.localRotation = Quaternion.Euler(-elbowDeg, 0f, 0f);
        }

        static Transform Node(string name, Transform parent, Vector3 local)
        {
            var t = new GameObject(name).transform;
            t.SetParent(parent, false);
            t.localPosition = local;
            t.gameObject.layer = LayerMask.NameToLayer("Robots");
            return t;
        }

        static void Visual(Transform parent, PrimitiveType type, string name, Vector3 pos, Vector3 scale, Color c, bool alongZ = false)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            Object.DestroyImmediate(go.GetComponent<Collider>());
            go.transform.SetParent(parent, false);
            go.transform.localPosition = pos;
            go.transform.localRotation = alongZ ? Quaternion.Euler(90f, 0f, 0f) : Quaternion.identity;
            go.transform.localScale = scale;
            go.layer = LayerMask.NameToLayer("Robots");
            var shader = Shader.Find("HDRP/Lit");
            var mat = new Material(shader != null ? shader : Shader.Find("Standard"));
            mat.SetColor("_BaseColor", c);
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
        }
    }
}
