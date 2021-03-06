import pupil_apriltags as apriltag
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import codecs, json
import math
from updater import add_object


def draw_cloud(img, points, Rt):
    imgP = []
    for p in points:
        P = Rt.dot(np.hstack((p, 1.0)))
        pixel = (P / P[-1])[:-1]
        imgP.append(pixel)
    for p in imgP:
        cv2.circle(img, (int(p[0]), int(p[1])), 1, (255, 0, 0), -1)
    """for i in range(len(imgP) - 1):
        pt1 = ( int(imgP[i][0]), int(imgP[i][1]) )
        pt2 = ( int(imgP[i+1][0]), int(imgP[i+1][1] ))
        cv2.line(img,pt1,pt2,(0,0,255),1)"""


def read_points(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        points = data["points"]
        pd3 = []
        for p in points:
            pd3.append([-p[0] / 1000, p[1] / 1000, p[2] / 1000])
    return pd3


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def deproject_pixel_to_point(dept_intrins, pixels, depth):

    points = np.zeros(3)
    fx = float(dept_intrins.fx)  # Focal length of x
    fy = float(dept_intrins.fy)  # Focal length of y
    ppx = float(dept_intrins.ppx)  # Principle Point Offsey of x (aka. cx)
    ppy = float(dept_intrins.ppy)
    x = (pixels[0] - ppx) / fx
    y = (pixels[1] - ppy) / fy

    points[0] = depth * x
    points[1] = depth * y
    points[2] = depth
    return points


def f32toHUE(image, min_depth, max_depth, width, height):
    hue_img = np.zeros((height, width, 3), dtype=np.float32)

    R, G, B = 0, 0, 0
    for i in range(width - 1):
        for j in range(height - 1):
            d = image[j][i]
            if min_depth <= d and d <= max_depth:
                dn = 1529 * (d - min_depth) / (max_depth - min_depth)

                if (
                    dn <= 255 or 1275 < dn and dn <= 1529
                ):  # 0 < dn <= 60 or 300 < dn < 360
                    R = 255
                elif dn <= 510:  # 60 < dn <= 120
                    R = 510 - dn
                elif dn <= 1020:  # 120 < dn <= 240
                    R = 0
                elif dn <= 1275:  # // 240 < dn <= 300
                    R = dn - 1020

                if dn <= 255:  # // 0 < dn <= 60
                    G = dn
                elif dn <= 765:  # // 60 < dn <= 180
                    G = 255
                elif dn <= 1020:  # // 180 < dn <= 240
                    G = 765 - dn
                elif dn <= 1529:  # // 180 < dn <= 360
                    G = 0

                if dn <= 510:  # // 0 < dn <= 120
                    B = 0
                elif dn <= 765:  # // 120 < dn <= 180
                    B = dn - 510
                elif dn <= 1275:  # // 180 < dn <= 300
                    B = 255
                elif dn <= 1529:  # // 300 < dn <= 360
                    B = 1529 - dn
                hue_img[j][i] = np.array([B, G, R])

    return hue_img


def get_data(json_path):

    with open(json_path) as json_file:
        data = json.load(json_file)
        rules = {}

        class_ids = []
        Rt_matrixes = []
        for k, v in data.items():
            rule_list = []
            class_ids.append(k)
            for rule, mat in v.items():
                rule_list.append(rule)
                Rt_matrixes.append(mat)
            rules[k] = rule_list
    return class_ids, rules, data


pipeline = (
    rs.pipeline()
)  # <- ???????????? pipeline ???????????????? ???????????? ?????? ???????????????????????????? ?? ??????????????
config = rs.config()  # <- ???????????????????????????? ???????????? ?????? ???????????????? ???????????????? ????????????
colorizer = rs.colorizer()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = pipeline.start(config)
detector = apriltag.Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)
font = cv2.FONT_HERSHEY_COMPLEX
print("START")
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_intrin = (
    profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
)
fx = float(intr.fx)  # Focal length of x
fy = float(intr.fy)  # Focal length of y
ppx = float(intr.ppx)  # Principle Point Offsey of x (aka. cx)
ppy = float(intr.ppy)  # Principle Point Offsey of y (aka. cy)
camera_params = [fx, fy, ppx, ppy]
IsCheck = False
class_names = [
    "?????????? - ??????????",
    "?????????? - ???????????? ????????????????????",
    "???????????? ??????????????_CPY",
    "????????_????????????_CPY",
    "?????? ????????????",
]
class_idx = 0
rules_idx = 0
IsTag = True

m_int = np.array([fx, 0, ppx, 0, fy, ppy, 0, 0, 1])


work_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(work_dir, "dataset")

js_int = {"camera": [{"intrinsics": m_int.tolist()}]}
json_name = "camera_params" ".json"
json_path = os.path.join(dataset_dir, json_name)
json.dump(
    js_int,
    codecs.open(json_path, "w", encoding="utf-8"),
    separators=(",", ":"),
    indent=4,
)

gubka = os.path.join(work_dir, "gubka.json")
privod = os.path.join(work_dir, "CPY.json")
g_points = read_points(gubka)
privod_points = read_points(privod)
detal2draw = [g_points, privod_points]
detal2draw_idx = 0

if os.path.exists(dataset_dir):
    pass
else:
    os.mkdir(dataset_dir)
name = 0
json_path = "/Users/andreymazko/Desktop/cam/vse/build/wrappers/python/dataset/rt_to_ojbects.json"
class_ids, rules, data = get_data(json_path)
class_id = class_ids[class_idx]
rule = rules[class_id][rules_idx]
Rt_to_detal = np.array(data[class_id[0]][rule])

print(f"???????????? ?????? {class_names[class_idx]}")
print(f"?????????????? ???????????????? {rules[class_id][rules_idx]}")


def format_name(name):
    name = str(name)
    zero_count = 4 - len(name)
    new_name = ""
    for i in range(zero_count):
        new_name += "0"
    new_name += name
    return new_name


try:
    while True:
        key = cv2.waitKey(1)
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        # Apriltag detection
        gray_rs_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        result = detector.detect(
            gray_rs_frame,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=0.06,
        )
        if key == ord("e"):
            detal2draw_idx += 1
            if detal2draw_idx >= len(detal2draw):
                detal2draw_idx = 0

        if key == ord("i"):
            print()
            print(f"?????????????????? ?????????????? ???? id{class_ids}")
            print(f"?? ???????????? ???????????? ???????????? object_id {class_id}")
            print(f"?????????????? ???????????????? {rules[class_id][rules_idx]}")

        if key == ord("c"):
            print()
            cids_len = len(class_ids)
            class_idx += 1
            if class_idx >= cids_len:
                class_idx = 0
            rules_idx = 0
            class_id = class_ids[class_idx]
            rule = rules[class_id][rules_idx]
            print(f"???????????? ?????? object_id {class_id}")
            print(f"?????????????? ???????????????? {rules[class_id][rules_idx]}")
            Rt_to_detal = np.array(data[class_id[0]][rule])
            detal2draw_idx += 1
            if detal2draw_idx >= len(detal2draw):
                detal2draw_idx = 0
        if key == ord("a"):
            try:
                add_object(json_path)
                class_ids, rules, data = get_data(json_path)
            except Exception as e:
                print(e)
        if key == ord("r"):
            print()
            rule_len = len(rules[class_id])
            rules_idx += 1
            if rules_idx >= rule_len:
                rules_idx = 0
            rule = rules[class_id][rules_idx]
            print(f"???????????? ?????? object_id {class_id}")
            print(f"?????????????? ???????????????? {rule}")

            Rt_to_detal = np.array(data[class_id[0]][rule])

        if key == ord("o"):
            if IsTag:
                for detect in result:
                    m_int = np.array([fx, 0, ppx, 0, fy, ppy, 0, 0, 1]).reshape(3, 3)
                    # rt for cam2tag
                    R_to_tag = detect.pose_R
                    t_to_tag = detect.pose_t
                    Rt_to_tag = np.hstack((R_to_tag, t_to_tag.reshape(3, 1)))
                    m_int_Rt = m_int.dot(Rt_to_tag)
                    resul_mat = Rt_to_tag.dot(Rt_to_detal)
                    # matrix for tag to pixel(detal center)
                    tag_to_detal = m_int_Rt.dot(Rt_to_detal)
                    Tmat = tag_to_detal
                    Point3D = np.array([0, 0, 0])
                    points = Tmat.dot(np.hstack((Point3D, 1.0)))

                    # pixel detal center
                    pixel = (points / points[-1])[:-1]

                    pixel = pixel.astype(int)

                    pose_t = resul_mat[:, 3:]
                    tvec_axis = pose_t
                    pose_R = resul_mat[:, :3]
                    pose_t = [t[0] * 1000 for t in pose_t]
                    R = []
                    R = cv2.Rodrigues(pose_R)[0]
                    axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(
                        -1, 3
                    )
                    imgpts, jac = cv2.projectPoints(axis, R, tvec_axis, m_int, None)
                    R = [r[0] * 180 / math.pi for r in R]
                    color_name = format_name(name) + "TAG.png"
                    color_path = os.path.join(dataset_dir, color_name)
                    cv2.imwrite(color_path, color_image)

                    js = {
                        "object": [
                            {
                                "class_id": class_id,
                                "rotation": R,
                                "translation": pose_t,
                                "rule": rule,
                            }
                        ]
                    }
                    json_name = format_name(name) + ".json"
                    json_path = os.path.join(dataset_dir, json_name)
                    json.dump(
                        js,
                        codecs.open(json_path, "w", encoding="utf-8"),
                        separators=(",", ":"),
                        indent=4,
                    )

                    IsTag = not IsTag
            else:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = f32toHUE(
                    depth_image, 500, 1500, depth_image.shape[1], depth_image.shape[0]
                )
                depth_name = format_name(name) + "_depth.png"
                depth_path = os.path.join(dataset_dir, depth_name)
                cv2.imwrite(depth_path, depth_colormap)

                color_name = format_name(name) + ".png"
                color_path = os.path.join(dataset_dir, color_name)
                cv2.imwrite(color_path, color_image)
                name += 1
                IsTag = not IsTag
        if key == ord("s"):
            IsCheck = not IsCheck
        if key == ord("t"):  # ???????????????????????? ?????? ??????????????
            detection = result
            for detect in detection:

                m_int = np.array([fx, 0, ppx, 0, fy, ppy, 0, 0, 1]).reshape(3, 3)

                dist_coeffs = np.zeros((5, 1))
                Point3D = deproject_pixel_to_point(
                    depth_intrin, [detect.center[0], detect.center[1]], detect.pose_t[2]
                )

                R_to_tag = detect.pose_R
                t_to_tag = detect.pose_t
                Rt_to_tag = np.hstack((R_to_tag, t_to_tag.reshape(3, 1)))
                m_int_Rt = m_int.dot(Rt_to_tag)
                tag_to_detal = m_int_Rt.dot(Rt_to_detal)
                m_int = np.array([fx, 0, ppx, 0, fy, ppy, 0, 0, 1]).reshape(3, 3)
                resul_mat = Rt_to_tag.dot(Rt_to_detal)
                Tmat = tag_to_detal
                Point3D = np.array([0, 0, 0])

                points = Tmat.dot(np.hstack((Point3D, 1.0)))

                pixel = (points / points[-1])[:-1]
                pixel = pixel.astype(int)
                pose_t = resul_mat[:, 3:]
                pose_R = resul_mat[:, :3]

                draw_cloud(color_image, detal2draw[detal2draw_idx], Tmat)
                R = []
                R = cv2.Rodrigues(pose_R)[0]
                axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(
                    -1, 3
                )
                imgpts, jac = cv2.projectPoints(axis, R, pose_t, m_int, None)

        if key == ord("w"):
            try:
                draw_cloud(color_image, detal2draw[detal2draw_idx], Tmat)
            except:
                passs
            # cv2.circle(color_image, tuple(pixel), 3, (255, 255, 0), 8)
            # DRAW
            """cv2.line(
                color_image, tuple(pixel), tuple(imgpts[0].ravel()), (255, 0, 0), 5
            )
            cv2.line(
                color_image, tuple(pixel), tuple(imgpts[1].ravel()), (0, 255, 0), 5
            )
            cv2.line(
                color_image, tuple(pixel), tuple(imgpts[2].ravel()), (0, 0, 255), 5
            )"""
        # Show images
        if IsCheck:
            cv2.rectangle(color_image, (0, 0), (650, 45), (0, 255, 100), -1)
            cv2.rectangle(color_image, (1100, 645), (1280, 720), (0, 255, 100), -1)
            cv2.putText(
                color_image,
                f"Class id: {class_id}",
                (1110, 665),
                font,
                0.7,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                color_image, f"rule {rule}", (1110, 700), font, 0.7, (255, 0, 0), 1
            )
            if IsTag:

                if len(result) == 0:

                    cv2.putText(
                        color_image,
                        "?????? ???? ????????????",
                        (15, 20),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )
                else:
                    cv2.putText(
                        color_image, "?????? ????????????", (15, 20), font, 0.7, (255, 0, 0), 1
                    )
                    cv2.putText(
                        color_image,
                        "?????????????? 'O' ?????????? ???????????????????? ?????????????? ",
                        (15, 40),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )
            else:
                if len(result) != 0:
                    cv2.putText(
                        color_image,
                        f"???????????????? ???????????? ???? ??????",
                        (15, 20),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )
                    cv2.putText(
                        color_image,
                        f"???????????????? ?????????????? ???????????????????? {rule}",
                        (20, 40),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )
                else:
                    cv2.putText(
                        color_image,
                        f"??????????????????, ?????? ???????????? ???????????????????? ??????????????????",
                        (15, 20),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )
                    cv2.putText(
                        color_image,
                        f"?????????????? 'O' ?????????? ?????????????? ????????",
                        (20, 40),
                        font,
                        0.7,
                        (255, 0, 0),
                        1,
                    )

        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", color_image)

        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
finally:

    # Stop streaming
    pipeline.stop()
