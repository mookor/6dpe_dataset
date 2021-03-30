import numpy as np
import os
import codecs, json
import math


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


def add_object(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)

    class_id = str(input("Введите Id нового объекта - "))

    assert class_id not in list(data.keys()), f"Объект class_id:{class_id} уже добавлен"

    rule_count = int(input("Введите количество возможных положений объекта - "))
    rules = []
    rt_detal_list = []
    rules_dict = {}
    for i in range(rule_count):
        rule = str(input(f"#{i+1} Введите правило связи тэг - точка: "))
        rules.append(rule)
        print()
        print("Введите смещение объекта по разным осям для текущего правила (в мм)")
        x_transpose = float(input("по оси X - "))
        y_transpose = float(input("по оси Y - "))
        z_transpose = float(input("по оси Z  - "))
        print()
        print(
            "Введите вращение объекта вокруг разных осей для текущего правила (в градусах)"
        )
        x_rotate = float(input("Вокруг оси X - "))
        y_rotate = float(input("Вокруг оси Y - "))
        z_rotate = float(input("Вокруг оси Z  - "))

        rvec_to_detal = [y_rotate, x_rotate, z_rotate]
        rvec_to_detal = np.array([r * math.pi / 180 for r in rvec_to_detal])
        R_to_detal = eulerAnglesToRotationMatrix(rvec_to_detal)
        # R_to_detal = np.array([0,-1,0,0,0,1,-1,0,0]).reshape(3,3)
        tvec_to_detal = np.array([y_transpose, x_transpose, -z_transpose])
        Rt_to_detal = np.hstack((R_to_detal, tvec_to_detal.reshape(3, 1)))
        Rt_to_detal = np.vstack((Rt_to_detal, [0, 0, 0, 1]))
        rules_dict[rule] = Rt_to_detal

    data[class_id] = rules_dict

    json.dump(
        data,
        codecs.open(json_path, "w", encoding="utf-8"),
        separators=(",", ":"),
        indent=4,
        cls=NumpyArrayEncoder,
    )
