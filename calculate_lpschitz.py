import numpy as np
import math

# calculate the lipschitz constant according to the covariance matrix

# Fix some value here
R = 1000
# we choose beta = 0
gamma = 0.1


def get_lipschitz(cov_matrix):
    # 计算奇异值
    singular_values = np.linalg.svd(cov_matrix, compute_uv=False)
    min_singular_value = np.min(singular_values)

    # 计算行列式
    determinant = np.linalg.det(cov_matrix)

    # print("最小的奇异值:", min_singular_value)
    # print("行列式:", determinant)

    # 这里我们是在二维平面上做的实验，所以 dimension = 2
    dimension = 2


    # calculate lipschitz
    lipschitz = 1 / min_singular_value + (2 * R * R / (gamma * gamma * min_singular_value * min_singular_value)) * (1 / ((2 * math.pi) ** dimension * determinant) + 1 / ((2 * math.pi) ** (dimension / 2) * math.sqrt(determinant)))

    return lipschitz


if __name__ == "__main__":

    eye_value = 10

    # 假设你有一个正方形协方差矩阵 cov_matrix
    # 例如：
    for eye_value in [1, 10, 100, 200, 500, 600, 700, 800, 900, 1000]:

        cov_matrix = np.array([[eye_value, 0], [0, eye_value]])

        lipschitz = get_lipschitz(cov_matrix)

        print(cov_matrix)

        print("lip", lipschitz)





