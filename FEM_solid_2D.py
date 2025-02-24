import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from meshpy import triangle
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon, Point

label = 'test'
letter_size = 12
plt.rc('font', family='Times New Roman', size=letter_size)


class LinearProblem2D:
    """
    划分网格、生成单元和边界条件，生成、组装线性刚度矩阵和荷载向量，显示求解位移、应力、应变，绘制单元变形图、位移云图、应力云图、应变云图，保存计算结果与图片
    """

    def __init__(self,
                 Boundary_Nodes,
                 Young_modulus,
                 Poisson_ratio,
                 max_area=20.0,
                 penalty=1e+4):
        """
            输入任意长度的边界节点数组，在它们围成的区域内均匀划分三角形单元。

                参数:
                - boundary_nodes: 二维数组，定义区域边界上的节点，用于确定求解区域
                - max_area: float，控制生成网格的最大单元面积，越小则点越密集

                返回:
                - nodes: 二维数组，生成的节点
                - elements: 二维数组，生成的三角形单元
        """
        self.Y_nodes = None
        self.X_nodes = None
        self.num_elements = None
        self.num_nodes = None
        self.elements = None
        self.nodes = None
        self.dist_mat = None
        self.ess_bc = {}
        self.nat_bc = {}
        self.constants = None
        self.boundary_nodes = Boundary_Nodes
        self.Young_modulus = Young_modulus  # 杨氏模量
        self.Poisson_ratio = Poisson_ratio  # 泊松比
        self.max_area = max_area
        self.penalty = penalty

    def set_elements(self, ):
        print('------------------------生成单元------------------------')
        start = time.time()
        # 定义边界边
        boundary_facets = [[i, (i + 1) % len(self.boundary_nodes)] for i in range(len(self.boundary_nodes))]

        info = triangle.MeshInfo()
        info.set_points(self.boundary_nodes)
        info.set_facets(boundary_facets)

        # 定义细化函数
        def needs_refinement(vertices, area):
            return area > self.max_area

        # 生成网格
        mesh = triangle.build(info, refinement_func=needs_refinement)

        end = time.time()
        t = end - start
        print(f'生成单元耗时: {t:.4e}s')

        self.nodes = np.array(mesh.points)  # 节点坐标，形为[[x_0, y_0], [x_1, y_1], ..., [x_(n-1), y_(n-1)]]
        self.elements = np.array(mesh.elements)  # 单元节点索引，形为[...[x_i, x_j, x_m]...]
        self.num_nodes = len(self.nodes)  # 节点数量
        self.num_elements = len(self.elements)  # 单元数量
        self.X_nodes = self.nodes[:, 0]
        self.Y_nodes = self.nodes[:, 1]

        len_nodes = len(self.X_nodes)
        dist_mat = lil_matrix((len_nodes, len_nodes))

        X_ele = self.X_nodes[self.elements]
        Y_ele = self.Y_nodes[self.elements]
        X_diff = X_ele[:, np.newaxis].reshape(-1, 3, 1) - X_ele[np.newaxis, :].reshape(-1, 1, 3)
        Y_diff = Y_ele[:, np.newaxis].reshape(-1, 3, 1) - Y_ele[np.newaxis, :].reshape(-1, 1, 3)
        distance = np.sqrt(X_diff ** 2 + Y_diff ** 2)
        row_indices = self.elements[:, :, np.newaxis].repeat(3, axis=2).reshape(-1)
        col_indices = self.elements[:, np.newaxis, :].repeat(3, axis=1).reshape(-1)
        dist_mat[row_indices, col_indices] = distance.reshape(-1)

        self.dist_mat = dist_mat.tocsr()

    def generate_nodes(self):
        return self.nodes

    def generate_elements(self):
        return self.elements

    def set_constants(self, constants):
        if not isinstance(constants, dict):
            raise ValueError("常数必须以字典类型给出")
        self.constants.update(constants)  # 更新而不是替换

    def generate_essential_condition(self,
                                     disp1_func=lambda x, y: 0.,  # type of x and y is <class 'float'>
                                     disp2_func=lambda x, y: 0.):
        print('----------------------生成本质边界条件----------------------')
        start = time.time()
        condition1 = (self.X_nodes == 0.)
        Idx1 = np.where(condition1)[0]
        X_ess1 = self.X_nodes[Idx1]
        Y_ess1 = self.Y_nodes[Idx1]
        condition2 = (self.X_nodes == 1)
        Idx2 = np.where(condition2)[0]
        X_ess2 = self.X_nodes[Idx2]
        Y_ess2 = self.Y_nodes[Idx2]

        """设置位移边界条件，形如 {node_id: [displacement_x, displacement_y]} (若自由边界则设置为None）"""
        Ess_bc = {}
        for idx, X, Y in zip(Idx1, X_ess1, Y_ess1):  # 施加位移荷载
            X = X.item()
            Y = Y.item()  # type(X) is <class 'float'>
            Ess_bc[idx] = [disp1_func(X, Y), disp2_func(X, Y)]
        for idx, X, Y in zip(Idx2, X_ess2, Y_ess2):
            X = X.item()
            Y = Y.item()  # type(X) is <class 'float'>
            Ess_bc[idx] = [disp1_func(X, Y), disp2_func(X, Y)]

        end = time.time()
        t = end - start
        print(f'生成本质边界条件耗时: {t:.4e}s')

        self.ess_bc = Ess_bc

    def generate_natural_condition(self,
                                   f_x=lambda x, y: 0.,  # type of x and y is <class 'float'>
                                   f_y=lambda x, y: 0.):
        print('----------------------生成自然边界条件----------------------')
        start = time.time()
        condition = (self.X_nodes == 48.)
        Idx = np.where(condition)[0]
        X_nat = self.X_nodes[Idx]
        Y_nat = self.Y_nodes[Idx]

        """设置力边界条件，形如 {node_id: [f_x, f_y]}"""
        Nat_bc = {}
        for idx, X, Y in zip(Idx, X_nat, Y_nat):  # 施加面力荷载
            Nat_bc[idx] = np.array([0., 0.])

            X = X.item()
            Y = Y.item()
            for x1, y1 in zip(X_nat, Y_nat):
                for x2, y2 in zip(X_nat, Y_nat):
                    id1 = np.where((self.X_nodes == x1) & (self.Y_nodes == y1))[0].item()
                    id2 = np.where((self.X_nodes == x2) & (self.Y_nodes == y2))[0].item()
                    grid_size = self.dist_mat[id1, id2]
                    Nat_bc[idx] += (np.array([f_x(X, Y), f_y(X, Y)])  # f_x = f_x(X, Y), f_y = f_y(X, Y)
                                    * (0.5 * grid_size))

        end = time.time()
        t = end - start
        print(f'生成自然边界条件耗时: {t:.4e}s')
        self.nat_bc = Nat_bc

    def compute_B_matrix(self, Nodes):
        """
        计算单元应变矩阵 B (3 行 6 列)
        """
        x1, y1 = Nodes[0]
        x2, y2 = Nodes[1]
        x3, y3 = Nodes[2]
        signed_area = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))  # 计算带正负号的三节点单元面积
        factor = 1 / (2 * signed_area)
        B = np.array([[y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
                      [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
                      [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]]) * factor
        return B, abs(signed_area)  # 曾出错误：忘记面积取绝对值

    def compute_constitutive_matrix(self):
        """
        计算本构矩阵 D
        """
        Young_modulus = self.Young_modulus  # 杨氏模量
        Poisson_ratio = self.Poisson_ratio  # 泊松比

        lmbd = Young_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        mu = Young_modulus / (2 * (1 + Poisson_ratio))

        D = np.array([[lmbd + 2 * mu, lmbd, 0],
                      [lmbd, lmbd + 2 * mu, 0],
                      [0, 0, mu]])
        return D

    def compute_element_stiffness(self, B, D, Area):
        """
        计算单元刚度矩阵
        """
        return np.dot(np.dot(B.T, D), B) * Area  # 曾出错误：忘记 × 面积

    def generate_stiffness_matrix(self, Displacements_bc):
        """
        生成总刚度矩阵 K_mat，并根据位移边界条件修改刚度矩阵 K
        K_mat 的 type 是 scipy.sparse._csr.csr_matrix，方便进行算数运算和矢量积运算
        """
        start = time.time()
        num_dofs = self.num_nodes * 2  # 自由度总数
        K_mat = lil_matrix((num_dofs, num_dofs))

        for element in self.elements:
            node_ids = element
            element_nodes = self.nodes[node_ids]

            # 计算单元刚度矩阵
            B, area = self.compute_B_matrix(element_nodes)
            D = self.compute_constitutive_matrix()
            Ke = self.compute_element_stiffness(B, D, area)

            # 将单元刚度矩阵装配到全局刚度矩阵中
            rows_K = np.repeat(node_ids * 2, 2) + np.tile([0, 1], 3)
            cols_K = np.repeat(node_ids * 2, 2) + np.tile([0, 1], 3)

            rows_Ke = np.arange(6)
            cols_Ke = np.arange(6)

            K_mat[np.ix_(rows_K, cols_K)] += Ke[np.ix_(rows_Ke, cols_Ke)]

        # 施加边界条件
        for node_id, node_displacement in Displacements_bc.items():
            r = node_id * 2
            c = node_id * 2

            # 相应主元素置大数
            for i in range(2):
                if node_displacement[i] is not None:
                    K_mat[r + i, c + i] = self.penalty

        end = time.time()
        t = end - start
        print(f'生成刚度矩阵耗时: {t:.4e}s')

        return K_mat.tocsr()  # 转换回CSR（压缩稀疏列矩阵）格式

    def compute_element_body_loads(self, Element, Area):
        """
        计算单元体力荷载向量
            b_x: float
                体力 x 方向分量
            b_y: float
                体力 y 方向分量
        """
        nodes_ele = self.nodes[Element]
        X_ele = nodes_ele[:, 0].reshape(-1, 1)
        Y_ele = nodes_ele[:, 1].reshape(-1, 1)
        b_x = 0.
        b_y = 0.

        # 设置体积力荷载向量
        ele_body_loads = np.array([])
        for X, Y in zip(X_ele, Y_ele):
            X = X.item()
            Y = Y.item()
            ele_body_loads = np.append(ele_body_loads, (np.array([b_x, b_y]) * (Area / 3)))
        return ele_body_loads.reshape(-1, 2)

    def generate_loading_vector(self, Forces_bc, Displacements_bc):
        """
        生成荷载向量
            Forces_bc: dictionary
                荷载边界条件，形如 {node_id: [fx, fy]}
            Displacements_bc: dictionary
                位移边界条件，形如 {node_id: [u, v]}
        """
        start = time.time()
        num_dofs = self.num_nodes * 2
        Loading_vector = np.zeros(num_dofs)

        for element in self.elements:
            node_ids = element
            element_nodes = self.nodes[node_ids]

            # 计算单元体力荷载
            B, area = self.compute_B_matrix(element_nodes)
            b_e = self.compute_element_body_loads(node_ids, area)

            # 将单元体力荷载装配到全局体力荷载向量中
            for i in range(3):
                for j in range(2):
                    r = node_ids[i] * 2 + j
                    Loading_vector[r] += b_e[i, j]

        # 处理加载边界条件
        for node_id, node_force in Forces_bc.items():
            row = node_id * 2
            Loading_vector[row:row + 2] += node_force

        # 处理位移边界条件
        for node_id, node_displacement in Displacements_bc.items():
            row = node_id * 2
            for i in range(2):
                if node_displacement[i] is not None:
                    Loading_vector[row + i] = node_displacement[i] * self.penalty

        end = time.time()
        t = end - start
        print(f'计算荷载向量耗时: {t:.4e}s')

        return Loading_vector

    def solve(self, save_path=f'./solutions/sol_{label}.npz'):
        """
        求解位移、应力、应变
        K_mat 的 type 是压缩稀疏列矩阵 scipy.sparse._csr.csr_matrix，方便进行算数运算和矢量积运算
        Loads_bc 形如 {node_id: [fx, fy]}
        Displacements_bc 形如 {node_id: [u, v]}
        Strains 的形状是 (num_nodes, 3)，type 是 numpy.ndarray
        Stresses 的形状是 (num_nodes, 3)，type 是 numpy.ndarray
        """
        Forces_bc = self.nat_bc
        Displacements_bc = self.ess_bc
        print('------------------------计算开始------------------------')
        start = time.time()
        Loading_vector = self.generate_loading_vector(Forces_bc, Displacements_bc)
        K_mat = self.generate_stiffness_matrix(Displacements_bc)
        Stresses = np.zeros((len(self.elements), 3))
        Strains = np.zeros((len(self.elements), 3))

        # 求解位移向量
        Displacements = spsolve(K_mat, Loading_vector)

        # 计算应力和应变
        for elem_idx, element in enumerate(self.elements):
            node_idx = element
            element_nodes = self.nodes[node_idx]
            B, area = self.compute_B_matrix(element_nodes)
            D = self.compute_constitutive_matrix()
            # 获取单元节点的位移
            displacements_element = np.zeros(6)
            for i in range(3):
                displacements_element[i * 2:i * 2 + 2] = Displacements[2 * node_idx[i]: 2 * node_idx[i] + 2]
            # 计算单元应变
            element_strain = np.dot(B, displacements_element)
            # 计算单元应力
            element_stress = np.dot(D, element_strain)
            # 存储单元应力和应变
            Strains[elem_idx, :] = element_strain
            Stresses[elem_idx, :] = element_stress

        end = time.time()
        t = end - start
        print(f'计算结束，共耗时: {t:.4e}s')

        np.savez(save_path, array1=self.nodes, array2=self.elements, array3=Displacements, array4=Stresses, array5=Strains)
        return Displacements, Stresses, Strains

    def plot_deformed_shapes(self, Displacements, scale_factor=10., savefig=False):
        Displacements = Displacements.reshape(-1, 2)
        fig = plt.figure()
        for elem in self.elements:
            node_ids = elem
            original_coords = self.nodes[node_ids]
            disp = Displacements[node_ids]
            deformed_coords = original_coords + scale_factor * disp

            # Plot original shape
            plt.plot(np.append(original_coords[:, 0], original_coords[0, 0]),
                     np.append(original_coords[:, 1], original_coords[0, 1]), 'r--')

            # Plot deformed shape
            plt.plot(np.append(deformed_coords[:, 0], deformed_coords[0, 0]),
                     np.append(deformed_coords[:, 1], deformed_coords[0, 1]), 'b-')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Deformed Shapes of Elements (Magnification: {scale_factor}×)')
        plt.axis('equal')
        plt.grid(True)

        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_deformed_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)

    def plot_displacement(self, Displacements, savefig=False):
        # 分离 x 方向和 y 方向的位移
        displacement_x = Displacements[::2]
        displacement_y = Displacements[1::2]

        # 创建网格
        X_plot = self.nodes[:, 0]
        Y_plot = self.nodes[:, 1]

        # 假设你有一个掩膜函数来确定哪些三角形需要掩膜
        # 这里提供一个示例函数，可以根据实际情况进行调整
        polygon = Polygon(self.boundary_nodes)

        def is_triangle_inside():
            inside_mask = np.ones(triangles.shape[0], dtype=bool)
            for i, tri_idx in enumerate(triangles):
                x_coords = X_plot[tri_idx]
                y_coords = Y_plot[tri_idx]
                centroid = Point(np.mean(x_coords), np.mean(y_coords))
                if polygon.contains(centroid):
                    inside_mask[i] = False
            return inside_mask

        # 创建三角剖分
        Tri = tri.Triangulation(X_plot, Y_plot)

        # 获取三角形数据
        triangles = Tri.triangles

        # 应用掩膜
        mask = is_triangle_inside()
        Tri.set_mask(mask)

        # 绘制位移图
        fig = plt.figure(figsize=(11.5, 5))

        plt.subplot(1, 2, 1)
        sc = plt.tripcolor(Tri, displacement_x, cmap='jet')
        plt.colorbar(sc)
        plt.axis('equal')
        plt.title('Displacement u')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        sc = plt.tripcolor(Tri, displacement_y, cmap='jet')
        plt.colorbar(sc)
        plt.axis('equal')
        plt.title('Displacement v')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_disp_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)

    def plot_stress(self, Stresses, savefig=False):
        Sig_x = Stresses[:, 0]
        Sig_y = Stresses[:, 1]
        Tau_xy = Stresses[:, 2]

        # 提取三角形的顶点坐标
        ele = self.elements
        xy = self.nodes
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        # 绘制三角形的伪彩色图
        fig = plt.figure(figsize=(16.6, 5))

        plt.subplot(1, 3, 1)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=Sig_x, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=Sig_y, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=Tau_xy, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\tau_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_stress_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)

    def plot_strain(self, Strains, savefig=False):
        e_xx = Strains[:, 0]
        e_yy = Strains[:, 1]
        e_xy = Strains[:, 2] * 0.5

        # 提取三角形的顶点坐标
        ele = self.elements
        xy = self.nodes
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        # 绘制三角形的伪彩色图
        fig = plt.figure(figsize=(16.9, 5))

        plt.subplot(1, 3, 1)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=e_xx, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=e_yy, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tripcolor(x_coords, y_coords, ele, facecolors=e_xy, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_strain_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)


boundary_nodes = np.array([
    [0., 0.],
    [1., 0.],
    [1., 1.],
    [0., 1.],
])

if __name__ == '__main__':
    E = 100.
    nu = 0.3
    w = 1e+4  # 罚数

    # 求解
    sol_path = f'solutions/sol_{label}.npz'
    linear_problem = LinearProblem2D(boundary_nodes, E, nu, penalty=w, max_area=1e-4)

    linear_problem.set_elements()


    class u_function:
        def __init__(self, ):
            self.P = 0.1
            self.I = 1 / 12
            self.G = E / (2 + 2 * nu)  # 剪切模量（=mu）
            self.h = 1.
            self.l = 1.

        def __call__(self, in_feature1, in_feature2):
            return (
                    (-self.P / (2 * E * self.I)) * in_feature1 ** 2 * (in_feature2 - 0.5)
                    - (nu * self.P / (6 * E * self.I)) * (in_feature2 - 0.5) ** 3
                    + (self.P / (6 * self.G * self.I)) * (in_feature2 - 0.5) ** 3
                    - (self.P * self.h ** 2 / (8 * self.G * self.I) - self.P * self.l ** 2 / (2 * E * self.I)) * (in_feature2 - 0.5)
            )


    class v_function:
        def __init__(self, ):
            self.P = 0.1
            self.I = 1 / 12
            self.G = E / (2 + 2 * nu)  # 剪切模量（=mu）
            self.h = 1.
            self.l = 1.

        def __call__(self, in_feature1, in_feature2):
            return (
                    (nu * self.P / (2 * E * self.I)) * in_feature1 * (in_feature2 - 0.5) ** 2
                    + (self.P / (6 * E * self.I)) * in_feature1 ** 3
                    - (self.P * self.l ** 2 / (2 * E * self.I) * in_feature1)
                    + (self.P * self.l ** 3 / (3 * E * self.I))
            )


    u_func = u_function()
    v_func = v_function()

    linear_problem.generate_essential_condition(disp1_func=u_func, disp2_func=v_func)
    linear_problem.generate_natural_condition()
    displacement, stress, strain = linear_problem.solve(save_path=sol_path)

    # 绘图
    # linear_problem.plot_deformed_shapes(displacement, scale_factor=0.1, savefig=False)
    linear_problem.plot_displacement(displacement, savefig=False)
    linear_problem.plot_stress(stress, savefig=False)
    linear_problem.plot_strain(strain, savefig=False)
