import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri
from meshpy import triangle
from scipy.sparse import lil_matrix, coo_matrix
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
                 Young_modulus=1.,
                 Poisson_ratio=0.3,
                 lmbd_func=None,
                 mu_func=None,
                 max_area=20.0,
                 penalty=1e+4):
        self.constants = None
        self.nodes = None
        self.num_nodes = None
        self.X_nodes = None
        self.Y_nodes = None
        self.elements = None
        self.signed_area = None
        self.area = None
        self.num_elements = None
        self.dist_mat = None
        self.B = None
        self.D = None
        self.load_vector = None
        self.K_mat = None
        self.displacements = None
        self.strains = None
        self.stresses = None
        self.ess_bc = np.empty((0, 3))
        self.nat_bc = np.empty((0, 3))
        self.bx = np.empty((0, 3))
        self.by = np.empty((0, 3))
        self.boundary_nodes = Boundary_Nodes
        self.Young_modulus = Young_modulus  # 杨氏模量
        self.Poisson_ratio = Poisson_ratio  # 泊松比
        self.lmbd_func = lmbd_func
        self.mu_func = mu_func
        self.max_area = max_area
        self.penalty = penalty

        self.lmbd = self.Young_modulus * self.Poisson_ratio / ((1 + self.Poisson_ratio) * (1 - 2 * self.Poisson_ratio))
        self.mu = self.Young_modulus / (2 * (1 + self.Poisson_ratio))

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

        element_nodes = self.nodes[self.elements]# 提取节点坐标
        # Nodes 形状为 (n_elements, 3, 2)
        x = element_nodes[:, :, 0]  # 形状为 (n_elements, 3)
        y = element_nodes[:, :, 1]  # 形状为 (n_elements, 3)

        # 计算带正负号的面积
        self.signed_area = 0.5 * (
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
        )  # 形状为 (n_elements,)
        self.area = np.abs(self.signed_area).reshape(-1, 1)

        dist_mat = lil_matrix((self.num_nodes, self.num_nodes))

        X_ele = self.X_nodes[self.elements]
        Y_ele = self.Y_nodes[self.elements]
        X_diff = X_ele[:, np.newaxis].reshape(-1, 3, 1) - X_ele[np.newaxis, :].reshape(-1, 1, 3)
        Y_diff = Y_ele[:, np.newaxis].reshape(-1, 3, 1) - Y_ele[np.newaxis, :].reshape(-1, 1, 3)
        distance = np.sqrt(X_diff ** 2 + Y_diff ** 2)
        row_indices = self.elements[:, :, np.newaxis].repeat(3, axis=2).reshape(-1)
        col_indices = self.elements[:, np.newaxis, :].repeat(3, axis=1).reshape(-1)
        dist_mat[row_indices, col_indices] = distance.reshape(-1)

        self.dist_mat = dist_mat.tocsr()

        num_dofs = self.num_nodes * 2
        self.load_vector = np.zeros(num_dofs)

        self.displacements = np.zeros((num_dofs, ))
        self.stresses = np.zeros((self.num_elements, 3))
        self.strains = np.zeros((self.num_elements, 3))

    def generate_nodes(self):
        return self.nodes

    def generate_elements(self):
        return self.elements

    def set_constants(self, constants):
        if not isinstance(constants, dict):
            raise ValueError("常数必须以字典类型给出")
        self.constants.update(constants)  # 更新而不是替换

    def generate_essential_condition(self,
                                     disp1_func,  # type of x and y is <class 'float'>
                                     disp2_func,
                                     x_condition=lambda x: x < np.inf,
                                     y_condition=lambda y: y < np.inf,
                                     couple_condition=lambda x, y: x < np.inf):
        condition1 = x_condition(self.X_nodes)
        condition2 = y_condition(self.Y_nodes)
        condition3 = couple_condition(self.X_nodes, self.Y_nodes)
        condition = condition1 & condition2 & condition3

        Idx_ess = np.where(condition)[0]
        X_ess = self.X_nodes[Idx_ess]
        Y_ess = self.Y_nodes[Idx_ess]

        """设置位移边界条件，形如 {node_id: [displacement_x, displacement_y]} (若自由边界则设置为np.nan）"""
        u_values = disp1_func(X_ess, Y_ess)  # 计算结果，可能包含 np.nan
        v_values = disp2_func(X_ess, Y_ess)  # 计算结果，可能包含 np.nan
        new_ess_bc = np.hstack((Idx_ess.reshape(-1, 1), u_values.reshape(-1, 1), v_values.reshape(-1, 1)))
        self.ess_bc = np.vstack((self.ess_bc, new_ess_bc))

    def generate_natural_condition(self,
                                   fx_func,
                                   fy_func,
                                   x_condition=lambda x: x < np.inf,
                                   y_condition=lambda y: y < np.inf,
                                   couple_condition=lambda x, y: x < np.inf, ):
        print('----------------------生成自然边界条件----------------------')
        start = time.time()
        condition1 = x_condition(self.X_nodes)
        condition2 = y_condition(self.Y_nodes)
        condition3 = couple_condition(self.X_nodes, self.Y_nodes)
        condition = condition1 & condition2 & condition3

        Idx_nat = np.where(condition)[0]
        X_nat = self.X_nodes[Idx_nat]
        Y_nat = self.Y_nodes[Idx_nat]

        """设置力边界条件，形如 [[node_id, f_x, f_y]...]"""
        id1 = Idx_nat[:, np.newaxis].repeat(Idx_nat.size, axis=1)
        id2 = Idx_nat[np.newaxis, :].repeat(Idx_nat.size, axis=0)
        grid_sizes = self.dist_mat[id1, id2].sum(axis=0) * 0.5
        f_x = np.multiply(grid_sizes, fx_func(X_nat, Y_nat))
        f_y = np.multiply(grid_sizes, fy_func(X_nat, Y_nat))
        new_nat_bc = np.hstack((Idx_nat.reshape(-1, 1), f_x.reshape(-1, 1), f_y.reshape(-1, 1)))

        self.nat_bc = np.vstack((self.nat_bc, new_nat_bc))

        end = time.time()
        t = end - start
        print(f'生成自然边界条件耗时: {t:.4e}s')

    def compute_element_area(self, ele_Nodes):
        # 提取节点坐标
        # Nodes 形状为 (n_elements, 3, 2)
        x = ele_Nodes[:, :, 0]  # 形状为 (n_elements, 3)
        y = ele_Nodes[:, :, 1]  # 形状为 (n_elements, 3)

        # 计算带正负号的面积
        signed_area = 0.5 * (
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
        )  # 形状为 (n_elements,)
        area = np.abs(signed_area).reshape(-1, 1)
        return signed_area, area

    def compute_B_matrix(self):
        """
        计算单元应变矩阵 B
        """
        ele_Nodes = self.nodes[self.elements]
        n_elements = ele_Nodes.shape[0]

        # 提取节点坐标
        # Nodes 形状为 (n_elements, 3, 2)
        x = ele_Nodes[:, :, 0]  # 形状为 (n_elements, 3)
        y = ele_Nodes[:, :, 1]  # 形状为 (n_elements, 3)

        # 计算带正负号的面积
        signed_area, area = self.compute_element_area(ele_Nodes)  # 形状为 (n_elements,)
        factor = 1 / (2 * signed_area)  # 形状为 (n_elements,)

        # 计算差值
        dy = np.roll(y, -1, axis=1) - np.roll(y, 1, axis=1)  # y2 - y3, y3 - y1, y1 - y2
        dx = np.roll(x, 1, axis=1) - np.roll(x, -1, axis=1)  # x3 - x2, x1 - x3, x2 - x1

        # 构造 B 矩阵
        B = np.zeros((n_elements, 3, 6))  # 形状为 (n_elements, 3, 6)
        B[:, 0, 0] = dy[:, 0]  # B11
        B[:, 0, 2] = dy[:, 1]  # B13
        B[:, 0, 4] = dy[:, 2]  # B15
        B[:, 1, 1] = dx[:, 0]  # B22
        B[:, 1, 3] = dx[:, 1]  # B24
        B[:, 1, 5] = dx[:, 2]  # B26
        B[:, 2, 0] = dx[:, 0]  # B31
        B[:, 2, 1] = dy[:, 0]  # B32
        B[:, 2, 2] = dx[:, 1]  # B33
        B[:, 2, 3] = dy[:, 1]  # B34
        B[:, 2, 4] = dx[:, 2]  # B35
        B[:, 2, 5] = dy[:, 2]  # B36

        # 乘以因子
        self.B = B * factor[:, None, None]  # 形状为 (n_elements, 3, 6)

        return self.B

    def compute_constitutive_matrix(self):
        """
        计算本构矩阵 D
        """
        ele_Nodes = self.nodes[self.elements]
        n_elements = ele_Nodes.shape[0]

        # 提取节点坐标
        # Nodes 形状为 (n_elements, 3, 2)
        x = ele_Nodes[:, :, 0]  # 形状为 (n_elements, 3)
        y = ele_Nodes[:, :, 1]  # 形状为 (n_elements, 3)

        if self.lmbd_func is None:
            lmbd = self.lmbd * np.ones(ele_Nodes.shape[0])  # 形状为 (n_elements, )
        else:
            lmbd = self.lmbd_func(self.stresses, self.strains)

        if self.mu_func is None:
            mu = self.mu * np.ones(ele_Nodes.shape[0])  # 形状为 (n_elements, )
        else:
            mu = self.mu_func(self.stresses, self.strains)


        self.D = np.zeros((n_elements, 3, 3))  # 形状为 (n_elements, 3, 3)
        self.D[:, 0, 0] = lmbd + 2 * mu
        self.D[:, 0, 1] = lmbd
        self.D[:, 1, 0] = lmbd
        self.D[:, 1, 1] = lmbd + 2 * mu
        self.D[:, 2, 2] = mu

        return self.D

    def compute_element_stiffness(self, B, D, Area):
        """
        计算单元刚度矩阵
        """
        Area = Area.reshape(-1, 1, 1)
        Ke = np.einsum('nji,njk,nkl->nil', B, D, B) * Area  # 形状为 (n_elements, 6, 6)

        return Ke 

    def generate_stiffness_matrix(self):
        """
        生成总刚度矩阵
        """
        start = time.time()
        num_dofs = self.nodes.shape[0] * 2  # 自由度总数
        element_nodes = self.nodes[self.elements]  # 形状为 (n_elements, 3, 2)

        # 批量计算所有单元的刚度矩阵
        area = self.compute_element_area(element_nodes)[1]
        Ke = self.compute_element_stiffness(self.B, self.D, area)  # Ke 形状为 (n_elements, 6, 6)

        # 计算全局自由度索引
        node_ids = self.elements  # 形状为 (n_elements, 3)
        rows_K = np.repeat(node_ids * 2, 2, axis=1) + np.tile([0, 1], 3)  # 形状为 (n_elements, 6)
        cols_K = np.repeat(node_ids * 2, 2, axis=1) + np.tile([0, 1], 3)  # 形状为 (n_elements, 6)

        # 将单元刚度矩阵展平为 COO 格式
        rows = rows_K.repeat(6, axis=1).flatten()  # 形状为 (n_elements * 36,)
        cols = cols_K.repeat(6, axis=0).flatten()  # 形状为 (n_elements * 36,)
        data = Ke.flatten()  # 形状为 (n_elements * 36,)

        # 使用 COO 格式装配全局刚度矩阵
        K_mat = coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs))

        # 施加边界条件
        K_mat = K_mat.tolil()
        disp_ids = self.ess_bc[:, 0].astype(int)
        disp1 = self.ess_bc[:, 1]
        disp2 = self.ess_bc[:, 2]
        disp_rows = disp_ids * 2
        disp_cols = disp_ids * 2
        valid_disp1 = ~np.isnan(disp1)
        valid_disp2 = ~np.isnan(disp2)
        K_mat[disp_rows[valid_disp1], disp_cols[valid_disp1]] = self.penalty
        K_mat[disp_rows[valid_disp2] + 1, disp_cols[valid_disp2] + 1] = self.penalty

        self.K_mat = K_mat.tocsc()

        end = time.time()
        t = end - start
        print(f'生成刚度矩阵耗时: {t:.4e}s')

    def generate_body_force(self,
                            bx_func=lambda x, y: 0 * x * y,
                            by_func=lambda x, y: 0 * x * y):
        """
        计算单元体力荷载向量
        """
        element_nodes = self.nodes[self.elements]
        X_ele = element_nodes[:, :, 0]
        Y_ele = element_nodes[:, :, 1]

        # 计算体力分量
        self.bx = np.vstack(
            (self.bx, bx_func(X_ele, Y_ele) * self.area / 3)
        )
        self.by = np.vstack(
            (self.by, by_func(X_ele, Y_ele) * self.area / 3)
        )

    def generate_loading_vector(self):
        """
        生成荷载向量
        """
        start = time.time()

        bx_ids = self.elements * 2
        by_ids = self.elements * 2 + 1
        self.load_vector[bx_ids] += self.bx
        self.load_vector[by_ids] += self.by

        # 处理加载边界条件
        if self.nat_bc.size != 0:
            load_ids = self.nat_bc[:, 0].astype(int)
            f_x = self.nat_bc[:, 1]
            f_y = self.nat_bc[:, 2]
            load_rows = load_ids * 2
            self.load_vector[load_rows] += f_x
            self.load_vector[load_rows + 1] += f_y

        # 处理位移边界条件
        if self.ess_bc.size != 0:
            disp_ids = self.ess_bc[:, 0].astype(int)
            disp1 = self.ess_bc[:, 1]
            disp2 = self.ess_bc[:, 2]
            disp_rows = disp_ids * 2

            valid_disp1 = ~np.isnan(disp1)
            valid_disp2 = ~np.isnan(disp2)

            self.load_vector[disp_rows[valid_disp1]] = disp1[valid_disp1] * self.penalty
            self.load_vector[disp_rows[valid_disp2] + 1] = disp2[valid_disp2] * self.penalty

        end = time.time()
        t = end - start
        print(f'计算荷载向量耗时: {t:.4e}s')

        return self.load_vector

    def solve(self, save_tag=True, save_path=f'./solutions/sol_{label}.npz'):
        """
        求解位移、应力、应变
        """
        print('------------------------计算开始------------------------')
        start = time.time()

        # 求解位移向量
        delta_disp = spsolve(self.K_mat, self.load_vector)

        end = time.time()
        t = end - start
        print(f'计算结束，共耗时: {t:.4e}s')

        # 计算应力和应变
        delta_disp_reshaped = delta_disp.reshape(-1, 2)
        delta_disp_ele = delta_disp_reshaped[self.elements].reshape(self.num_elements, 6)

        delta_strains = np.einsum('ijk,ik->ij', self.B, delta_disp_ele)  # 形状为 (n_elements, 3)
        delta_stresses = np.einsum('ijk,ik->ij', self.D, delta_strains)  # 形状为 (n_elements, 3)

        self.displacements += delta_disp
        self.strains += delta_strains
        self.stresses += delta_stresses

        if save_tag:
            np.savez(
                save_path,
                array1=self.nodes, array2=self.elements,
                array3=self.displacements, array4=self.stresses, array5=self.strains
            )

        return self.displacements, self.stresses, self.strains

    def plot_deformed_shapes(self, scale_factor=10., savefig=False):
        Displacements = self.displacements.reshape(-1, 2)
        fig = plt.figure()
        for elem in self.elements:
            node_ids = elem
            original_coords = self.nodes[node_ids]
            disp = Displacements[node_ids]
            deformed_coords = original_coords + scale_factor * disp

            plt.plot(np.append(original_coords[:, 0], original_coords[0, 0]),
                     np.append(original_coords[:, 1], original_coords[0, 1]), 'r--')

            plt.plot(np.append(deformed_coords[:, 0], deformed_coords[0, 0]),
                     np.append(deformed_coords[:, 1], deformed_coords[0, 1]), 'b-')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Deformed Shapes of Elements (Magnification: {scale_factor}×)')
        plt.axis('equal')
        plt.grid(True)

        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_deformed_{label}.png', dpi=300, bbox_inches='tight', format='png',
                        transparent=False)

    def plot_displacement(self, savefig=False):
        displacement_x = self.displacements[::2]
        displacement_y = self.displacements[1::2]

        X_plot = self.nodes[:, 0]
        Y_plot = self.nodes[:, 1]

        polygon = Polygon(boundary_nodes)

        def is_triangle_inside():
            inside_mask = np.ones(triangles.shape[0], dtype=bool)
            for i, tri_idx in enumerate(triangles):
                coord1 = X_plot[tri_idx]
                coord2 = Y_plot[tri_idx]
                centroid = Point(np.mean(coord1), np.mean(coord2))
                if polygon.contains(centroid):
                    inside_mask[i] = False
            return inside_mask

        Tri = tri.Triangulation(X_plot, Y_plot)
        triangles = Tri.triangles 
        mask = is_triangle_inside() 
        Tri.set_mask(mask)

        fig = plt.figure(figsize=(11.5, 5))

        plt.subplot(1, 2, 1)
        plt.tricontourf(Tri, displacement_x, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title('Displacement u')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        plt.tricontourf(Tri, displacement_y, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title('Displacement v')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

        if savefig:
            fig.savefig(f'./figures/fig_disp_{label}.png', dpi=300, bbox_inches='tight', format='png',
                        transparent=False)

    def plot_stress(self, savefig=False, x_target=0, y_target=0):
        Sig_x = self.stresses[:, 0]
        Sig_y = self.stresses[:, 1]
        Tau_xy = self.stresses[:, 2]

        ele = self.elements
        xy = self.nodes
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

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
            fig.savefig(f'./figures/fig_stress_{label}.png', dpi=300, bbox_inches='tight', format='png',
                        transparent=False)

    def plot_strain(self, savefig=False):
        e_xx = self.strains[:, 0]
        e_yy = self.strains[:, 1]
        e_xy = self.strains[:, 2] * 0.5

        ele = self.elements
        xy = self.nodes
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

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
            fig.savefig(f'./figures/fig_strain_{label}.png', dpi=300, bbox_inches='tight', format='png',
                        transparent=False)


boundary_nodes = np.array([
    [0., 0.],
    [1., 0.],
    [1., 1.],
    [0., 1.],
])

if __name__ == '__main__':
    E = 100.
    nu = 0.3
    w = 1e+16  # 罚数

    # 求解
    sol_path = f'solutions/sol_{label}.npz'
    linear_problem = LinearProblem2D(boundary_nodes, E, nu, penalty=w, max_area=1e-4)

    linear_problem.set_elements()
    linear_problem.compute_B_matrix()
    linear_problem.compute_constitutive_matrix()


    def load_boundary_0(in_feature):
        return in_feature == 0.


    def load_boundary_1(in_feature):
        return in_feature == 1.


    class u_function:
        def __init__(self, ):
            self.P = 0.1
            self.I = 1 / 12
            self.G = E / (2 + 2 * nu)  # 剪切模量（=mu）
            self.h = 1.
            self.l = 1.

        def __call__(self, x, y):
            P = self.P
            I = self.I
            G = self.G
            h = self.h
            l = self.l
            return (
                    (-P / (2 * E * I)) * x ** 2 * (y - 0.5)
                    - (nu * P / (6 * E * I)) * (y - 0.5) ** 3
                    + (P / (6 * G * I)) * (y - 0.5) ** 3
                    - (P * h ** 2 / (8 * G * I) - P * l ** 2 / (2 * E * I)) * (y - 0.5)
            )

    class v_function:
        def __init__(self, ):
            self.P = 0.1
            self.I = 1 / 12
            self.G = E / (2 + 2 * nu)  # 剪切模量（=mu）
            self.h = 1.
            self.l = 1.

        def __call__(self, x, y):
            P = self.P
            I = self.I
            G = self.G
            h = self.h
            l = self.l
            return (
                    (nu * P / (2 * E * I)) * x * (y - 0.5) ** 2
                    + (P / (6 * E * I)) * x ** 3
                    - (P * l ** 2 / (2 * E * I) * x)
                    + (P * l ** 3 / (3 * E * I))
            )


    u_func = u_function()
    v_func = v_function()

    linear_problem.generate_body_force()
    linear_problem.generate_essential_condition(disp1_func=u_func,
                                                disp2_func=v_func,
                                                x_condition=load_boundary_1)
    linear_problem.generate_essential_condition(disp1_func=u_func,
                                                disp2_func=v_func,
                                                x_condition=load_boundary_0, )
    linear_problem.generate_stiffness_matrix()
    linear_problem.generate_loading_vector()

    linear_problem.solve(save_tag=True, save_path=sol_path)

    # 绘图
    # linear_problem.plot_deformed_shapes(scale_factor=0.1, savefig=False)
    linear_problem.plot_displacement(savefig=False)
    linear_problem.plot_stress(savefig=False)
    # linear_problem.plot_strain(savefig=False)
