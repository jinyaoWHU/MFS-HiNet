import numpy as np


def find_topnode(edge_list, class_num):
    edge_array = np.array(edge_list)
    sub_edge_array = edge_array[class_num:].reshape(-1)
    counts = np.bincount(sub_edge_array)
    return np.argmax(counts)


def get_children(edge_array, current_node, parent_node):
    # 找所有包含 current_node 的边
    edges = edge_array[np.any(edge_array == current_node, axis=1)]

    # 去掉回头边
    if parent_node != -1:
        mask = np.any(edges == parent_node, axis=1)
        edges = edges[~mask]

    # 把 current_node 统一放到第一列
    indices = np.where(edges == current_node)[1]
    for i, idx in enumerate(indices):
        edges[i, [0, idx]] = edges[i, [idx, 0]]

    return edges[:, 1]


# def tree_structure(T, class_num):
#     edge_list = list(T.G.edges)
#     edge_array = np.array(edge_list)

#     node_num = np.max(edge_array)
#     classifier_num = node_num - class_num + 2
#     classifier_matrix = torch.zeros((classifier_num, node_num + 1))

#     # ===== root =====
#     top_node = find_topnode(edge_list, class_num)
#     root_children = get_children(edge_array, top_node, -1)

#     level = 0

#     # ✔ root 只要有子节点就记录（保证不会空）
#     if len(root_children) > 0:
#         classifier_matrix[level, top_node] = 2
#         classifier_matrix[level, root_children] = 1
#         level += 1

#     # ===== DFS =====
#     stack = [(top_node, child) for child in root_children]

#     while stack:
#         parent_node, current_node = stack.pop()

#         if level >= classifier_num:
#             break

#         children = get_children(edge_array, current_node, parent_node)

#         # ❗关键：只要有子节点就记录（保证叶子类别能被覆盖）
#         if len(children) == 0:
#             continue

#         classifier_matrix[level, current_node] = 2
#         classifier_matrix[level, children] = 1

#         level += 1

#         # ✔ 是否继续往下（只影响DFS，不影响当前记录）
#         next_nodes = children[children > class_num - 1]

#         for node in reversed(next_nodes):  # 保持DFS顺序稳定
#             stack.append((current_node, node))

#     # ===== 删除全0行 =====
#     nonzero_rows = torch.any(classifier_matrix != 0, dim=1)
#     classifier_matrix = classifier_matrix[nonzero_rows]

#     return classifier_matrix

def tree_structure(T, class_num, max_depth=None):
    """
    构建层次分类器矩阵

    Parameters
    ----------
    T : tree
        树结构对象，T.G 为 networkx graph

    class_num : int
        叶子类别数量
        默认:
            0 ~ class_num-1 为叶子类别
            >= class_num 为内部节点

    max_depth : int or None
        最大展开深度

        None:
            完整展开整棵树

        0:
            root直接分类所有叶子类别

        1:
            root展开一层
            超过部分自动折叠为叶子类别

    Returns
    -------
    classifier_matrix : torch.Tensor
        每行对应一个局部分类器

        2 : 当前分类节点
        1 : 当前分类器需要区分的类别
        0 : 无关节点
    """

    import numpy as np
    import torch

    edge_list = list(T.G.edges)
    edge_array = np.array(edge_list)

    node_num = np.max(edge_array)

    # =========================================================
    # 获取root节点
    # =========================================================
    top_node = find_topnode(edge_list, class_num)

    # =========================================================
    # 获取所有内部节点
    # =========================================================
    internal_nodes = set(edge_array.flatten()) - set(range(class_num))

    # root也算分类器
    classifier_num = len(internal_nodes)

    classifier_matrix = torch.zeros(
        (classifier_num, node_num + 1),
        dtype=torch.float32
    )

    # =========================================================
    # 工具函数：获取孩子节点
    # =========================================================
    def get_children_local(current_node, parent_node):

        children = []

        for edge in edge_list:

            u, v = edge

            if u == current_node and v != parent_node:
                children.append(v)

            elif v == current_node and u != parent_node:
                children.append(u)

        return np.array(children, dtype=int)

    # =========================================================
    # 工具函数：获取所有叶子后代
    # =========================================================
    def get_leaf_descendants(current_node, parent_node):

        children = get_children_local(current_node, parent_node)

        # 当前就是叶子类别
        if len(children) == 0:

            if current_node < class_num:
                return [current_node]

            return []

        leaf_nodes = []

        for child in children:

            # 叶子类别
            if child < class_num:

                leaf_nodes.append(child)

            # 内部节点
            else:

                leaf_nodes.extend(
                    get_leaf_descendants(child, current_node)
                )

        return leaf_nodes

    # =========================================================
    # root处理
    # =========================================================
    level = 0

    root_children = get_children_local(top_node, -1)

    # ===== 深度限制 =====
    if max_depth is not None and max_depth == 0:

        merged_children = []

        for child in root_children:

            # 叶子类别
            if child < class_num:

                merged_children.append(child)

            # 内部节点
            else:

                merged_children.extend(
                    get_leaf_descendants(child, top_node)
                )

        merged_children = np.unique(merged_children)

        classifier_matrix[level, top_node] = 2
        classifier_matrix[level, merged_children] = 1

        level += 1

    else:

        classifier_matrix[level, top_node] = 2
        classifier_matrix[level, root_children] = 1

        level += 1

    # =========================================================
    # DFS
    # =========================================================
    stack = [(top_node, child, 1) for child in root_children]

    while stack:

        parent_node, current_node, depth = stack.pop()

        # =====================================================
        # 当前节点的真实孩子
        # =====================================================
        children = get_children_local(current_node, parent_node)

        # 没有孩子
        if len(children) == 0:
            continue

        # =====================================================
        # 是否达到最大深度
        # =====================================================
        if max_depth is not None and depth >= max_depth:

            merged_children = []

            for child in children:

                # 叶子类别
                if child < class_num:

                    merged_children.append(child)

                # 内部节点
                else:

                    merged_children.extend(
                        get_leaf_descendants(child, current_node)
                    )

            merged_children = np.unique(merged_children)

            # 没有有效叶子类别
            if len(merged_children) == 0:
                continue

            classifier_matrix[level, current_node] = 2
            classifier_matrix[level, merged_children] = 1

            level += 1

            # 不再继续DFS
            continue

        # =====================================================
        # 正常局部分类器
        # =====================================================
        classifier_matrix[level, current_node] = 2
        classifier_matrix[level, children] = 1

        level += 1

        # =====================================================
        # 继续DFS（仅内部节点）
        # =====================================================
        next_nodes = children[children >= class_num]

        for node in reversed(next_nodes):

            stack.append(
                (current_node, node, depth + 1)
            )

    # =========================================================
    # 删除全0行
    # =========================================================
    nonzero_rows = torch.any(classifier_matrix != 0, dim=1)

    classifier_matrix = classifier_matrix[nonzero_rows]

    return classifier_matrix