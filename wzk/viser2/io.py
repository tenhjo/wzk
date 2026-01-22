import numpy as np
import trimesh

from wzk import spatial
import viser


def get_all_handles(scene: viser.SceneApi) -> dict[str, viser.SceneNodeHandle]:
    handles: dict[str, viser.SceneNodeHandle] = dict(scene._handle_from_node_name)
    return handles


def _get_local_trafo_dict(scene: viser.SceneApi) -> dict[str, np.ndarray]:
    local_trafo_dict: dict[str, np.ndarray] = {}
    for name, h in get_all_handles(scene=scene).items():

        impl = getattr(h, "_impl", None)
        if impl is None:
            continue

        wxyz = np.asarray(getattr(impl, "wxyz", None), dtype=np.float64)
        pos = np.asarray(getattr(impl, "position", None), dtype=np.float64)
        assert wxyz.shape == (4,)
        assert pos.shape == (3,)
        local_trafo_dict[name] = spatial.trans_quat2frame(trans=pos, quat=wxyz)

    return local_trafo_dict


def _get_global_trafo_dict(scene: viser.SceneApi) -> dict[str, np.ndarray]:

    local_trafo_dict = _get_local_trafo_dict(scene=scene)

    def parent_name(node_name: str) -> str | None:
        node_name = node_name.rstrip("/")
        if node_name == "" or node_name == "/":
            return None
        if "/" not in node_name[1:]:
            return ""  # top-level parent (root)
        return node_name.rsplit(sep="/", maxsplit=1)[0]

    # ---- compute world transforms with memoization ----
    global_trafo_dict: dict[str, np.ndarray] = {}

    def world_T(node_name: str) -> np.ndarray:
        if node_name in global_trafo_dict:
            return global_trafo_dict[node_name]
        T_local = local_trafo_dict.get(node_name, np.eye(4))
        p = parent_name(node_name)
        if p is None:
            Tw = T_local
        else:
            Tw = world_T(p) @ T_local
        global_trafo_dict[node_name] = Tw
        return Tw

    for name, h in get_all_handles(scene=scene).items():
        world_T(name)

    return global_trafo_dict


def combine_scene_to_trimesh(scene: viser.SceneApi) -> trimesh.Trimesh | None:
    """
    Combine all MeshHandle geometries in a viser Scene into one trimesh geometry
    """

    global_trafo_dict = _get_global_trafo_dict(scene=scene)
    mesh_list: list[trimesh.Trimesh] = []

    for name, h in get_all_handles(scene=scene).items():
        cls = type(h).__name__
        # -------- MeshHandle from add_mesh_simple --------
        if isinstance(h, viser.MeshHandle):
            props = h._impl.props
            if props is None:
                continue

            mesh = trimesh.Trimesh(vertices=np.asarray(props.vertices).copy(),
                                   faces=np.asarray(props.faces).copy(), process=False)

            # Apply scale from mesh props, then node world transform
            mesh.apply_transform(spatial.scale_matrix(props.scale))
            mesh.apply_transform(global_trafo_dict[name])

            mesh_list.append(mesh)
            continue

        if isinstance(h, viser.BatchedMeshHandle):
            raise NotImplementedError

        else:
            continue

    if not mesh_list:
        return None

    combined = trimesh.util.concatenate(mesh_list)

    combined.remove_unreferenced_vertices()
    return combined
