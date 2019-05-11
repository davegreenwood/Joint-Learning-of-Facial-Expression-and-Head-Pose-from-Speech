# pylint: disable=locally-disabled,  E1101, E0401
import argparse
from pkg_resources import resource_filename
import json
import numpy as np
import pyrr

from glumpy import app, gl, glm, gloo
from glumpy.app.movie import record


parser = argparse.ArgumentParser(description='Render a Speech Animation.')
parser.add_argument('infile', type=argparse.FileType('r'))
parser.add_argument('outfile')
parser.add_argument('--no-rigid', action='store_false',
                    help='Do not render rigid transformations')
args = parser.parse_args()


# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

framerate = 59.94
savename = args.outfile


def read_shader(fname):
    with open(fname, 'r') as fid:
        return fid.read()


def read_data():
    with args.infile as fid:
        data = json.load(fid)
    return data


def calc_edges(tris):
    def g(x): return tuple(sorted(x))

    def f(t): return g([t[0], t[1]]), g([t[1], t[2]]), g([t[2], t[0]])
    edges = set(i for t in tris for i in f(t))
    return np.array(sorted([x for x in edges]), dtype=np.uint32).flatten()


def calc_box(pts):
    _mean = pts.mean(0)
    _max = np.abs(pts).max(0) - _mean + 15
    bb = np.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
                   [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]],
                  dtype=np.float32) * _max + _mean
    return bb


def calc_mtx(translation, rotation):
    tt = np.array(translation, dtype=np.float32)
    rr = np.array(rotation, dtype=np.float32)

    if not args.no_rigid:
        return [np.eye(4).T for r in rr]

    def f(r, t):
        r = r / 180 * np.pi
        r = pyrr.matrix33.create_from_eulers(r)
        M = np.eye(4)
        M[0:3, 0:3] = r
        M[0:3, 3] = t
        return M.astype(np.float32).T

    return [f(r, t) for r, t in zip(rr, tt)]

# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------


data = read_data()
deforms = np.array(data['deforms'], np.float32)
tris = np.array(data['triangulation'], np.uint32)
mats = calc_mtx(data['translation'], data['rotation'])
bb = calc_box(deforms[0])
pts = np.vstack([deforms[0], bb])
n_mesh_pts = len(deforms[0])
n_deforms = len(deforms)
n_pts = len(pts)

# colours
grey = np.array([[0.7, 0.7, 0.7, 1]] * n_mesh_pts)
red = np.array([[1, 0, 0, 1]] * 8)
colors = np.vstack([grey, red])

# buffers
V = np.zeros(n_pts, [('a_position', np.float32, 3),
                     ('a_color', np.float32, 4)])

V['a_position'] = pts
V['a_color'] = colors

tri_idx = np.array(tris, dtype=np.uint32).flatten()
edge_idx = calc_edges(tris)
bb_idx = np.array([0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1,
                   5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4],
                  dtype=np.uint32) + n_mesh_pts

# change view of numpy arrays
V = V.view(gloo.VertexBuffer)
tri_idx = tri_idx.view(gloo.IndexBuffer)
edge_idx = edge_idx.view(gloo.IndexBuffer)
bb_idx = bb_idx.view(gloo.IndexBuffer)

# -----------------------------------------------------------------------------
# Shaders
# -----------------------------------------------------------------------------

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
uniform vec4   u_color;         // Global color
attribute vec4 a_color;         // Vertex color
attribute vec3 a_position;      // Vertex position
varying vec4   v_color;         // Interpolated fragment color (out)
varying vec2   v_texcoord;      // Interpolated fragment texture coords (out)

void main()
{
    v_color = u_color * a_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""


fragment = """
varying vec4   v_color;         // Interpolated fragment color (in)
void main()
{
    gl_FragColor = v_color;
}
"""

# -----------------------------------------------------------------------------
# OpenGL
# -----------------------------------------------------------------------------

program = gloo.Program(vertex, fragment)
program.bind(V)
program['u_model'] = mats[0]
program['u_view'] = glm.translation(0, 0, -1200)
window = app.Window(width=1200, height=1200, color=(1, 1, 1, 1))
k = 0


@window.event
def on_draw(dt):
    # index
    global k
    V["a_position"][:-8, :] = deforms[k]

    program['u_model'] = mats[k]
    k = (k + 1) % n_deforms

    window.clear()
    # mesh
    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
    program['u_color'] = 1, 1, 1, 1
    program.draw(gl.GL_TRIANGLES, tri_idx)

    # bb
    if args.no_rigid:
        program.draw(gl.GL_LINES, bb_idx)

    # outlines
    gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
    gl.glEnable(gl.GL_BLEND)
    gl.glDepthMask(gl.GL_FALSE)
    program['u_color'] = 0, 0, 0, 1
    program.draw(gl.GL_LINES, edge_idx)
    gl.glDepthMask(gl.GL_TRUE)


@window.event
def on_resize(width, height):
    program['u_projection'] = glm.perspective(
        15.0, width / float(height), 2.0, 2000.0)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)


with record(window, savename, fps=framerate):
    app.run(framerate=framerate, framecount=n_deforms)
