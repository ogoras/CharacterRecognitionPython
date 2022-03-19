import os
import math
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from PIL import Image

#parametry

imside = 28
padding = 3
brush_size = 2
factor = -(2.0/brush_size)**2

in_root_dir = "data/Contours/"
out_root_dir = "data/images/"

def draw_pixel(img, x, y, val):
    if (x < 0 or x >= imside or y <0 or y >= imside):
        return
    if (255 - img[x][y] > val):
        img[x][y] += val
    else:
        img[x][y] = 255

def draw_dot(img, x, y):
    #draw_pixel(img, x, y, 255)
    for ox in range(-brush_size,brush_size+1):
        for oy in range(-brush_size,brush_size+1):
            draw_pixel(img, x+ox, y+oy, 20*math.exp( factor * (ox**2 + oy**2)))



# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])




def convert_directory(in_dir, out_dir, dirname):
    if dirname[0] == '.':
        return
    directory = in_dir + dirname
    directory2 = out_dir + dirname
    try:
        os.makedirs(directory2)
    except FileExistsError:
        pass
    for filename in os.listdir(directory):
        out_filename = directory2 + "/" + filename
        out_filename = out_filename[:-4] + ".jpg"
        filename = directory + "/" + filename
        if os.path.exists(out_filename):
            continue
        f = open(filename, "r")
        
        strokes = []
        strokes.append([])
        maximum_x = (-math.inf)
        maximum_y = (-math.inf)
        minimum_x = (math.inf)
        minimum_y = (math.inf)
        
        stroke_index = 0
        for line in f.readlines():
            line = line[:-1]
            if line == '':
                strokes.append([])
                stroke_index += 1
                continue
            coords = line.split(' ')
#transformacje związane z obróceniem tablicy w środowisku (powinno być trochę bardziej elegancko :?)
            strokes[stroke_index].append([float(coords[0]), -float(coords[1])])
            if (maximum_x < float(coords[0])):
                maximum_x = float(coords[0])
            if (maximum_y < -float(coords[1])):
                maximum_y = -float(coords[1])
            if (minimum_x > float(coords[0])):
                minimum_x = float(coords[0])
            if (minimum_y > -float(coords[1])):
                minimum_y = -float(coords[1])
        
        imrange = imside - 2*padding
        
        range_x = maximum_x - minimum_x
        range_y = maximum_y - minimum_y

        maxrange = max(range_x , range_y)
        if maxrange == 0:
            normalized_strokes = [[[imside/2, imside/2]]]
        else:
            ratio =  min(range_x, range_y) / maxrange
    
            if (range_x > range_y):
                normalized_strokes = [[[(vector[0]-minimum_x)*imrange/maxrange + padding, 
                               (vector[1]-minimum_y)*imrange/maxrange + padding + imrange*(1-ratio)/2] 
                                       for vector in stroke] for stroke in strokes]
            else:
                normalized_strokes = [[[(vector[0]-minimum_x)*imrange/maxrange + padding + imrange*(1-ratio)/2,
                               (vector[1]-minimum_y)*imrange/maxrange + padding] for vector in stroke] for stroke in strokes]
        
        img = 255 * np.zeros([imside,imside], dtype=np.uint8)
    
        for normed_stroke in normalized_strokes:
            if (len(normed_stroke) == 0):
                continue
            if (len(normed_stroke) == 1):
                draw_dot(img, int(normed_stroke[0][0]), int(normed_stroke[0][1]))
            else:    
                path = evaluate_bezier(np.array(normed_stroke), 50)
                px, py = path[:,0], path[:,1]
                n = len(px)
                for index in range(n):
                    draw_dot(img, int(px[index]), int(py[index]))
        
        im = Image.fromarray(img)
        im.save(out_filename)
    print("Directory " + dirname + " finished conversion.")


for sub_dirname in os.listdir(in_root_dir):
    try:
        os.makedirs(out_root_dir + sub_dirname)
    except FileExistsError:
        pass
    for dirname in os.listdir(in_root_dir + sub_dirname):
        convert_directory(in_root_dir + sub_dirname + "/", out_root_dir + sub_dirname + "/", dirname)
