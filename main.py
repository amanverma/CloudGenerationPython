# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:05:33 2012

@author: Aman
"""

from vtk import *
import vtk
from vtk.util.numpy_support import *
import numpy as np
import itertools
import pprint

def generate_random_points(num_points, dims):
    g = lambda i: np.random.random_sample(num_points) * (dims[2 * i + 1] - dims[2 * i]) + dims[2 * i]
    x = g(0)
    y = g(1)
    z = g(2)
    return list(np.column_stack((x, y, z)))

def pointsToPolydata(pts):

    vtkPtsArray = numpy_to_vtk(np.array(pts, dtype=np.float32))
    vtkPts = vtkPoints()
    vtkPts.SetData(vtkPtsArray)
    vtkPts2 = vtkPoints()
    vtkPts2.DeepCopy(vtkPts)
    pd = vtkPolyData()
    pd.Allocate()
    pd.SetPoints(vtkPts2)
    print vtkPts.GetPoint(1), pts[1]
    ids = xrange(len(pts))
    for i in ids:
        idList = vtkIdList()
        idList.InsertNextId(i)
        pd.InsertNextCell(VTK_VERTEX, idList)
    return pd

class SectorScanGenerator(object):
    def __init__(self, sector_bounds, pts, error=1.0):
        self.pts = pts
        self.sector_bounds = np.array(sector_bounds, dtype=float)
        self.error = error
        self.first = True

    def transform_bounds(self, origin):
        transformer = origin[0], origin[0], origin[1], origin[1], origin[2], origin[2]
        return self.sector_bounds + transformer

    def create_sector_scan(self, origin):
        bounds = self.transform_bounds(origin)
        bounds_checker = lambda x, idx: bounds[2 * idx] <= x <= bounds[2 * idx + 1]
        point_selector = lambda x: bounds_checker(x[0], 0) and bounds_checker(x[1], 1) and bounds_checker(x[2], 2)
        selected_points = np.array(filter(point_selector, self.pts))
        if len(selected_points) == 0:
            return None
        sector_scan_points = selected_points - origin
        if not self.first:
            return sector_scan_points + np.random.random(sector_scan_points.shape) * self.error
        else:
            self.first = False
            return sector_scan_points

def visualize(pts, pts2):
    pd = pointsToPolydata(pts)
    pd2 = pointsToPolydata(pts2)
    view = vtkRenderView()
    rep = vtkRenderedSurfaceRepresentation()
    rep.SetInput(pd)
    theme = vtkViewTheme()
    theme.SetPointSize(1)
    rep.ApplyViewTheme(theme)
    rep2 = vtkRenderedSurfaceRepresentation()
    rep2.SetInput(pd2)
    theme2 = vtkViewTheme()
    theme2.SetPointSize(5)
    rep2.ApplyViewTheme(theme2)
    view.AddRepresentation(rep)
    view.AddRepresentation(rep2)
    #view.ResetCamera()
    iren = view.GetInteractor()
    iren.Initialize()
    iren.Start()

class Writer(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.file.write("""parameters
EuclideanFitnessEpsilon.Value 0.8
MaximumIterations.Value 250
MaxCorrespondenceDistance.Value 5
RANSACIterations.Value 10
TransformationEpsilon.Value 1.0
Tolerance.Value 4.0
MaximumFitness.Value 0.5
DefaultPercentageofPoints.Value 60
MaxNearestSquaredDistanceforFitness.Value 3
NumberOfFramesForAlignment.Value 3
MinimumKeypoints.Value 4
Debug.Value 1
endparameters
""")
    def write(self, points, origin):
        self.file.write('reset\n')
        self.file.write('# Origin: %f %f %f\n'%(origin[0], origin[1], origin[2]))
        pts_string = ('%f %f %f\n'%(i[0], i[1], i[2]) for i in points)
        self.file.writelines(pts_string)


    def __del__(self):
        self.file.write('end\n')
        self.file.write('writeOutput out.vtk')
        self.file.close()

if __name__ == '__main__':
    # For 3D, make the last two elements of fullBounds -30 and 30.
    fullBounds = [-50, 250, -50, 50, -50, 250]
    sectorScanBounds = [-50, 30, -30, 50, -50, 50]

    pts = generate_random_points(150,fullBounds)
    generator = SectorScanGenerator(sectorScanBounds, pts, error = 0.0)
    pd = pointsToPolydata(list(pts))
    pd.Update()
    print len(pts)
    print pd
    writer = vtkPolyDataWriter()
    writer.SetInput(pd)
    writer.SetFileName('points.vtk')
    writer.Update()
    start = np.zeros(3)
    min_pos = 0
    max_pos = 250
    step = 1
    np.set_printoptions(precision=2)
    writer = Writer('Input-traversal3D.txt')
    
    for i in np.arange(min_pos, max_pos, step):
        origin = start
        if i != min_pos:
            if (i<100):
                origin =  (0,0,0) #(i+np.random.random() ,0 , 0)            
            if(i>=100 and i<200):
                origin = (0,0,0) #(i + np.random.random(), 0 , i-100 + np.random.random())
            iasdfG12345*()f(i>=200):
                origin = (0,0,0) #(i+np.random.random(),0,100)

            v1 = origin[0]
            v2 = origin[2]
            import matplotlib.pyplot as pt
            pt.title('Origin X,Z')
            pt.xlabel('X Position')
            pt.ylabel('Z Position')
            pt.plot(v1,v2,'bo')
            
        sector_points = generator.create_sector_scan(origin)
        newPolydata = pointsToPolydata(list(sector_points))
        angle =i;
        newPolydata.Update()
        transform = vtk.vtkTransform()
        transform.RotateY(i)
        transform.Update()
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInput(newPolydata)
        transformFilter.Update()
        nPolyData = vtkPolyData()
        nPolydata = transformFilter.GetOutput()
        print newPolydata
        print nPolydata
        wrt = vtkPolyDataWriter()
        wrt.SetInput(nPolydata);
        wrt.SetFileName(str(i)+'.vtk')
        wrt.Update()
        if sector_points is not None:
            writer.write(sector_points, origin)
    pt.show()
