import numpy as np
import cv2
from osgeo import gdal, osr, ogr
import os
import subprocess
import sys
import json
import random
import csv
from config import cfg

shape_file_type = cfg.output_shape_type

def raster_to_shape(in_image, in_path, out_path):
    in_file = in_path + in_image
    os.mkdir(out_path + in_image.split('.')[0])
    
    # Vectorizing with gdal.Polygonize
    try:
        sourceRaster = gdal.Open(in_file)
        band = sourceRaster.GetRasterBand(1)
        if shape_file_type == 'ESRI Shapefile':
            driver = ogr.GetDriverByName("ESRI Shapefile")
            outShp = out_path + in_image.split('.')[0] + '/' + in_image.replace('.tif','.shp')
        elif shape_file_type == 'GeoJSON':
            driver = ogr.GetDriverByName("geojson")
            outShp = out_path + in_image.split('.')[0] + '/' + in_image.replace('.tif','.geojson')
        
        # If shapefile already exist, delete it
        if os.path.exists(outShp):
            driver.DeleteDataSource(outShp)
            
        outDatasource = driver.CreateDataSource(outShp)            
        # get proj from raster            
        srs = osr.SpatialReference()
        srs.ImportFromWkt( sourceRaster.GetProjectionRef() )
        # create layer with proj
        outLayer = outDatasource.CreateLayer(outShp, srs)
        
        # Add class column (0,255) to shapefile
        newField = ogr.FieldDefn('Class', ogr.OFTInteger)
        outLayer.CreateField(newField)
        gdal.Polygonize(band, None,outLayer, 0,[],callback=None)  
        outDatasource.Destroy()
        sourceRaster=None
    except Exception as e1:
        print('gdal Polygonize Error: ' + str(e1))
    
    # Smoothing the Vectors using SimplifyPreserveTopology
    try:
        sieveBuildingSize = cfg.min_building_area
        tolerance = cfg.tolerance
        
        ioShpFile = ogr.Open(outShp, update = 1)
        
        lyr = ioShpFile.GetLayerByIndex(0)
        lyr.ResetReading()
        
        field_defn = ogr.FieldDefn("Area", ogr.OFTReal)
        lyr.CreateField(field_defn)

        for i in lyr:
            geom = i.GetGeometryRef()
            area = round(geom.GetArea())
            
            lyr.SetFeature(i)
            i.SetField( "Area", area )
            lyr.SetFeature(i)

            objectClass = i.GetFieldAsInteger('class')

            # if area is less than threshold size or polygon not of building class, remove polygon
            if objectClass == 0 or (objectClass == 255 and area < sieveBuildingSize):
                lyr.DeleteFeature(i.GetFID())
        
        lyr.ResetReading()
        for i in lyr:
            geom = i.GetGeometryRef()
            i.SetGeometry(geom.SimplifyPreserveTopology(tolerance))
            lyr.SetFeature(i)
            
        ioShpFile.Destroy()
    except Exception as e2:
        print('Polygon smoothing Error: ' + str(e2))

def convert_raster_to_geoTiff(in_image, in_img_path, in_raster_path, out_path):
    try:
        fileformat = "GTiff"
        dataset = gdal.Open(in_img_path + in_image, gdal.GA_ReadOnly)
        driver = gdal.GetDriverByName(fileformat)
        dst_ds = driver.Create(out_path + in_image, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize, bands=1, eType=gdal.GDT_Byte)
        dst_ds.SetGeoTransform(dataset.GetGeoTransform())
        dst_ds.SetProjection(dataset.GetProjection())        
        img = cv2.imread(in_raster_path + in_image, 0)
        dst_ds.GetRasterBand(1).WriteArray(img)
    except Exception as e:
        print("Error: " + str(e))

def raster_to_shape_dir(in_path, out_path):
    for fi in [x for x in os.listdir(in_path) if x.endswith('.tif')]:
        raster_to_shape(fi, in_path, out_path)

def convert_raster_to_geoTiff_dir(in_img_path, in_raster_path, out_path):
    for fi in [x for x in os.listdir(in_raster_path) if x.endswith('.tif')]:
        convert_raster_to_geoTiff(fi, in_img_path, in_raster_path, out_path)
