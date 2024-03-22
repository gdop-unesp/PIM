import rebound
import folium
from nrlmsise00 import msise_model
import datetime
from datetime import datetime, timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import sin,cos,sqrt
from pyproj import Transformer
from PyAstronomy import pyasl
import os
import math
import sympy as sym
import variablesPIM
import validationPIM
from sympy import *
import sympy as sym
from tkinter import *


import datetime
from pyproj import Transformer
import numpy as np
import PyAstronomy.pyasl as pyasl

def conv_eq_to_horizon(year, month, day, hour, minute, second, altitude, longitude, latitude, right_ascension, declination):
    '''
    Converts equatorial coordinates to horizontal coordinates, returning the azimuth and altitude.

    Args:
        year (int): Year.
        month (int): Month.
        day (int): Day.
        hour (int): Hour.
        minute (int): Minute.
        second (int): Second.
        altitude (float): Altitude of the observer in meters.
        longitude (float): Longitude of the observer in degrees.
        latitude (float): Latitude of the observer in degrees.
        right_ascension (float): Right ascension of the object in degrees.
        declination (float): Declination of the object in degrees.

    Returns:
        tuple: A tuple containing the azimuth (in degrees) and altitude (in degrees) of the object.
    '''
    jd = datetime.datetime(year, month, day, hour, minute, second)
    jd_utc = pyasl.jdcnv(jd)  # Convert from Gregorian to Julian calendar
    right_ascension = float(right_ascension)
    declination = float(declination)

    altitude, azimuth, _ = pyasl.eq2hor(jd_utc, right_ascension, declination, lon=longitude, lat=latitude, alt=altitude*1000)
    return azimuth[0], altitude[0]

def conv_spherical_to_cartesian_geo(spherical_coords):
    '''
    Converts spherical coordinates to geographic coordinates.

    Args:
        spherical_coords (tuple): A tuple containing the radial distance (in km), polar angle (in radians), and azimuthal angle (in radians).

    Returns:
        np.ndarray: A 1D numpy array with the x, y, and z coordinates (in km) in the geographic coordinate system.
    '''
    r_c = float(spherical_coords[0] * 1000)
    lon_c = float(np.rad2deg(spherical_coords[1]))
    lat_c = float(np.rad2deg(spherical_coords[2]))
    trans_proj_geo = Transformer.from_crs("lla", {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
    x_c, y_c, z_c = trans_proj_geo.transform(lat_c, lon_c, r_c)
    return np.array([x_c, y_c, z_c]) / 1000

def conv_spherical_to_cartesian(spherical_coords):
    '''
    Converts spherical coordinates to Cartesian coordinates.

    Args:
        spherical_coords (tuple): A tuple containing the radial distance (in km), polar angle (in radians), and azimuthal angle (in radians).

    Returns:
        np.ndarray: A 1D numpy array with the x, y, and z coordinates (in km) in the Cartesian coordinate system.
    '''
    x_c = spherical_coords[0] * np.cos(spherical_coords[2]) * np.cos(spherical_coords[1])
    y_c = spherical_coords[0] * np.cos(spherical_coords[2]) * np.sin(spherical_coords[1])
    z_c = spherical_coords[0] * np.sin(spherical_coords[2])
    return np.array([x_c, y_c, z_c])

def conv_cartesian_to_geographic(xyz):
    '''
    Converts Cartesian coordinates to geographic coordinates.

    Args:
        xyz (np.ndarray): A 1D numpy array with the x, y, and z coordinates (in km) in the Cartesian coordinate system.

    Returns:
        np.ndarray: A 1D numpy array with the radial distance (in km), longitude (in radians), and latitude (in radians) in the geographic coordinate system.
    '''
    xyz_scaled = [xyz[0] * 1000, xyz[1] * 1000, xyz[2] * 1000]    
    transformer = Transformer.from_crs({"proj": 'geocent', "datum": 'WGS84', "ellps": 'WGS84'}, "lla")
    latitude, longitude, altitude = transformer.transform(xyz_scaled[0], xyz_scaled[1], xyz_scaled[2])
    return np.array([altitude / 1000, np.radians(longitude), np.radians(latitude)])

def conv_cartesian_to_spherical(xyz):
    '''
    Converts Cartesian coordinates to spherical coordinates.

    Args:
        xyz (np.ndarray): A 1D numpy array with the x, y, and z coordinates (in km) in the Cartesian coordinate system.

    Returns:
        np.ndarray: A 1D numpy array with the spherical coordinates (radius, polar angle, azimuthal angle) in radians.
    '''
    radius = sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
    polar_angle = np.arctan2(xyz[1], xyz[0])
    azimuthal_angle = np.arctan2(xyz[2], sqrt(xyz[0]**2 + xyz[1]**2))
    return np.array([radius, polar_angle, azimuthal_angle])

def translation(a, b):
    '''
    Performs vector addition.

    Args:
        a (np.ndarray): A 1D numpy array with the coordinates of vector a.
        b (np.ndarray): A 1D numpy array with the coordinates of vector b.

    Returns:
        np.ndarray: A 1D numpy array with the coordinates of the resulting vector.
    '''
    return (a + b)

def euclidean_distance_spherical(p1, p2):
    '''
    Calculates the Euclidean distance between two points in spherical coordinates.

    Args:
        p1 (np.ndarray): A 1D numpy array containing the spherical coordinates of the first point (radius, longitude, latitude).
        p2 (np.ndarray): A 1D numpy array containing the spherical coordinates of the second point (radius, longitude, latitude).

    Returns:
        float: The Euclidean distance between the two points.
    '''
    p1_deg = np.copy(p1)
    p2_deg = np.copy(p2)
    
    # Convert degrees to radians
    for i in range(1, 3):
        p1_deg[i] = np.radians(p1_deg[i])
        p2_deg[i] = np.radians(p2_deg[i])
    
    # Convert spherical coordinates to geocentric cartesians
    cart_p1 = conv_spherical_to_cartesian_geo(p1_deg)
    cart_p2 = conv_spherical_to_cartesian_geo(p2_deg)
    
    # Calculating the euclidean distance between positions
    return np.linalg.norm(cart_p1 - cart_p2)

def meteor_velocity(p1v, p2v, time): 
    '''
    Calculates the velocity of a meteor from its initial and final positions and the time between these positions.

    Args:
        p1v (np.ndarray): A 1D numpy array containing the initial position of the meteor in spherical coordinates (radius, azimuth, elevation).
        p2v (np.ndarray): A 1D numpy array containing the final position of the meteor in spherical coordinates (radius, azimuth, elevation).
        time (float): The time (in seconds) between the two positions.

    Returns:
        float: The velocity (in km/s) of the meteor.
    '''
    return euclidean_distance_spherical(p1v, p2v) / float(time)

def meteor_line_geocentric(meteor, station):
    '''
    Determines the line of the meteor in geocentric coordinates.

    Args:
        meteor (tuple): A tuple of three floats containing the meteor's apparent position in spherical coordinates (azimuth, elevation, distance).
        station (tuple): A tuple of three floats containing the station's geodetic coordinates in degrees (longitude, latitude, altitude).

    Returns:
        np.ndarray: A 1D numpy array with the geocentric coordinates of the meteor's line.
    '''    
    # Check input arguments
    r_cam, x_m, y_m, z_m = symbols('rCam xM yM zM')
    m_cartesian = conv_spherical_to_cartesian(meteor)
    m_geocentric = rot_axis2(np.pi / 2 - station[2]) @ m_cartesian
    m_geocentric = rot_axis3(station[1]) @ m_geocentric
    m_geocentric = np.array([-m_geocentric[0], m_geocentric[1], m_geocentric[2]])
    station_cartesian = conv_spherical_to_cartesian_geo(station)
    m_geocentric = translation(m_geocentric, station_cartesian)
    return m_geocentric

def plane_equation(station, meteor_a, meteor_b):
    '''
    Determines the equation of the plane defined by the station and two meteors' coordinates.

    Args:
        station (tuple): A tuple containing the station's coordinates in spherical coordinates (latitude, longitude, altitude).
        meteor_a (np.ndarray): A 1D numpy array containing the spherical coordinates of the first meteor (azimuth, zenith, distance).
        meteor_b (np.ndarray): A 1D numpy array containing the spherical coordinates of the second meteor (azimuth, zenith, distance).

    Returns:
        sympy.Expr: The equation of the plane defined by the station and the two meteors.
    '''
    # Convert station coordinates to Cartesian coordinates
    station_cartesian = conv_spherical_to_cartesian_geo(station)
    
    # Convert meteor coordinates to Cartesian coordinates
    meteor_a_geocentric = meteor_line_geocentric(meteor_a, station)
    meteor_b_geocentric = meteor_line_geocentric(meteor_b, station)

    # Calculate the normal vector of the plane defined by the station and the two meteors
    plane_normal = np.cross((meteor_a_geocentric - station_cartesian), (meteor_b_geocentric - station_cartesian))
    
    # Define the equation of the plane in terms of its normal vector and a point on the plane
    x, y, z = sym.symbols('xM yM zM')
    plane_equation = np.dot(plane_normal, np.array([x, y, z]))
    d = -1. * plane_equation.subs([(x, station_cartesian[0]), (y, station_cartesian[1]), (z, station_cartesian[2])])
    
    return plane_equation + d

def intersection_points(station, meteor_line_equation, meteor_position_100km):
    '''
    Finds the intersection points between the planes of each camera and returns the points of each camera on the meteor.

    Args:
        station (tuple): A tuple containing the geographical coordinates of the station (latitude, longitude, altitude).
        meteor_line_equation (sympy expression): The equation of the meteor's line.
        meteor_position_100km (np.ndarray): The position vector of the meteor in the horizontal coordinate system (100 km).

    Returns:
        tuple: A tuple containing the geographical coordinates of the intersection point between the planes of each camera and the meteor's line.
    '''
    r_cam, x_m, y_m, z_m = sym.symbols('rCam xM yM zM')
    meteor_cam = np.array([r_cam, meteor_position_100km[1], meteor_position_100km[2]])
    meteor_cam_geocentric = meteor_line_geocentric(meteor_cam, station)
    f = meteor_cam_geocentric[1] - (meteor_line_equation[y_m].subs(z_m, meteor_cam_geocentric[2]))
    r_f = solve(f, r_cam)
    x_coord = meteor_cam_geocentric[0].subs(r_cam, r_f[0])
    y_coord = meteor_cam_geocentric[1].subs(r_cam, r_f[0])
    z_coord = meteor_cam_geocentric[2].subs(r_cam, r_f[0])
    intersection_point = np.array([x_coord, y_coord, z_coord])
    intersection_point_geographic = conv_cartesian_to_geographic(intersection_point)
    intersection_point_geographic[1] = np.rad2deg(intersection_point_geographic[1])
    intersection_point_geographic[2] = np.rad2deg(intersection_point_geographic[2])
    
    return intersection_point_geographic
    
def meteor_data_geocentric(alt1, lon1, lat1, alt2, lon2, lat2, az1A, h1A, az2A, h2A, az1B, h1B, az2B, h2B):
    '''
    Determines the coordinates of the intersection points between the planes of each camera and the meteor's line.

    Args:
        alt1 (float): Altitude of station 1 in meters.
        lon1 (float): Longitude of station 1 in degrees.
        lat1 (float): Latitude of station 1 in degrees.
        alt2 (float): Altitude of station 2 in meters.
        lon2 (float): Longitude of station 2 in degrees.
        lat2 (float): Latitude of station 2 in degrees.
        az1A (float): Azimuth of meteor 1 observed from camera A in degrees.
        h1A (float): Zenith angle of meteor 1 observed from camera A in degrees.
        az2A (float): Azimuth of meteor 2 observed from camera A in degrees.
    '''
    station_1 = np.array([alt1, np.radians(lon1), np.radians(lat1)])
    station_2 = np.array([alt2, np.radians(lon2), np.radians(lat2)])
    meteor_1A_100km = np.array([300., np.radians(az1A), np.radians(h1A)])
    meteor_2A_100km = np.array([300., np.radians(az2A), np.radians(h2A)])
    meteor_1B_100km = np.array([300., np.radians(az1B), np.radians(h1B)])
    meteor_2B_100km = np.array([300., np.radians(az2B), np.radians(h2B)])
    x_m, y_m, z_m = symbols('xM yM zM')
    plane_1 = plane_equation(station_1, meteor_1A_100km, meteor_1B_100km)
    plane_2 = plane_equation(station_2, meteor_2A_100km, meteor_2B_100km)
    solution = sym.solve([plane_1, plane_2], (x_m, y_m, z_m))
    print(solution)
    intersection_point_1A_cam = intersection_points(station_1, solution, meteor_1A_100km)
    intersection_point_1B_cam = intersection_points(station_1, solution, meteor_1B_100km)
    intersection_point_2A_cam = intersection_points(station_2, solution, meteor_2A_100km)
    intersection_point_2B_cam = intersection_points(station_2, solution, meteor_2B_100km)
    return {'v1Acam' : intersection_point_1A_cam,'v1Bcam' : intersection_point_1B_cam,'v2Acam' : intersection_point_2A_cam,'v2Bcam' : intersection_point_2B_cam}

def read_input_file(file_path):
    '''
    Read the input file with meteor parameters.

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple: A tuple with the following elements:
            - str: Path to the input file.
            - pd.DataFrame: A pandas DataFrame with the meteor parameters.
            - list: A list with the meteor date and time [year, month, day, hour, minute, second].
            - str: The name of the meteor and the directory for calculation.
    '''
    df = pd.read_csv(file_path, sep='=', comment='#', index_col=0).transpose()
    df = df.apply(pd.to_numeric, errors='ignore')
    date_time = [df['ano'][0], df['mes'][0], df['dia'][0], df['hora'][0], df['minuto'][0], df['segundo'][0]]
    df['massaPont'] = df['massaPont'].astype(str)
    meteor_name = str(df['meteorN'][0])
    
    option = df['opcao'][0]

    return df, date_time, meteor_name, option

def points_intervals_case_1(readout):
    '''
    Calculate points and intervals for Case 1.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.

    Returns:
        tuple: A tuple containing latitude, longitude, altitude, deltaT for the two points.
    '''
    p1_lat = readout['P1lat'][0]
    p1_lon = readout['P1lon'][0]
    p1_alt =  readout['P1alt'][0]
    p2_lat =  readout['P2lat'][0]
    p2_lon = readout['P2lon'][0]
    p2_alt = readout['P2alt'][0]
    delta_t = readout['deltaT'][0]
    v_cam = readout['cam'][0]

    return p1_lat, p1_lon, p1_alt, p2_lat, p2_lon, p2_alt, delta_t, v_cam

def points_intervals_case_2(readout):
    '''
    Calculate points and intervals for Case 2.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.
        meteor_points (dict): A dictionary containing meteor points.

    Returns:
        tuple: A tuple containing latitude, longitude, altitude, deltaT for the two points.
    '''
    
    az1_ini = float(readout['az1Ini'][0])
    h1_ini  = float(readout['h1Ini'][0])
    az2_ini = float(readout['az2Ini'][0])
    h2_ini  = float(readout['h2Ini'][0])
    az1_fin = float(readout['az1Fin'][0])
    h1_fin  = float(readout['h1Fin'][0])
    az2_fin = float(readout['az2Fin'][0])
    h2_fin  = float(readout['h2Fin'][0])

    return az1_ini, h1_ini, az2_ini, h2_ini, az1_fin, h1_fin, az2_fin, h2_fin

def points_intervals_case_3(readout, date_meteor, alt1, lon1, lat1, alt2, lon2, lat2 ):
    '''
    Calculate points and intervals for Case 3.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.
        date_meteor (list): A list with meteor date and time [year, month, day, hour, minute, second].
        meteor_points (dict): A dictionary containing meteor points.

    Returns:
        tuple: A tuple containing latitude, longitude, altitude, deltaT for the two points.
    '''
    
    ra1_ini  = float(readout['ra1Ini'][0])
    dec1_ini = float(readout['dec1Ini'][0])
    ra2_ini  = float(readout['ra2Ini'][0])
    dec2_ini = float(readout['dec2Ini'][0])
    ra1_fin  = float(readout['ra1Fin'][0])
    dec1_fin = float(readout['dec1Fin'][0])
    ra2_fin  = float(readout['ra2Fin'][0])
    dec2_fin = float(readout['dec2Fin'][0])
    y, m, d, ho, mi, se = date_meteor[0],date_meteor[1],date_meteor[2],date_meteor[3],date_meteor[4],date_meteor[5]
    az1_ini, h1_ini = conv_eq_to_horizon(y, m, d, ho, mi, se, alt1, lon1, lat1, ra1_ini, dec1_ini)
    az2_ini, h2_ini = conv_eq_to_horizon(y, m, d, ho, mi, se, alt2, lon2, lat2, ra2_ini, dec2_ini)
    az1_fin, h1_fin = conv_eq_to_horizon(y, m, d, ho, mi, se, alt1, lon1, lat1, ra1_fin, dec1_fin)
    az2_fin, h2_fin = conv_eq_to_horizon(y, m, d, ho, mi, se, alt2, lon2, lat2, ra2_fin, dec2_fin)


    return az1_ini, h1_ini, az2_ini, h2_ini, az1_fin, h1_fin, az2_fin, h2_fin

def points_intervals_case_0(readout):
    '''
    Calculate points and intervals for Case 0.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.

    Returns:
        tuple: A tuple containing latitude, longitude, altitude, deltaT, Vx, Vy, Vz for the two points.
    '''
    p1_alt,p1_lon,p1_lat = readout['alt4d'][0],readout['lon4d'][0],readout['lat4d'][0]
    p2_alt,p2_lon,p2_lat = p1_alt,p1_lon,p1_lat
    vx4,vy4,vz4= readout['Vx4d'][0]*1000.,readout['Vy4d'][0]*1000.,readout['Vz4d'][0]*1000.
    delta_t=0
    return p1_lat, p1_lon, p1_alt, p2_lat, p2_lon, p2_alt, delta_t, vx4, vy4, vz4

def points_intervals_cases_2_or_3(readout):
    alt1 = float(readout['alt1'][0])
    lon1 = float(readout['lon1'][0])
    lat1 = float(readout['lat1'][0])
    alt2 = float(readout['alt2'][0])
    lon2 = float(readout['lon2'][0])
    lat2 = float(readout['lat2'][0])
    return alt1, lon1, lat1, alt2, lon2, lat2

def mass_point (readout):
    '''
    Parse and extract mass point information from the input DataFrame.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.

    Returns:
        list: A list of mass points.
    '''
    mass_points = []
    if readout['massaPont'][0].find(',') == -1:
        mass_points.append(float(readout['massaPont'][0]))

    else:
        mass_points_string = readout['massaPont'][0].split(sep=',')
        for i in mass_points_string:
            mass_points.append(float(i))
    return mass_points

def write_data (readout,meteor_n_path,p1_lat,p1_lon,p1_alt,p2_lat,p2_lon,p2_alt,delta_t,datetime_meteor,option,vx4=0,vy4=0,vz4=0):
    '''
    Write calculated data to an atmospheric_exit file.

    Args:
        readout (pd.DataFrame): A pandas DataFrame with meteor parameters.
        meteor_n (str): Name of the meteor.
        p1_lat (float): Latitude of point 1.
        p1_lon (float): Longitude of point 1.
        p1_alt (float): Altitude of point 1.
        p2_lat (float): Latitude of point 2.
        p2_lon (float): Longitude of point 2.
        p2_alt (float): Altitude of point 2.
        vx4 (float): Velocity along X-axis.
        vy4 (float): Velocity along Y-axis.
        vz4 (float): Velocity along Z-axis.
        delta_t (float): Time interval.
        datetime_meteor (datetime): Meteor date and time.
    '''
    input_file = open((meteor_n_path + '/data.txt'),'w')
    input_file.write(("Meteor: " + meteor_n_path + '\n \n'))
    linha = 'P1: lat: ' + str(p1_lat) + '  lon: ' + str(p1_lon) + '  alt: ' + str(p1_alt) + '\n'
    input_file.write(linha)

    if option != 4:
        linha = 'P2: lat: ' + str(p2_lat) + '  lon: ' + str(p2_lon) + '  alt: ' + str(p2_alt) + '\n'
        input_file.write(linha)

    else:
        linha = 'Vx (km/s): ' + str(vx4/1000) + '   Vy (km/s): ' + str(vy4/1000) + '  Vz (km/s): ' + str(vz4/1000) + '\n'
        input_file.write(linha)

    linha = 'time: ' + str(delta_t) + ' \n'
    input_file.write(linha)
    linha = 'date: ' + datetime_meteor.strftime("%Y-%m-%d %H:%M:%S") + '\n'
    input_file.write(linha)
    input_file.close()

def calculate_rho(alt, lat, lon):
    '''
    Calculates the atmospheric density (RHO) at a given altitude, latitude, and longitude.

    Args:
        alt (float): The altitude in meters above the Earth's surface.
        lat (float): The latitude in degrees.
        lon (float): The longitude in degrees.

    Returns:
        float: The atmospheric density (RHO) in kg/m³ at the specified location.
    '''

    # Get the current time
    meteor_hour = datetime.datetime.now()

    # Calculate atmospheric density using the MSISE model
    RHO = float(msise_model(
        datetime.datetime(meteor_hour.year, meteor_hour.month, meteor_hour.day, meteor_hour.hour, meteor_hour.minute, meteor_hour.second),
        alt=alt / 1000., lat=lat, lon=lon, f107=150., f107a=150, ap=4, lst=16)[0][5])

    # Convert RHO to kg/m³
    RHO *= 1000.

    return RHO

def update_particle(sim, rho, cd, a_section, mass_particle, alt_m, j):
    '''
    Update the particle's position and velocity based on atmospheric parameters.

    Args:
        sim (rebound.Simulation): The simulation object.
        rho (float): Atmospheric density in kg/m³ at the particle's location.
        cd (float): Drag coefficient.
        a_section (float): Cross-sectional area.
        mass_particle (float): Mass of the particle.
        j (int): Index of the particle.

    Returns:
        float: The time step (step) used for integration.
    '''
    ps = sim.particles
    vm = np.sqrt(ps[j].vx**2 + ps[j].vy**2 + ps[j].vz**2)

    if 150 < (alt_m[j] / 1000) < 990:
        step = -10000 / vm
    else:
        step = -2000 / vm

    ps[j].ax -= rho * cd * a_section * (np.sqrt(ps[j].vx**2 + ps[j].vy**2 + ps[j].vz**2)) * ps[j].vx / (2. * mass_particle)
    ps[j].ay -= rho * cd * a_section * (np.sqrt(ps[j].vx**2 + ps[j].vy**2 + ps[j].vz**2)) * ps[j].vy / (2. * mass_particle)
    ps[j].az -= rho * cd * a_section * (np.sqrt(ps[j].vx**2 + ps[j].vy**2 + ps[j].vz**2)) * ps[j].vz / (2. * mass_particle)

    return step

def geographic_to_eccf(geographic_point, lat, lon):
    '''
    Converts a geographic point to Earth-Centered Earth-Fixed (ECCF) coordinates.

    Args:
        geographic_point (np.ndarray): A 1D numpy array containing the geographic coordinates (x, y, z) in km.
        lat (float): Latitude of the point in degrees.
        lon (float): Longitude of the point in degrees.

    Returns:
        np.ndarray: A 1D numpy array with the ECCF coordinates of the point.
    '''
    rotated_plane = rot_axis2(np.pi / 2 - np.radians(lat)) @ (geographic_point)
    rotated_plane = rot_axis3(np.radians(lon)) @ (rotated_plane)
    rotated_plane = np.array([-rotated_plane[0], rotated_plane[1], rotated_plane[2]])
    return rotated_plane


def PIMRun(arquivoMeteoroEntrada, path, meteoritesField = False):
    
    readout, date_meteor, meteor_name, option = read_input_file(arquivoMeteoroEntrada)
    
###############################################################################################
# Points and Interval between the points
    if option == 1:
        P1_lat, P1_lon, P1_alt, P2_lat, P2_lon, P2_alt, deltaT= points_intervals_case_1(readout)
    
    elif option == 2 or 3:
        alt1, lon1, lat1, alt2, lon2, lat2 = points_intervals_cases_2_or_3(readout)

        if option == 2:
            az1ni, h1Ini, az2Ini, h2Ini, az1Fin, h1Fin, az2Fin, h2Fin = points_intervals_case_2(readout)
            
        elif option == 3:
            P1_lat, P1_lon, P1_alt, P2_lat, P2_lon, P2_alt, deltaT = points_intervals_case_3(readout, date_meteor, alt1, lon1, lat1, alt2, lon2, lat2 )
        
        meteor_points = (meteor_data_geocentric(alt1, lon1, lat1, alt2, lon2, lat2, 
                                az1ni, h1Ini, az2Ini, h2Ini, az1Fin, h1Fin, az2Fin, h2Fin))
        
        if readout['cam'][0] == 1:
            P1_alt,P1_lon,P1_lat = meteor_points['v1Acam'][0],meteor_points['v1Acam'][1],meteor_points['v1Acam'][2]
            P2_alt,P2_lon,P2_lat = meteor_points['v1Bcam'][0],meteor_points['v1Bcam'][1],meteor_points['v1Bcam']  [2]
            deltaT = readout['deltaT1'][0]
            vCam   = readout['cam'][0]

        else:
            P1_alt,P1_lon,P1_lat = meteor_points['v2Acam'][0],meteor_points['v2Acam'][1],meteor_points['v2Acam'][2]
            P2_alt,P2_lon,P2_lat = meteor_points['v2Bcam'][0],meteor_points['v2Bcam'][1],meteor_points['v2Bcam'][2]
            deltaT = readout['deltaT2'][0]
            vCam   = readout['cam'][0]

    else:
        P1_lat, P1_lon, P1_alt, P2_lat, P2_lon, P2_alt, deltaT, Vx4, Vy4, Vz4 = points_intervals_case_0(readout)
        
###############################################################################################
# Consolidating Data    
    datetimeMeteor = datetime.datetime(date_meteor[0],date_meteor[1],date_meteor[2],     
                         date_meteor[3],date_meteor[4],date_meteor[5])      # Meteor Timestamp (year, month, day, hour, minute, second)
    
    mass_point_var  = mass_point(readout)            # Masses for impact points kg [0.001, 0.01, 0.1, 1, 10, 50, 100, 150]
    CD         = readout['CD'][0]
    dens_meteor = readout['densMeteor'][0]       # Density
    massInt    = readout['massaInt'][0]          # Meteor mass
    tInt       = readout['tInt'][0]             # Integration time (days)
    tIntStep   = readout['tIntStep'][0]         # Integration step
    tamHill    = readout['tamHill'][0]          # Hill sphere size for close encounter

###############################################################################################
# Directory Creation
    print(meteor_name)
    meteor_name = meteor_name + " - Analyses"
    validationPIM.createDirIfDoesntExist(os.path.join(variablesPIM.directory, path), meteor_name)

###############################################################################################
# Record general information
    meteor_name_path = os.path.join(variablesPIM.directory, path, meteor_name)
    if option != 4:
        write_data(readout,meteor_name_path,P1_lat,P1_lon,P1_alt,P2_lat,P2_lon,P2_alt,deltaT,datetimeMeteor,option)
    else:
        write_data(readout,meteor_name_path,P1_lat,P1_lon,P1_alt,P2_lat,P2_lon,P2_alt,deltaT,datetimeMeteor,option,Vx4,Vy4,Vz4)
        
###############################################################################################
# Consolidating Data    
    P1 = np.array([P1_alt, P1_lon, P1_lat])
    P2 = np.array([P2_alt, P2_lon, P2_lat])
    distance = euclidean_distance_spherical(P1,P2)
    velocity = meteor_velocity(P1,P2,deltaT)
    dist_stations = euclidean_distance_spherical(np.array([alt1, lon1, lat1]),np.array([alt2, lon2, lat2]))
    legth_cam1 = euclidean_distance_spherical(meteor_points['v1Acam'],meteor_points['v1Bcam'])
    legth_cam2 = euclidean_distance_spherical(meteor_points['v2Acam'],meteor_points['v2Bcam'])
    speed_cam1 = meteor_velocity(meteor_points['v1Acam'],meteor_points['v1Bcam'],readout['deltaT1'][0])
    speed_cam2 = meteor_velocity(meteor_points['v2Acam'],meteor_points['v2Bcam'],readout['deltaT2'][0])
    initial_distance = euclidean_distance_spherical(meteor_points['v2Acam'],meteor_points['v1Acam'])
    final_distance = euclidean_distance_spherical(meteor_points['v2Bcam'],meteor_points['v1Bcam'])
    
###############################################################################################
# Write Data
    if option == 1:
        str_points_cam = f'Meteor Length (km):\n{distance}  \n--------------\n'
        str_points_cam += f'Meteor Speed (km/s):\n{velocity}\n--------------\n'
        with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
            filesCam.write(str_points_cam)
        if ( velocity < 11.):
            with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
                filesCam.write("Slow velocity")

    if option == 2 or option == 3:
        str_points_cam =  str(meteor_points) + '\n' + '--------------\n'
        str_points_cam += f'Distance between stations: \n{dist_stations}\n--------------\n'
        str_points_cam += f'Meteor cam1 Length (km):   \n{legth_cam1}   \n--------------\n'
        str_points_cam += f'Meteor cam2 Length (km):   \n{legth_cam2}   \n--------------\n'
        str_points_cam += f'Meteor cam1 Speed (km/s):  \n{speed_cam1}   \n--------------\n'
        str_points_cam += f'Meteor cam2 Speed (km/s):  \n{speed_cam2}   \n--------------\n'
        str_points_cam += "\n-----Distance Points A B between cameras-------\n"
        str_points_cam += f'Initial distance of the meteor between the cameras (km): \n{initial_distance}\n--------------\n'
        str_points_cam += f'Final distance of the meteor between the cameras (km): \n{final_distance}    \n--------------\n'
        
        with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
            filesCam.write(str_points_cam)
            filesCam.write("\n using cam = " + str(vCam)+ '\n')
            
        if (vCam == 1):
            if (meteor_velocity(meteor_points['v1Acam'],meteor_points['v1Bcam'],readout['deltaT1'][0]) < 11.):
                with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
                    filesCam.write("slow velocity")
            return
        elif (vCam == 2):
            if (meteor_velocity(meteor_points['v2Acam'],meteor_points['v2Bcam'],readout['deltaT2'][0]) < 11.):
                with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
                    filesCam.write("slow velocity")
                return
            
    if option == 4:
        velocityOp4 = str(sqrt(Vx4**2+Vy4**2+Vz4**2)/1000.)
        str_points_cam =f'Meteor Speed (km/s):\n{velocityOp4}\n--------------\n'
        with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
            filesCam.write(str_points_cam)
        if (velocityOp4<11.):
            with open(meteor_name_path+ '/'+'data.txt',"a") as filesCam:
                filesCam.write("slow velocity")

    print(str_points_cam)

###############################################################################################
# Initial meteor data in geocentric coordinates
# Create lists according to the number of masses to be integrated
    transproj_cart = Transformer.from_crs({"proj":'geocent',"datum":'WGS84',"ellps":'WGS84'},"lla")  
    transproj_geo = Transformer.from_crs("lla",{"proj":'geocent',"ellps":'WGS84',"datum":'WGS84'})

    A = []
    for i in mass_point_var:                     # Determine the masses to be run
        v = i/dens_meteor                        # Define the volume (mass/density)
        r = (v*3./(4.*math.pi))**(1./3.)/100.    # Define the radius
        A.append(math.pi*r*r)

###############################################################################################
# Initial conditions of the meteor, position, and velocity    
    X1    = [None] * len(mass_point_var)
    Y1    = [None] * len(mass_point_var)
    Z1    = [None] * len(mass_point_var)
    X2    = [None] * len(mass_point_var)
    Y2    = [None] * len(mass_point_var)
    Z2    = [None] * len(mass_point_var)
    Vx1   = [None] * len(mass_point_var)
    Vy1   = [None] * len(mass_point_var)
    Vz1   = [None] * len(mass_point_var)
    altM  = [None] * len(mass_point_var)
    latM  = [None] * len(mass_point_var)
    lonM  = [None] * len(mass_point_var)
    altM2 = [None] * len(mass_point_var)
    latM2 = [None] * len(mass_point_var)
    lonM2 = [None] * len(mass_point_var)

    particulas = [None] * len(mass_point_var)

    for i in range (len(mass_point_var)):
        X1[i],Y1[i],Z1[i] = transproj_geo.transform(P1_lat,P1_lon,P1_alt*1000.)
        X2[i],Y2[i],Z2[i] = transproj_geo.transform(P2_lat,P2_lon,P2_alt*1000.)
        if option == 4:
            Vx1[i] = Vx4
            Vy1[i] = Vy4
            Vz1[i] = Vz4
        else:
            Vx1[i] = (X2[i]-X1[i])/deltaT
            Vy1[i] = (Y2[i]-Y1[i])/deltaT
            Vz1[i] = (Z2[i]-Z1[i])/deltaT
        particulas[i]=i
        
    X = [None] * len(mass_point_var)
    Y = [None] * len(mass_point_var)
    Z = [None] * len(mass_point_var)
    Vx= [None] * len(mass_point_var)
    Vy= [None] * len(mass_point_var)
    Vz= [None] * len(mass_point_var)

################################################################################
# Build the wind table
    wind_is = False
    if os.path.isfile(str(variablesPIM.directory) + '/' + 'windTable.csv'):
        wind_table = pd.read_csv(variablesPIM.directory + '/' + 'windTable.csv',delimiter=';')
        wind_is = True
        
################################################################################
# Integration forward
# Integration of falling particles (data saved in .out files)

    # Initialize particle properties
    for i in range(len(mass_point_var)):
        X[i], Y[i], Z[i] = (X1[i] + X2[i]) / 2, (Y1[i] + Y2[i]) / 2, (Z1[i] + Z2[i]) / 2
        Vx[i], Vy[i], Vz[i] = Vx1[i], Vy1[i], Vz1[i]
        latM[i], lonM[i], altM[i] = transproj_cart.transform(X[i], Y[i], Z[i])

    # Create atmospheric_exit files for saving results
    if (meteoritesField):
        atmospheric_exit = open((meteor_name_path + '/atmospheric_exit.out'), 'w')
        fileFalling = open((meteor_name_path + '/fall.dat'), 'w')
        fileFalling.write("time(s) vel alt(km) lon lat \n")

        # Perform integration for each particle
        for j in range(len(mass_point_var)):
            time = 0
            step = 5
            ASection = A[j]
            massaParticula = mass_point_var[j]

            # Integrate until the particle's altitude is above 0
            while altM[j] > 0.:
                os.system('clear')
                Vm = np.sqrt(Vx[j]**2 + Vy[j]**2 + Vz[j]**2)

                # Adjust the time step based on altitude
                if ((altM[j] / 1000) < 0.005):
                    step = 0.1 / Vm
                elif ((altM[j] / 1000) < 0.040):
                    step = 2 / Vm
                elif ((altM[j] / 1000) < 0.150):
                    step = 20 / Vm
                elif ((altM[j] / 1000) < 0.2):
                    step = 50 / Vm
                elif ((altM[j] / 1000) < 0.4):
                    step = 100 / Vm
                elif ((altM[j] / 1000) < 1):
                    step = 200 / Vm
                elif ((altM[j] / 1000) < 3):
                    step = 500 / Vm
                elif ((altM[j] / 1000) < 5):
                    step = 1000 / Vm
                else:
                    step = 2000 / Vm
                
                if (wind_is):
                    wind_alt = wind_table.iloc[(wind_table['HGHT'] - altM[j]).abs().argsort()[:1]]
                    vWxHorizont,vWyHorizont = wind_alt['vx'].iloc[0] , wind_alt['vy'].iloc[0]
                    vWx,vWy,vWz = geographic_to_eccf(np.array([vWxHorizont,vWyHorizont,0.]),latM[j],lonM[j])
                else:
                    vWx,vWy,vWz = 0.,0.,0.

                print(time, "massa(kg): ", mass_point_var[j], "altura atual (km): ", altM[j] / 1000.)
                
                # Write altitude, velocity, and position to the atmospheric_exit file
                fileFalling.write(str(str(time) + " " + str(Vm) + " " + str(altM[j] / 1000) + " " + str(lonM[j]) + " " + str(latM[j]) + " \n"))

                # Initialize the simulation and add the Earth and meteor particle
                sim = rebound.Simulation()
                sim.integrator = "BS"
                sim.units = ('m', 's', 'kg')
                sim.add(m = 5.97219e24, hash ='earth')
                meteor = rebound.Particle(m = mass_point_var[j], x = X[j], y = Y[j], z = Z[j], vx = Vx[j], vy = Vy[j], vz = Vz[j])
                sim.add(meteor)

                ps = sim.particles

                # Calculate atmospheric density (RHO) at the meteor's location
                RHO = calculate_rho(altM[j], latM[j], lonM[j])

                # Define the drag force due to atmospheric density
                # Apply the drag force and integrate the simulation
                def drag(reb_sim):
                    vRx = ps[1].vx - vWx
                    vRy = ps[1].vy - vWy
                    vRz = ps[1].vz - vWz
                    vRMod = math.sqrt(vRx**2 + vRy**2 + vRz**2)
                    ps[1].ax -= RHO*CD*ASection*(vRMod)*vRx/(2.*massaParticula)
                    ps[1].ay -= RHO*CD*ASection*(vRMod)*vRy/(2.*massaParticula)
                    ps[1].az -= RHO*CD*ASection*(vRMod)*vRz/(2.*massaParticula)
                sim.additional_forces = drag                
                sim.force_is_velocity_dependent = 1
                sim.integrate(step)

                # Update time and meteor's position and velocity
                time += step
                global latA, lonA, altA
                latA, lonA, altA = latM[j], lonM[j], altM[j]
                X[j], Y[j], Z[j], Vx[j], Vy[j], Vz[j] = ps[1].x, ps[1].y, ps[1].z, ps[1].vx, ps[1].vy, ps[1].vz
                latM[j], lonM[j], altM[j] = transproj_cart.transform(ps[1].x, ps[1].y, ps[1].z)

                # Adjust time step for low altitudes
                if ((altM[j] / 1000) < 1):
                    step = 0.00001 / Vm
                if ((altM[j] / 1000) < 5):
                    step = 0.2 / Vm
                if ((altM[j] / 1000) < 10):
                    step = 100 / Vm

            # Write the final results for the meteor
            print(mass_point_var[j], latA, lonA, altA)
            file.write('  1-  ' + str(time) + ' g: ' + str(lonA) + " , " + str(latA) + ' , ' + str(altA / 1000.) + ' @399 \n')

        # Close the atmospheric_exit files
        atmospheric_exit.close()
        fileFalling.close()
        
###################################################################################################################
# Analysis of fall points
        atmospheric_exit  = pd.read_csv((meteor_name_path+'/atmospheric_exit.out'), sep='\s+',header=None)
        exclude = [0,2,4,6,7,8]
        atmospheric_exit.drop(atmospheric_exit.columns[exclude],axis=1,inplace=True)
        atmospheric_exit.insert(0, "mass", "Any")
        atmospheric_exit['mass'] = mass_point_var
        columns = ['mass','time(s)','lon','lat']
        atmospheric_exit.columns = columns
        file = open((meteor_name_path + '/data.txt'),'a')
        file.write(('\n \n Strewn Field: \n'))
        file.write(atmospheric_exit.to_string(index=False))
        file.write('\n \n')
        file.close()
        
################################################################################################################
# Falling map (forward integration)

        # Add a 'kg' suffix to the 'mass' column
        atmospheric_exit['mass'] = atmospheric_exit['mass'].astype(str) + " kg"

        # Create a map centered at the midpoint of lat and lon
        map_osm = folium.Map(
            location=[atmospheric_exit['lat'][atmospheric_exit.shape[0] // 2], atmospheric_exit['lon'][atmospheric_exit.shape[0] // 2]],
            zoom_start=14
        )

        # Add markers for the initial and final meteor positions
        folium.Marker(
            location=[P1_lat, P1_lon],
            popup=folium.Popup('Initial Meteor', show=True),
            icon=folium.map.Icon(color='blue')
        ).add_to(map_osm)

        folium.Marker(
            location=[P2_lat, P2_lon],
            popup=folium.Popup('Final Meteor', show=True),
            icon=folium.map.Icon(color='blue')
        ).add_to(map_osm)

        # Add markers for impact points with meteorite mass information
        for _, row in atmospheric_exit.iterrows():
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=folium.Popup(row['mass'], show=True),
                icon=folium.map.Icon(color='yellow')
            ).add_to(map_osm)

        # Save the fall map to an HTML file
        map_osm.save(meteor_name_path+ '/strewnField.html')

###################################################################################################################
# Reverse Integration
# Reverse integration up to 1000 km altitude (data saved in .out and .dat files)
# Using the mass in massaInt

    # Initialize particle properties
    for i in range(len(mass_point_var)):
        X[i], Y[i], Z[i] = (X1[i] + X2[i]) / 2, (Y1[i] + Y2[i]) / 2, (Z1[i] + Z2[i]) / 2
        Vx[i], Vy[i], Vz[i] = Vx1[i], Vy1[i], Vz1[i]
        latM[i], lonM[i], altM[i] = transproj_cart.transform(X[i], Y[i], Z[i])

    # Create atmospheric_exit files for saving results
    fileCart  = open((meteor_name_path + '/Cartesian.dat'), 'w')
    fileCoord = open((meteor_name_path + '/Coordinate.dat'), 'w')

    fileCart.write("time(s) x y z vx vy vz \n")
    fileCoord.write("time(s) vel alt(km) lon lat \n")

    # Set the initial time and find the index of the mass in massaInt
    time = 0
    step = -5
    massaParticula = float(massInt)
    v = massaParticula/dens_meteor
    r = (v*3./(4.*math.pi))**(1./3.)/100.
    ASection = (math.pi*r*r)
    j = 0
    mass_point_var[0] = massaParticula

    # Perform reverse integration until altitude reaches 1000 km
    while altM[j] < 1000e3:
        os.system('clear')
        print(time, "current altitude (km): ", altM[j] / 1000.)
        print(i, flush=True)
        
        Vm = np.sqrt(Vx[j]**2 + Vy[j]**2 + Vz[j]**2)
        if ((altM[j]/1000)>150) and ((altM[j]/1000)<990):
            step =- 10000/Vm
        else:
            step =- 2000/Vm
        
        if (wind_is):
            wind_alt = wind_table.iloc[(wind_table['HGHT'] - altM[j]).abs().argsort()[:1]]
            vWxHorizont, vWyHorizont = wind_alt['vx'].iloc[0] , wind_alt['vy'].iloc[0]
            vWx, vWy, vWz = geographic_to_eccf(np.array([vWxHorizont, vWyHorizont, 0.]), latM[j], lonM[j])
        else:
            vWx, vWy, vWz = 0.,0.,0.
            
        # Write Cartesian and Coordinate data to atmospheric_exit files
        strCart = str(time) + " " + str(X[j] / 1000.) + " " + str(Y[j] / 1000.) + " " + str(Z[j] / 1000.) + " " + str(
            Vx[j] / 1000.) + " " + str(Vy[j] / 1000.) + " " + str(Vz[j] / 1000.) + " \n"
        fileCart.write(strCart)
        fileCoord.write(
            str(time) + " " + str(Vm) + " " + str(altM[j] / 1000) + " " + str(lonM[j]) + " " + str(latM[j]) + " \n")
        
        sim = rebound.Simulation()
        sim.integrator = "ias15"
        sim.units = ('m', 's', 'kg')
        sim.add(m=5.97219e24, hash='earth')
        meteor = rebound.Particle(m=mass_point_var[j],x=X[j],y=Y[j],z=Z[j],vx=Vx[j],vy=Vy[j],vz=Vz[j])
        sim.add(meteor)
        
        ps = sim.particles


        # Calculate atmospheric density (RHO) based on current altitude and location
        RHO = calculate_rho(altM[j], latM[j], lonM[j])

        # Update particle position and velocity and get the time step
        #step = updateParticle(sim, RHO, CD, ASection, massaParticula, j)

        def drag(reb_sim):
            vRx = ps[1].vx - vWx
            vRy = ps[1].vy - vWy
            vRz = ps[1].vz - vWz
            vRMod = math.sqrt(vRx**2 + vRy**2 + vRz**2)
            ps[1].ax -= RHO*CD*ASection*(vRMod)*vRx/(2.*massaParticula)
            ps[1].ay -= RHO*CD*ASection*(vRMod)*vRy/(2.*massaParticula)
            ps[1].az -= RHO*CD*ASection*(vRMod)*vRz/(2.*massaParticula)
        sim.additional_forces = drag
        sim.force_is_velocity_dependent = 1
        
        # Integrate the simulation with the calculated time step
        sim.integrate(step)

        # Update time, particle position, velocity, and location
        time += step
        X[j], Y[j], Z[j], Vx[j], Vy[j], Vz[j] = ps[j].x, ps[j].y, ps[j].z, ps[j].vx, ps[j].vy, ps[j].vz
        latM[j], lonM[j], altM[j] = transproj_cart.transform(ps[j].x, ps[j].y, ps[j].z)

    # Save the final Cartesian data to a file
    with open((meteor_name_path + '/FinalCartesian.dat'), 'w') as f:
        f.write(strCart)
    fileCart.close()
    fileCoord.close()
    
################################################################################################################
# Save final cartesian
    # Calculate the time of entry into the atmosphere
    with open(meteor_name_path+'/FinalCartesian.dat','r') as f:
        entry = f.read()
    end_time_flight = entry.index(" ")
    time_of_flight = float(entry[0:end_time_flight])
    time_of_flight_num = timedelta(seconds=-time_of_flight)
    
    print("Flight time for", massInt, "kg =", time_of_flight_num)
    initialTime = (datetimeMeteor - time_of_flight_num).strftime("%Y-%m-%d %H:%M:%S")
    print("Entry time:", initialTime)