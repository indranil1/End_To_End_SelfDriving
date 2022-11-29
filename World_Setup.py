import carla
import os
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pycocotools
import math
from scipy.spatial import distance
import queue
from PIL import Image
from PIL import _imaging
import cv2
import time
import math
from mpmath import cot, acot
print("Entering the world creation module!")

# Client creation
client = carla.Client('localhost', 2000)


towns=['Town07']

def get_turning_angle(degree1,degree2):
    if abs((float(degree1)+float(degree2))/2.0) <=0.1:
        return(0)
    print("degree1=", degree1)
    print("degree2=", degree2)
    # degree1=degree1*math.pi/180
    # degree2=degree2*math.pi/180
    degree=(((degree1)+(degree2))/2)
    # print(degree)
    return(degree)
    
def folder_classifier(degree):
    classification_step=0.25
    folder=int(degree/classification_step)
    # print(folder)
    return folder




for town in towns:
    world = client.load_world(town)
    # client.set_timeout(10.0) # seconds
    steer_angle_list=[]
    # World Creation
    image_value_pair={}
    print(client.get_available_maps())
    while True:
        #world = client.load_world('Town01')
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=30.0,
            sun_altitude_angle=70.0)
        world.set_weather(weather)
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05 #must be less than 0.1, or else physics will be noisy
        #must use fixed delta seconds and synchronous mode for python api controlled sim, or else 
        #camera and sensor data may not match simulation properly and will be noisy 
        settings.synchronous_mode = True 
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library().filter('*')
        #for blueprint in random.sample(list(blueprints), 5):
        #    print(blueprint.id)vehicle'
        #    for attr in blueprint:
        #    print('  - {}'.format(attr))***
        actor_list = []
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('mkz_2017')) # lets choose a vehicle at random
    
        # lets choose a random spawn point
        transform = random.choice(world.get_map().get_spawn_points()) 
    
        #spawn a vehicle
        while True:
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle!=None:
                break
            
        actor_list.append(vehicle)
        break
    vehicle.set_autopilot(True)
    
    # Adding random objects
    blueprint_library = world.get_blueprint_library()
    weirdobj_bp = blueprint_library.find('static.prop.fountain')
    weirdobj_transform = random.choice(world.get_map().get_spawn_points())
    
    # for getting camera image
    weirdobj_transform = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))
    weird_obj = world.try_spawn_actor(weirdobj_bp, weirdobj_transform)
    actor_list.append(weird_obj)
    
    # example for getting camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    actor_list.append(camera)
    
    #example for getting depth camera image
    camera_depth = blueprint_library.find('sensor.camera.depth')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_d = world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
    image_queue_depth = queue.Queue()
    camera_d.listen(image_queue_depth.put)
    actor_list.append(camera_d)
    
    #example for getting semantic segmentation camera image
    camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_seg = world.spawn_actor(camera_semseg, camera_transform, attach_to=vehicle)
    image_queue_seg = queue.Queue()
    camera_seg.listen(image_queue_seg.put)
    actor_list.append(camera_seg)
    
    world.tick()
    
    #rgb camera
    image = image_queue.get()
    
    #semantic segmentation camera
    image_seg  = image_queue_seg.get()
    
    #depth camera
    image_depth = image_queue_depth.get()
    
    
    # image_seg.save_to_disk("test_images/%06d_semseg.png" %(image.frame), carla.ColorConverter.CityScapesPalette)
    # image_depth.save_to_disk("test_images/%06d_depth.png" %(image.frame), carla.ColorConverter.LogarithmicDepth)
    
    
    m=world.get_map()
    start_pose = random.choice(m.get_spawn_points())
    waypoint = m.get_waypoint(start_pose.location)
    waypoint = random.choice(waypoint.next(1.5))
    vehicle.set_transform(waypoint.transform)
    
    
    # path="test_images"+town+"/"+str(image.frame)+".png"
    # image.save_to_disk(path)
    # path="test_images"+town+"/"+str(image.frame)+"_semseg.png"
    # image_seg.save_to_disk(path, carla.ColorConverter.CityScapesPalette)
    # path="test_images"+town+"/"+str(image.frame)+"_depth.png"
    # image.save_to_disk(path, carla.ColorConverter.LogarithmicDepth)
    
    
    # from detectron2.structures import BoxMode
    # #in sychronous mode, client controls step of simulation and number of steps
    dataset_dicts = []
    global_count=0
    for i in range(10000):
        #step
        world.tick()
    
        #rgb camera
        image = image_queue.get()
    
        #semantic segmentation camera
        image_seg  = image_queue_seg.get()
        #image_seg.convert(carla.ColorConverter.CityScapesPalette)
    
        #depth camera
        image_depth = image_queue_depth.get()
        #image_dnsepth.convert(carla.ColorConverter.LogarithmicDepth)
        
       
    
        velocity=None
        velocity=vehicle.get_velocity()
        velocity=(18/5)*math.sqrt((velocity.x**2)+(velocity.y**2)+(velocity.z**2))
        
        if velocity==None:
            print("Exception: Vehicle not found")
            continue
            
        if i%10==0:
            # image.save_to_disk("test_images/%06d.png" %(image.frame))
            # image_seg.save_to_disk("test_images/%06d_semseg.png" %(image.frame), carla.ColorConverter.CityScapesPalette)
            # #image_depth.save_to_disk("test_images/%06d_depth.png" %(image.frame), carla.ColorConverter.LogarithmicDepth)
    
            # img = mpimg.imread("test_images/%06d.png" % image.frame)
            # img_semseg = mpimg.imread("test_images/%06d_semseg.png" % image.frame)
            # #img_depth = mpimg.imread("test_images/%06d_depth.png" % image.frame)
            
            degree_turn=get_turning_angle(vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel), vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel))
            folder=folder_classifier(degree_turn)
            if folder==0:
                pass
            else:
                folder=(folder*.25)+(np.sign(folder)*.125)
            path="test_images/"+str(folder)+"/"+str(image.frame)+town+".png"
            image.save_to_disk(path)
            # i = np.array(image.raw_data)
            # #np.save("iout.npy", i)
            # cv2.imshow("",i)
            # cv2.waitKey(1)
            # path="test_images/"+str(folder)+"/"+str(image.frame)+"_semseg.png"
            # image_seg.save_to_disk(path, carla.ColorConverter.CityScapesPalette)
            path="test_images/"+str(folder)+"/"+str(image.frame)+town+"_depth.png"
            image.save_to_disk(path, carla.ColorConverter.LogarithmicDepth)
            
            
            
            ## COCO format stuff, each image needs to have these keys
            
            image_value_pair[str(image.frame)+".png"]=[[vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)],velocity]
            
            
            if abs(degree_turn)<2:
                vehicle.set_transform(waypoint.transform) 
                
            waypoint = random.choice(waypoint.next(1.5))    
             
            
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    
    with open(town+'image_angle_vel.txt', 'w') as f:
        for image in image_value_pair.keys():
            f.write(image+"\t"+str(image_value_pair[image])+"\n")
            