import bpy
from math import radians, degrees, pi, cos
from bpy.app import handlers
from datetime import datetime


def clear_handlers():
    frame_handlers = [getattr(handlers, name)
        for name in dir(handlers) if name.startswith("render_")]

    for  handler in frame_handlers:
        handler.clear()


def install_package(pname):
    """
    pname = name of the package (string)
    pname = 'scipy'
    """
    import subprocess
    import sys
    from pathlib import Path

    py_exec = str(sys.executable)
    # Get lib directory
    lib = Path(py_exec).parent.parent / "lib"
    print(lib)
    # Ensure pip is installed
    subprocess.call([py_exec, "-m", "ensurepip", "--user" ])
    # Update pip (not mandatory)
    subprocess.call([py_exec, "-m", "pip", "install", "--upgrade", "pip" ])
    # Install packages
    subprocess.call([py_exec,"-m", "pip", "install", "--upgrade", f"--target={str(lib)}", pname])


#https://www.youtube.com/watch?v=rIhXHSdMWmc&ab_channel=CGPython
#https://github.com/CGArtPython/blender_plus_python/blob/main/in_or_out/in_or_out_done.py
def apply_glare_composite_effect():
    bpy.context.scene.use_nodes = True

    render_layer_node = bpy.context.scene.node_tree.nodes.get("Render Layers")
    comp_node = bpy.context.scene.node_tree.nodes.get("Composite")

    # remove node_glare from the previous run
    old_node_glare = bpy.context.scene.node_tree.nodes.get("Glare")
    if old_node_glare:
        bpy.context.scene.node_tree.nodes.remove(old_node_glare)

    # create Glare node
    node_glare = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeGlare")
#    node_glare.size = 7
#    node_glare.glare_type = "FOG_GLOW"
#    node_glare.quality = "HIGH"
#    node_glare.threshold = 0.2

    # create links
    bpy.context.scene.node_tree.links.new(render_layer_node.outputs["Image"], node_glare.inputs["Image"])
    bpy.context.scene.node_tree.links.new(node_glare.outputs["Image"], comp_node.inputs["Image"])


def clear_scene(fps=24):
    for coll in bpy.data.collections:
        if coll:
            obs = [o for o in coll.objects]
            while obs:
                bpy.data.objects.remove(obs.pop())

        bpy.data.collections.remove(coll)

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
        
#    remove materials
#    for material in bpy.data.materials:
#        print(i)
#        material.user_clear()
#        bpy.data.materials.remove(material)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    bpy.context.scene.render.use_file_extension = False
    bpy.context.scene.render.filepath = r"{{rendered_videofile_name}}"  #os.path.join("{{conf_res['temp_dir_name']}}", "rendered_file.mp4") #"{{output_dir}}" + '/rendered_file'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'LOWEST'
    bpy.context.scene.render.resolution_x = int({{ conf_res['frame_res'][0] }})
    bpy.context.scene.render.resolution_y = int({{ conf_res['frame_res'][1] }})
#    640×360   426x240
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.fps=fps

    bpy.context.scene.use_gravity = False

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.preview_adaptive_threshold = 0.9
    bpy.context.scene.cycles.adaptive_threshold = 0.9
    bpy.context.scene.cycles.samples = 1024

    bpy.context.scene.cycles.use_denoising = False

#    USE Nodes
#    bpy.context.scene.use_nodes = True
#    nodes = bpy.context.scene.node_tree.nodes
#    nodes.clear()

#    node_render = nodes.new(type="CompositorNodeRLayers")
#    node_glare = nodes.new(type="CompositorNodeGlare")
#    node_comp = nodes.new(type="CompositorNodeComposite")

    apply_glare_composite_effect()

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

#        bpy.context.scene.eevee.use_gtao = True
#        bpy.context.scene.eevee.use_bloom = True


#        SHADING checkboxes (not working ....do manualy)
    for s in bpy.data.screens:
        for a in s.areas:
            if a.type == 'VIEW_3D':
                a.spaces[0].shading.use_scene_lights_render = True
                a.spaces[0].shading.use_scene_world_render = True



def calc_wp(T, ecc, incl):
    w = (2*pi) / (24/T*60*60)
    print(f"w={w}")
    GM = 3.986E+14
    a = 6_768_601
    Re = 6_378_137
    J2 = 1.0826E-03
    
    res = -1.25 * (Re*Re)/(a*(1-ecc*ecc)**2)**2  
    res = res * J2 * w * cos(radians(incl))
    return res
    


if __name__ == "__main__":
#    install_package(pname="skyfield")  
    from skyfield.api import EarthSatellite, load, wgs84, Star
    from skyfield.framelib import itrs      
    
    eph = load('de421.bsp')
    earth, sun = eph['earth'], eph['sun']
    
    FPS = {{ conf_res['fps'] }}
    
    clear_scene(fps=FPS)
    show_earth_map = False
    
    my_scale = 1000
    
    
    #######  Add EARTH   ###########
    bpy.ops.mesh.primitive_uv_sphere_add(radius=6371/my_scale) # Earth.
    earth_obj = bpy.context.active_object
    earth_obj.name = "Earth"
    mod_subsurf = earth_obj.modifiers.new("mu_modif", "SUBSURF")
    mod_subsurf.levels=5
    
    if show_earth_map:
        #    World image MAP
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.location = (-400,400)
        texImage.image = bpy.data.images.load("D:\\FTP\\zvity\\DB_914\\Model_Blender\\earth_color_10K.tif")
        
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

        if earth_obj.data.materials:
            # assign to 1st material slot
            earth_obj.data.materials[0] = mat
        else:
            # no slots
            earth_obj.data.materials.append(mat)

    
    ################ADD SAT container
    #    empty cube
    bpy.ops.object.empty_add(type='CUBE', align='WORLD', 
                            location=(0, 0, 0), 
                            scale=(1,1,1),
                            radius = 0.2
                                )
    empty_cube_obj = bpy.context.active_object
#    empty_cube_obj.parent = earth_obj
    empty_cube_obj.name = "sat_conteiner"

    constraint = empty_cube_obj.constraints.new(type='TRACK_TO')
    constraint.target=earth_obj
    
    # Precession conteiner
    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', 
                            location=(0, 0, 0), 
                            scale=(1,1,1),
                            radius = 0.2
                                )
    empty_prec_obj = bpy.context.active_object
    empty_prec_obj.name = "sat_prec_conteiner"
    empty_prec_obj.parent = empty_cube_obj
    
    ############ SATELLITE ###
    # path to the blend
    filepath = "{{conf_res['sat_model_path']}}"
#    filepath = "D:\FTP\zvity_old\DB_914\Model_Blender\Topex_pass\sat_Topex_model\Topex_size_princ_bsdf.blend"
    # filepath = "\\\\LKD-SERVER\server_share\zvity\DB\ZVIT-DB_914-2022\Model_Blender\sat_Topex_model\Topex_size_princ_bsdf.blend"

#    # name of collection(s) to append or link
#    coll_name = "Satellite_col"

#    # append, set to true to keep the link to the original file
    link = False

#    # link all collections starting with 'MyCollection'
#    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
#        data_to.collections = [c for c in data_from.collections if c.startswith(coll_name)]

#    # link collection to scene collection
#    for coll in data_to.collections:
#        if coll is not None:
#           bpy.context.scene.collection.children.link(coll)
    
#    obj_name = "Topex-Posidon-composite"  #"Cylinder"
    obj_name = "Satellite"  #"Cylinder"
    # link all objects starting with 'Cube'
    with bpy.data.libraries.load(filepath, link=link) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name.startswith(obj_name)]

    #link object to current scene
    for obj in data_to.objects:
        if obj is not None:
           #bpy.context.scene.objects.link(obj) # Blender 2.7x
           bpy.context.collection.objects.link(obj) # Blender 2.8x
        
    
#    for m in bpy.data.materials:
#        print(m)
#        if m is "SolarFaces"
    
#    bpy.ops.outliner.show_one_level()
#    bpy.ops.outliner.show_one_level(open=False)

#    bpy.data.materials["GPMcore-SolarFaces1"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces2"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces3"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces4"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces5"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces6"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces7"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3
#    bpy.data.materials["GPMcore-SolarFaces8"].node_tree.nodes["Glossy BSDF"].inputs[1].default_value = 0.3


#    sp_value = 0.01
#    specular_value = 0.9
#    metallic = 0.1
#    bpy.data.materials["GPMcore-SolarFaces1"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces1"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces1"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#        
#    bpy.data.materials["GPMcore-SolarFaces2"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces2"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces2"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#    
#    bpy.data.materials["GPMcore-SolarFaces3"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces3"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces3"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#    
#    bpy.data.materials["GPMcore-SolarFaces4"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces4"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces4"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value

#    
#    bpy.data.materials["GPMcore-SolarFaces5"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces5"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces5"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#    
#    bpy.data.materials["GPMcore-SolarFaces6"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces6"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces6"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#    
#    bpy.data.materials["GPMcore-SolarFaces7"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces7"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces7"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value
#    
#    bpy.data.materials["GPMcore-SolarFaces8"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces8"].node_tree.nodes["Principled BSDF"].inputs[6].default_value = metallic
#    bpy.data.materials["GPMcore-SolarFaces8"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = specular_value

#    sp_value = 0.684
#    
#    bpy.data.materials["GPMcore-SolarFaces1"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces2"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces3"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces4"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces5"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces6"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces7"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value
#    bpy.data.materials["GPMcore-SolarFaces8"].node_tree.nodes["Specular BSDF"].inputs[2].default_value = sp_value

#    bpy.data.objects["Cylinder"].parent = empty_cube_obj
    
                        
    sat_debug_scale = 1   # 100 or 1    Bigger to see something
#    bpy.ops.mesh.primitive_cylinder_add(vertices=128, 
#                            depth=(7.4/my_scale)*sat_debug_scale, 
#                            radius=(1.2/my_scale)*sat_debug_scale
#                            ) # depth - lenght HUUUGE !!!!
#    sat_obj = bpy.context.active_object
    sat_obj = bpy.data.objects["Satellite"] #["Cylinder"]
    sat_obj.name = "Cosmos2502"
    sat_obj.scale = sat_obj.scale / my_scale * sat_debug_scale
#    sat_obj.parent = empty_cube_obj

    sat_obj.location =(0, 0, 0)
    sat_obj.parent = empty_prec_obj
    
#    sat_sp = bpy.data.objects["Topex-Posidon-sp"] #["Cylinder"]
#    sat_sp.name = "Satellite_SP"
#    sat_sp.scale = sat_sp.scale / my_scale * sat_debug_scale
#    sat_sp.parent = empty_prec_obj
    
#    ARROWS
    bpy.ops.object.empty_add(type='ARROWS', align='WORLD', 
                            location=(0, 0, 0), 
#                            scale=((10/my_scale)*sat_debug_scale*5, 
#                                (10/my_scale)*sat_debug_scale*5, 
#                                (10/my_scale)*sat_debug_scale*5)
#                            scale = (5,5,5)
                                )
    
    arrows_obj = bpy.context.active_object
    arrows_obj.parent = sat_obj
    arrows_obj.name = "arrows"
    arrows_obj.hide_render = True
#    arrows_obj.scale = ((10/my_scale)*sat_debug_scale*5, 
#                        (10/my_scale)*sat_debug_scale*5, 
#                        (10/my_scale)*sat_debug_scale*5
#                        )
    arrows_obj.scale = (10,10,10)
    arrows_obj.scale = arrows_obj.scale / my_scale * sat_debug_scale
    if sat_debug_scale != 1:
        arrows_obj.scale = arrows_obj.scale * 200

    
    ts = load.timescale()

    start_date = "{{conf_res['lc_start_date']}}"
    start_time = "{{conf_res['lc_start_time']}}"
    start_dt= datetime.strptime(start_date + " " + start_time, '%Y-%m-%d %H:%M:%S')

    sec = list(range(start_dt.second, start_dt.second + {{conf_res["lc_duration"]}} + 1, 1))
    times = ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, sec)

    # print(times)


#    Orbit visualisation
#    tmo = ts.utc(2021, 8, 10, 20, 7, list(range(25-2200-1, 25 + 2200 + 1, 30)))
    tmo = ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, list(range(25-2200-1, 25 + 2200 + 1, 30)))


    line1 = "{{conf_res['tle_line1']}}" #'1 22076U 92052A   21160.57864517 -.00000059  00000-0  95149-5 0  9996'
    line2 = "{{conf_res['tle_line2']}}" # '2 22076  66.0409 341.5772 0007830 272.3340 147.4137 12.81032194348634'
    line1 = line1.strip()
    line2 = line2.strip()
    # print(line1)
    # print(line2)
    satellite = EarthSatellite(line1, line2, "{{conf_res['sat_name']}}", ts)


    myCol = bpy.data.collections.new("Orbit")
    bpy.context.scene.collection.children.link(myCol)
    for t in tmo:
        geocentric = satellite.at(t)
#        sat_obj.location = geocentric.frame_xyz(itrs).km / my_scale

        bpy.ops.object.empty_add(type='CIRCLE', radius=0.01, 
                                            align='WORLD', 
                                            location=geocentric.frame_xyz(itrs).km / my_scale, 
                                            rotation=(0, 0, 0), 
                                            scale=(1, 1, 1))
    
        obj = bpy.context.active_object
        # Remove object from all collections not used in a scene
        bpy.ops.collection.objects_remove_all()
        # add it to our specific collection
        bpy.data.collections['Orbit'].objects.link(obj)
    
    
    
    incl = line2.split()[2]
    incl = float(incl)
    print(incl, 90-incl)
    sat_obj.rotation_euler[2] = -1 * radians(90-incl)  #radians(90-incl)
#    sat_obj.delta_rotation_euler[1] = radians(90-incl)


    


    bpy.context.scene.frame_end = {{ conf_res["lc_duration"] }} * FPS

    
##    Satellite Materials
#    sat_mat = bpy.data.materials.new(name="Material_satellite")
#    sat_mat.use_nodes = True

#    emmNode = sat_mat.node_tree.nodes.new(type="ShaderNodeBsdfAnisotropic")

#    material_out = sat_mat.node_tree.nodes.get('Material Output')
#    sat_mat.node_tree.nodes.remove(sat_mat.node_tree.nodes.get('Principled BSDF')) #title of the existing node when materials.new

#    sat_mat.node_tree.links.new(emmNode.outputs['BSDF'], material_out.inputs['Surface'])

#    
#    if sat_obj.data.materials:
#        # assign to 1st material slot
#        sat_obj.data.materials[0] = sat_mat
#    else:
#        # no slots
#        sat_obj.data.materials.append(sat_mat)

##############################
    
    #####
    # Create Sun (Light)
    light_data = bpy.data.lights.new('light', type='SUN')
    light_data.specular_factor = 5
    light_data.energy = 1.38
    light = bpy.data.objects.new('light', light_data)
    bpy.context.collection.objects.link(light)
    light.data.use_contact_shadow = True

    t = times[0]
#    print(t)
    sun_position = earth.at(t).observe(sun)
    sun_position = sun_position.frame_xyz(itrs)
    light.location = sun_position.km / my_scale #/ 1000
#    print(sun_position.m)
    
    constraint_sun = light.constraints.new(type='TRACK_TO')
    constraint_sun.target=earth_obj


    #### CREATE OBSERVER
    topos = wgs84.latlon(48.633, 22.33, elevation_m=180)
#    print(topos.at(t).position.km )
    uzh_pos = topos.itrs_xyz.km / my_scale
    
#    print(uzh_pos*1000)

    
    ##### Create CAMERA
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    bpy.context.collection.objects.link(cam)
    # add camera to scene
    scene = bpy.context.scene
    scene.camera=cam
#    cam.location=(25, -3, 20)
    cam.location=uzh_pos

    if sat_debug_scale == 1:
        cam.data.lens = 1700
    else:
        cam.data.lens=70

    cam.data.clip_end = 1e+06
#    cam.data.clip_start = 0.001


    
    # Satellite tracking
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.target=empty_cube_obj
#    constraint.target=sat_obj

    # sat_period = {{ [var["value"] for var in {{var_list}} if var['name'] == 'p_spin'][0] }}
    # p_phase = {{ [var["value"] for var in {{var_list}} if var['name'] == 'p_phase'][0] }}
    # pr_angle = {{ [var["value"] for var in {{var_list}} if var['name'] == 'pr_angle'][0] }}
    # pr_period = {{ [var["value"] for var in {{var_list}} if var['name'] == 'pr_period'][0] }}
    # pr_phase = {{ [var["value"] for var in {{var_list}} if var['name'] == 'pr_phase'][0] }}
    

#    Satellite period of self rotation


#    Initial Satellite angle in rotation (Phase)
#    sat_obj.rotation_euler[0] += radians(180) #110
#    sat_obj.rotation_euler[1] += radians(0) #110
    sat_obj.rotation_euler[0] = radians( {{p_phase}} )


#    Precession Angle and precession Period
#    sat_obj.delta_rotation_euler[1] = radians(0) # Y
    # empty_prec_obj.rotation_euler[0] = radians(2.5) #radians(-45)  #24.4, axis = 0
    empty_prec_obj.rotation_euler[0] = radians(pr_angle) #330 


    empty_prec_obj.rotation_euler[2] = radians(pr_phase) # Initial precession phase


    # 1620  #173.3 * 24 * 3600 
    
    
#    sat_sp.rotation_euler[0] = radians(-12)


#    bpy.context.scene.eevee.use_bloom = True
#    bpy.context.scene.eevee.use_gtao = True


#    Initial Satellite precession angle in rotation (Phase)
#    sat_obj.delta_rotation_euler[0] = radians(0)
    
#----------------------------------------------------------------

#    sat_obj.rotation_euler[2]=radians(-90)
    
#    precession
#    sat_obj.delta_rotation_euler[1] = radians(20)
#    sat_obj.rotation_euler[1] = radians(20)




#    Add Text
#https://blender.stackexchange.com/questions/163487/how-to-add-text-in-blender-using-python
#    bpy.data.curves.new(type="FONT", name="Font Curve").body = "my text"
#    text_obj = bpy.data.objects.new(name="Font Object", object_data=bpy.data.curves["Font Curve"])
#    bpy.context.scene.collection.objects.link(text_obj)
#    text_obj.parent = cam
#    text_obj.location[0] = 0.2
#    text_obj.location[1] = -1
#    text_obj.location[2] = -0.35
#    text_obj.data.size = 0.03

    

    
    frame_num = 0
    for t in times:
#        print(t.utc)
        geocentric = satellite.at(t)
#        print(geocentric.frame_xyz(itrs).m) 
        
        bpy.context.scene.frame_set(frame_num)

#        sat.location = geocentric.position.km / my_scale        
#        sat_obj.location = geocentric.frame_xyz(itrs).km / my_scale
        empty_cube_obj.location = geocentric.frame_xyz(itrs).km / my_scale
        empty_cube_obj.keyframe_insert(data_path="location", index=-1)
        
        # FIXED FoV
        if frame_num == 0:
            cam_f = cam.data.lens
            sat_dist = (empty_cube_obj.location - cam.location).length
        else:
            cam_new_f = cam_f * (
                (empty_cube_obj.location - cam.location).length/sat_dist
            )
#            print(cam_new_f,  (empty_cube_obj.location - cam.location).length)
            bpy.context.scene.camera.data.lens = cam_new_f
            bpy.context.scene.camera.data.keyframe_insert("lens")
        
#        Rotation of Satellite
#        sat.rotation_euler[0] += radians(360/60)  #360 degrees / 30 sec
#        print(sat.rotation_euler[0])
#        bpy.data.objects["Cylinder"].select_set(True)

#        sat.select_set(True) 
#        bpy.ops.transform.rotate(value=radians(360/sat_period), orient_axis="X", orient_type='LOCAL')
        
        if frame_num > 0 and sat_period > 0:
            sat_obj.rotation_euler[0] -= radians(360/sat_period) #sat_axis_rot 
            sat_obj.keyframe_insert(data_path="rotation_euler", index=0)
        
#        bpy.context.scene.render.stamp_note_text = f"sat_rot =" + "%2.2f" % sat_obj.rotation_euler[0]
         
            
#        Precession
#        sat_obj.delta_rotation_euler[0] += radians(360/prec_per) # Z
#        sat_obj.rotation_euler[2] += radians(360/prec_per) # Z

        if frame_num > 0 and pr_period > 0:
            empty_prec_obj.rotation_euler[2] += radians(360/pr_period) #!!!!!!!!!!!!!!!!!!!!
            empty_prec_obj.keyframe_insert(data_path="rotation_euler", index=2)
        

#        sat_obj.keyframe_insert(data_path="rotation_euler", index=2)
        
#        sat_obj.keyframe_insert(data_path="delta_rotation_euler", index=1)
#        sat_obj.keyframe_insert(data_path="delta_rotation_euler", index=0)
        frame_num += FPS
        



#    # Iterate over all the objects animation function curves
    for fc in sat_obj.animation_data.action.fcurves:
        fc.extrapolation = 'LINEAR' # Set extrapolation type
#        

    print("finish")
