import bpy
from dataclasses import dataclass
from typing import ClassVar
import os
import random
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path
import json
import math
from abc import ABC, abstractmethod
import dataformat as df


class NoiseFunction(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass


class Uniform(NoiseFunction):
    def __init__(self, min: float = 0.0, max: float = 1.0):
        self.min = min
        self.max = max

    def __call__(self) -> float:
        return random.uniform(self.min, self.max)


class NoNoise(NoiseFunction):
    def __call__(self) -> float:
        return 0.0


@dataclass
class Button:
    BUTTON_PREFIX: ClassVar = "button_"
    button: object
    initial_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def scale(self):
        return self.button.scale.copy()
    
    @scale.setter
    def scale(self, scale):
        self.button.scale = scale

    @property
    def name(self):
        return self.button.name

    def reset(self):
        self.button.scale = self.initial_scale.copy()

    def __post_init__(self):
        self.initial_scale = self.scale


@dataclass
class ButtonDistraction(Button):
    BUTTON_PREFIX: ClassVar = "distraction_"


@dataclass
class Velcro:
    VELCRO_PREFIX: ClassVar = "velcro_"
    velcro: object
    initial_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def scale(self):
        return self.velcro.scale.copy()
    
    @scale.setter
    def scale(self, scale):
        self.velcro.scale = scale

    def reset(self):
        self.velcro.scale = self.initial_scale.copy()

    @property
    def name(self):
        return self.velcro.name
    
    def __post_init__(self):
        self.initial_scale = self.scale


@dataclass
class KeyPoint:
    KEYPOINT_PREFIX: ClassVar = "kp_"
    keypoint: object

    @property
    def name(self):
        return self.keypoint.name


@dataclass
class Background:
    BACKGROUND_PREFIX: ClassVar = "background"
    background: object

    @property
    def name(self):
        return self.background.name


@dataclass
class Scene:
    name: str
    scene: object
    camera: object
    light: object
    cloth: object
    buttons: list[Button]
    distractions: list[ButtonDistraction]
    velcros: list[Velcro]
    key_points: list[KeyPoint]
    background: Background

    @classmethod
    def from_current_bpy_context(cls, name: str):
        scene = bpy.context.scene
        camera = bpy.data.objects.get("Camera")
        light = bpy.data.objects.get("Light")
        cloth = bpy.data.objects.get("Cloth")
        key_points = get_keypoints()
        buttons = get_buttons()
        distractions = get_distractions()
        velcros = get_velcros()
        background = get_background()
        return cls(name, scene, camera, light, cloth, buttons, distractions, velcros, key_points, background)
    
    def __post_init__(self):
        self._configure_render()
        self._enable_cycles_gpu(use_cpu=True)

    def set_object_texture(self, object, texture):
        if texture == "<random color>":
            random_color = (random.random(), random.random(), random.random(), 1.0)
            self.set_uniform_color(object, random_color)
        elif isinstance(texture, str) and os.path.isfile(bpy.path.abspath(texture)):
            self.set_image_texture(object, texture)
        elif isinstance(texture, str):
            self.set_random_image_texture(object, texture)
        elif isinstance(texture, (tuple, list)):
            self.set_uniform_color(object, texture)

    def set_image_texture(self, object, image_path: str):
        """Set an image texture to the given object."""
        material_name = f"ImageTextureMaterial[{os.path.basename(image_path)}]"
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        material_bsdf = material.node_tree.nodes['Principled BSDF']
        texture_node = material.node_tree.nodes.new('ShaderNodeTexImage')
        texture_node.image = bpy.data.images.load(bpy.path.abspath(image_path))
        for l in list(material_bsdf.inputs["Base Color"].links):
            material.node_tree.links.remove(l)
        material.node_tree.links.new(texture_node.outputs['Color'], material_bsdf.inputs['Base Color'])
        object.active_material = material

    def set_random_image_texture(self, object, texture_folder) -> None:
        """Set a random image texture from the given folder to the given object."""
        folder = Path(texture_folder)
        image_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if image_files:
            image_path = random.choice(image_files)
            self.set_image_texture(object, str(image_path))

    def set_builtin_texture(self, object, texture_name: str):
        """Set a basic texture (e.g. Checker, Noise) to the given object."""
        material_name = f"BasicTextureMaterial[{texture_name}]"
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        material_bsdf = material.node_tree.nodes['Principled BSDF']
        texture_node = material.node_tree.nodes.new(f"ShaderNodeTex{texture_name}")
        for l in list(material_bsdf.inputs["Base Color"].links):
            material.node_tree.links.remove(l)
        material.node_tree.links.new(texture_node.outputs['Color'], material_bsdf.inputs['Base Color'])
        object.active_material = material

    def set_uniform_color(self, object, color: tuple | list):
        """Set a uniform color material to the given object."""
        material_name = f"UniformColorMaterial[{color[0]:.2f},{color[1]:.2f},{color[2]:.2f}]"
        colored_material = bpy.data.materials.new(name=material_name)
        colored_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
        object.active_material = colored_material

    def set_light(self, type, energy):
        """Configure the light with the given type and energy."""
        self.light.data.type = type
        self.light.data.energy = energy

    def orbit_around_object_with_roll(
            self, 
            orbiter, target, 
            radius, 
            h_angle, v_angle, 
            roll_angle, 
            noise_x: NoiseFunction = NoNoise(), 
            noise_y: NoiseFunction = NoNoise(), 
            noise_z: NoiseFunction = NoNoise()
        ):
        tgt = target.matrix_world.translation

        az = math.radians(h_angle)      # azimuth
        el = math.radians(v_angle)      # elevation
        roll = math.radians(roll_angle) # roll about view axis

        # 1) Position on sphere around target (world Z-up)
        pos = Vector((
            tgt.x + noise_x() + radius * math.cos(el) * math.cos(az),
            tgt.y + noise_y() + radius * math.cos(el) * math.sin(az),
            tgt.z + noise_z() + radius * math.sin(el),
        ))
        orbiter.location = pos

        # 2) Look-at: make camera local -Z point toward target, local +Y as "up"
        forward = (tgt - pos).normalized()
        look_q = forward.to_track_quat('-Z', 'Y')

        # 3) View axis in WORLD space (camera local -Z transformed by look_q)
        view_axis_world = look_q @ Vector((0.0, 0.0, -1.0))

        # 4) Roll around that viewing axis
        roll_q = Matrix.Rotation(roll, 4, view_axis_world).to_quaternion()

        # 5) Compose: first look-at, then roll around the *current* view axis
        orbiter.rotation_mode = 'QUATERNION'
        orbiter.rotation_quaternion = roll_q @ look_q

    def move_randomly_object(self, target, x_noise_func, y_noise_func, z_noise_func):
        """
        Move randomly an object.
        """
        location = target.location
        location.x += x_noise_func()
        location.y += y_noise_func()
        location.z += z_noise_func()

    def tilt_randomly_camera(self, noise_func):
        roll = math.radians(noise_func())
        self.camera.matrix_world = self.camera.matrix_world @ Matrix.Rotation(roll, 4, 'Z')

    def set_cloth_mass(self, vertex_mass: float):
        self.cloth.modifiers["Cloth"].settings.mass = vertex_mass

    def set_parameters(self, params):
        button_color = params.get("button_color").value
        cloth_color = params.get("cloth_color").value
        background_color = params.get("background_color").value
        velcro_color = params.get("velcro_color").value
        camera_distance = params.get("camera_distance").value
        camera_h_angle = params.get("camera_h_angle").value
        camera_v_angle = params.get("camera_v_angle").value
        light_h_angle = params.get("light_h_angle").value
        light_v_angle = params.get("light_v_angle").value
        light_distance = params.get("light_distance").value
        brightness = params.get("brightness").value
        button_scale = params.get("button_scale").value
        vertex_mass = params.get("vertex_mass").value
        # set the parameters
        self.set_object_texture(self.cloth, cloth_color)
        for button in self.buttons:
            button.reset() # reset the scale property
            self.set_object_texture(button.button, button_color)
            button.scale = button.scale * button_scale
        for velcro in self.velcros:
            velcro.reset()
            self.set_object_texture(velcro.velcro, velcro_color)
            velcro.scale = velcro.scale * button_scale
        for distraction in self.distractions:
            distraction.reset()
            self.set_object_texture(distraction.button, button_color)
            distraction.scale = distraction.scale * button_scale
        # set cloth simulation mass
        self.set_cloth_mass(vertex_mass)
        # change the light
        self.set_light(type='POINT', energy=brightness)
        self.orbit_around_object_with_roll(self.light, self.cloth, radius=light_distance, h_angle=light_h_angle, v_angle=light_v_angle, roll_angle=0)
        # change the camera, and roll it randomly
        roll_angle = Uniform(-25.0, 25.0)()
        self.orbit_around_object_with_roll(
            self.camera, self.cloth,
            radius=camera_distance,
            h_angle=camera_h_angle,
            v_angle=camera_v_angle,
            roll_angle=roll_angle,
            noise_x=Uniform(-2, 2),
            noise_z=Uniform(-2.0, 2.0)
        )

    def _configure_render(self):
        self.scene.render.image_settings.file_format = "PNG"
        self.scene.render.image_settings.color_mode = "RGB"

    def _enable_cycles_gpu(self, prefer="OPTIX", use_cpu=False):
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.device = "GPU"
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences
        # Select backend
        cycles_prefs.compute_device_type = prefer  # "OPTIX" or "CUDA"
        # Refresh device list
        cycles_prefs.get_devices()
        # Disable everything first
        for d in cycles_prefs.devices:
            d.use = False
        # Enable GPUs (and optionally CPU)
        enabled_any_gpu = False
        for d in cycles_prefs.devices:
            if d.type in {"OPTIX", "CUDA"}:
                d.use = True
                enabled_any_gpu = True
            elif d.type == "CPU":
                d.use = bool(use_cpu)
        if not enabled_any_gpu and not use_cpu:
            raise RuntimeError("No GPU devices were enabled for Cycles. ")
        
    def _render_size(self):
        render = self.scene.render
        width = int(render.resolution_x * render.resolution_percentage / 100)
        height = int(render.resolution_y * render.resolution_percentage / 100)
        return width, height

    def project_world_point(self, world_point: Vector, anyway=False):
        width, height = self._render_size()

        ndc = world_to_camera_view(self.scene, self.camera, world_point)
        x_ndc = float(ndc.x)
        y_ndc = float(ndc.y)
        z = float(ndc.z)

        x_px = x_ndc * width
        y_px = (1.0 - y_ndc) * height  # convert bottom-left origin to top-left origin

        if (not anyway) and (z < 0 or x_ndc < 0 or x_ndc > 1 or y_ndc < 0 or y_ndc > 1):
            return None

        return {
            "x_px": x_px,
            "y_px": y_px,
            "x_ndc": x_ndc,
            "y_ndc": y_ndc,
            "z": z,
        }

    def project_object_center(self, target, anyway=False):
        world_center = bbox_center_world(target)
        return self.project_world_point(world_center, anyway=anyway)

    def project_object_bbox(self, target, clamp=True):
        width, height = self._render_size()

        corners_world = bbox_corners_world(target)
        projected = [world_to_camera_view(self.scene, self.camera, p) for p in corners_world]

        # If every corner is behind the camera, reject it
        if all(float(p.z) < 0 for p in projected):
            return None

        xs = [float(p.x) * width for p in projected]
        ys = [(1.0 - float(p.y)) * height for p in projected]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Reject if completely outside image
        if x_max < 0 or x_min > width or y_max < 0 or y_min > height:
            return None

        # Clamp to image borders if wanted
        if clamp:
            x_min = max(0.0, min(float(width), x_min))
            x_max = max(0.0, min(float(width), x_max))
            y_min = max(0.0, min(float(height), y_min))
            y_max = max(0.0, min(float(height), y_max))

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w <= 0 or bbox_h <= 0:
            return None

        # get the normalized coordinates
        cx = (0.5 * (x_min + x_max)) / width
        cy = (0.5 * (y_min + y_max)) / height
        w = bbox_w / width
        h = bbox_h / height

        return df.BoundingBox(
            cx=cx,
            cy=cy,
            w=w,
            h=h
        )


    def project_pairs(self):
        """
        Project a pair <button, velcro>.
        TODO: The model should ideally output a position for button and velcro if one of them is visible.
        For now, the visibility is not handled, so we assume the button is the main detection point.
        """
        pairs = []
        for button, velcro in zip(self.buttons, self.velcros):
            bbox_button = self.project_object_bbox(button.button)
            bbox_velcro = self.project_object_bbox(velcro.velcro)
            if bbox_button is None:
                continue
            button = df.Button(bbox_button, visible=True)
            velcro = df.Fastener(bbox_velcro, visible=True, type="velcro")
            pair = df.Pair(button, velcro)
            pairs.append(pair)
        return pairs



def bbox_corners_world(obj) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]



class Renderer:
    def __init__(self, scene: Scene, output_dir: Path, count = 0):
        self.scene = scene
        self.output_dir = output_dir
        self.rendered_count = count

    @property
    def current_output_filename(self):
        return f"{self.scene.name}_{self.rendered_count:08d}"
    
    def get_current_annotation(self):
        # get resolution of the image
        render = self.scene.scene.render
        width = int(render.resolution_x * render.resolution_percentage / 100)
        height = int(render.resolution_y * render.resolution_percentage / 100)
        image_info = df.ImageInfo(url="", width=width, height=height)
        # project the pairs
        projected_pairs = self.scene.project_pairs()
        cloth = df.Cloth(type="<simulation>", segmentation="", pairs=projected_pairs)
        # finally get the annotation
        annotation = df.Annotation(image_info, cloth)
        return annotation

    def render_at(self, parameters, frame_range: list[int]):
        # set the scene parameters
        self.scene.set_parameters(parameters)
        # randomly take frames
        target_frames = random.sample(frame_range, 1) # in fact, only one random frame
        target_frames = sorted(target_frames)
        scene = self.scene.scene
        for i in range(target_frames[-1] + 1):
            # set the frame and update the simulation
            scene.frame_set(i)
            bpy.context.view_layer.update()
            # if the frame is to be render, render it
            if i in target_frames:
                # get the filename
                filename = self.current_output_filename
                out_image_path = self.output_dir / "images" / f"{filename}.png"
                out_json_path = self.output_dir / "annotations" / f"{filename}.json"
                # render and write the image
                scene.render.filepath = str(out_image_path)
                bpy.ops.render.render(write_still=True)
                # project points
                annotation = self.get_current_annotation()
                annotation.image.url = out_image_path.name
                with open(out_json_path, "w") as file:
                    json.dump(annotation.to_json(), file, indent=2)
                # print
                print(f"==> frame '{i}' rendered in path '{filename}'")
                # go to next
                self.rendered_count += 1
                


def bbox_center_world(obj) -> Vector:
    # obj.bound_box is 8 corners in local space
    corners_world = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners_world, Vector((0,0,0))) / 8.0

def get_keypoints():
    kps = [KeyPoint(obj) for obj in bpy.data.objects if obj.name.startswith(KeyPoint.KEYPOINT_PREFIX)]
    kps.sort(key=lambda o: o.keypoint.name)
    return kps

def get_buttons():
    buttons = [Button(obj) for obj in bpy.data.objects if obj.name.startswith(Button.BUTTON_PREFIX)]
    buttons.sort(key=lambda o: o.button.name)
    return buttons

def get_velcros():
    velcros = [Velcro(obj) for obj in bpy.data.objects if obj.name.startswith(Velcro.VELCRO_PREFIX)]
    velcros.sort(key=lambda o: o.velcro.name)
    return velcros

def get_distractions():
    distractions = [ButtonDistraction(obj) for obj in bpy.data.objects if obj.name.startswith(ButtonDistraction.BUTTON_PREFIX)]
    return distractions

def get_background():
    return bpy.data.objects.get(Background.BACKGROUND_PREFIX)
