# Scene Graph Simulator
# Evaluate by compare to AI2THOR environment graphs

"""
Scene Graph
Design

Node
- Static / Dynamic
- Named / Unamed
- Area Class (Kitchen, Bedroom, etc)
- Affordance (Go, Toggle, etc)

Edge
- Relation

"""

from scene_graph import SceneGraph
import numpy as np

class SceneSim:
	def __init__(self):
		pass

	def step(self, scene_graph=None, )

if __name__ == "__main__":
	# import ai2thor
	# print(ai2thor.__version__)
	# controller = ai2thor.controller.Controller('Very Low', False, True)
	# controller.start()
	# controller.step(dict(action='Initialize', grid_size=0.25, headless=True, visibilityDistance=0.75))
	# event = controller.step(dict(action='Pass'))
	# controller.reset('FloorPlan25')
	# controller.reset(scene_name)
	# env = ThorEnvironment(controller=controller)
	# env.graph = env.graph.from_pickle(environment_file)

	# Load each of the stored SceneGraphs
	import glob
	scene_graph_files = glob.glob("D:/Datasets/saved_environments_floorplan25/saved_environments_floorplan25/*.gz")
	print(scene_graph_files)