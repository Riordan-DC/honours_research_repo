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

# SceneSim functions

def ss_step(idx: int, scene_graph: SceneGraph):
	"""
	Performs action defined by affordance node at index (idx) in the scene graph (scene_graph)
	"""
	pass

def ss_ai2thor_explore(controller):
	"""
	Explores an AI2THOR environment and returns a SceneGraph
	"""
	pass

def ss_set_node(scene_graph, node):
	"""
	Sets a node in the graph. If it exists, it updates the node, if not it adds a new node.
	"""
	pass


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
	
	for sg in scene_graph_files:
		scene_graph = SceneGraph()
		scene_graph = scene_graph.from_pickle(sg)