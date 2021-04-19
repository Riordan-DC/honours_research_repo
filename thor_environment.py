from scene_graph import SceneGraph
import numpy as np, termcolor


class Task:
    def __init__(self):
        pass

    def done(self, env):
        pass

    def reward(self, env):
        pass
# =============================================================================
# =============================================================================
# =============================================================================

class FindTask(Task):
    def __init__(self, target_object_type='Cup'):
        self.target_object_type = target_object_type
    # =========================================================================
    def reward(self, env):
        event = env.controller.step(action='Pass')
        for obj in [o for o in event.metadata['objects'] if o['visible']]:
            if obj['objectType'] == self.target_object_type:
                return 1
        return 0.0
    # =========================================================================
    def done(self, env):
        if self.reward(env)==1:
            return True
        else:
            return False
    # =========================================================================
    def is_possible(self, env):
        event = self.controller.step(action='Pass')
        targets = [obj for obj in event.metadata['objects'] if obj['objectType']==self.target_object_type]
        if len(targets) == 0:
            return False, targets
        else:
            return True, targets


# =============================================================================
# =============================================================================
# =============================================================================

class PickTask(Task):
    def __init__(self, possible_targets, max_timesteps=200, curious=True):
        self.possible_targets = possible_targets
        self.target_object_type = np.random.choice(self.possible_targets)
        self.max_timesteps = max_timesteps
        self.curious = curious
        self.timestep = -1
        self.picked_objects = {}
    # =========================================================================
    def reward(self, env, event=None):
        if event is None: event = env.controller.step(action='Pass')

        # return reward 1 if we picked it up
        rew_i = 0
        rew_e = 0
        for obj in [o for o in event.metadata['objects'] if o['isPickedUp']]:
            if self.curious:
              if obj['objectType'] not in self.picked_objects.keys():
                  self.picked_objects[obj['objectType']] = 1
              else:
                  self.picked_objects[obj['objectType']] += 1
              rew_i += (1.0/(self.picked_objects[obj['objectType']]+1))
            if obj['objectType'] == self.target_object_type:
                rew_e = 1
        return rew_e, rew_i
    # =========================================================================
    def done(self, env):
        self.timestep += 1
        rew_e, rew_i = self.reward(env)
        if self.timestep % self.max_timesteps == 0 or rew_e > 0:
            self.target_object_type = np.random.choice(self.possible_targets)
            self.timestep = 0
            return True
        else:
            return False
    # =========================================================================
    def is_possible(self, env):
        event = self.controller.step(action='Pass')
        targets = [obj for obj in event.metadata['objects'] if obj['objectType']==self.target_object_type]
        if len(targets) == 0:
            return False, targets
        else:
            return True, targets


# =============================================================================
# =============================================================================
# =============================================================================


class ThorEnvironment:
    def __init__(self, scene='FloorPlan29', grid_size=0.25, controller=None, word_vec=None, verbose=False):

        # initialise the AI2Thor controller
        if controller is None:
            self.controller = ai2thor.controller.Controller(scene=scene, gridSize=grid_size, visibilityDistance=0.5)
        else:
            self.controller = controller

        # scene graph
        self.graph = SceneGraph(word_vec)

        self.verbose = verbose
        self.last_pose = None

    # ===============================================================================================
    def hide_objects(self):
        event = self.controller.step(action='Pass')

        # remove all objects from those receptacles
        sourceTypes = ['CounterTop', 'Sink']
        sources = [obj for obj in event.metadata['objects'] if obj['objectType'] in sourceTypes]


        # those are the receiving receptacles we will put objects to
        sinkTypes = ['Cabinet']
        sinks = [obj for obj in event.metadata['objects'] if obj['objectType'] in sinkTypes]

        for src in sources:
            print('Clearing', src['objectId'])
            for obj in src['receptacleObjectIds']:
                sink = np.random.choice(sinks)
                print('\tattempting to move', obj, 'to', sink['objectId'])
                event = self.controller.step(action = 'GetSpawnCoordinatesAboveReceptacle', objectId=sink['objectId'], anywhere=True)
                for pos in event.metadata['actionReturn']:
                    event = self.controller.step(action='PlaceObjectAtPoint', objectId=obj, position=pos)
                    if event.metadata['lastActionSuccess']:
                        break

        # check we were successful
        sourcesIds = [s['objectId'] for s in sources]
        event = self.controller.step(action='Pass')
        for obj in event.metadata['objects']:
            if obj['parentReceptacles'] is not None:
                for p in obj['parentReceptacles']:
                    if p in sourcesIds:
                        return False
        return True

    # ===============================================================================================
    def update_object(self, obj, pose=None):

        # does the object exist in the graph?
        idx = self.graph.find('name', obj['name'])

        # if not, add it and its affordances
        if idx == []:
            idx = self.graph.add_object(obj['objectType'].lower(), data=obj)
        else:
            # if the object exists, update it with the observation given in obj
            assert(len(idx)==1)
            idx = idx[0]
            self.graph.nodes[idx]['data'] = obj

        # if pose is given, check that the object has a goto affordance with that pose connected to it
        if pose is not None:
            if len([g for g in self.graph.get_affordances(idx, 'go') if self.graph.nodes[g]['data']['pose']==pose])==0:
                self.graph.add_affordance(idx, 'go', data={'pose':pose})

        # add other affordances as neccessary
        if obj['pickupable'] and not self.graph.has_affordance(idx, 'pick'):
            self.graph.add_affordance(idx,'pick')

        if obj['receptacle'] and not self.graph.has_affordance(idx, 'put'):
            self.graph.add_affordance(idx,'put')

        if obj['openable'] and not self.graph.has_affordance(idx, 'open'):
            self.graph.add_affordance(idx,'open')

        if obj['openable'] and not self.graph.has_affordance(idx, 'close'):
            self.graph.add_affordance(idx,'close')

        if obj['toggleable'] and not self.graph.has_affordance(idx, 'toggle'):
            self.graph.add_affordance(idx,'toggle')

        # if the object is a receptable, check if its children are already in the map and make sure there is an edge between them
        if obj['receptacleObjectIds'] is not None:
            for child in obj['receptacleObjectIds']:
                try:
                    # for the children that are in the map already
                    child_idx = self.graph.find('objectId', child)[0]

                    # is there an edge from parent to child?
                    if not child_idx in self.graph.successors(idx):
                        self.graph.add_relation_edge(idx, child_idx, 'contains')

                    # is there an edge from child to parent?
                    if not idx in self.graph.successors(child_idx):
                        self.graph.add_relation_edge(child_idx, idx, 'in')

                except IndexError:
                    # the child object is not yet in the map, we can't do anything about it now
                    pass


        # if the object is inside a receptable, check if the receptable is already in the map and make sure there is an edge between them
        if obj['parentReceptacles'] is not None:
            for parent in obj['parentReceptacles']:
                try:
                    # for the parents that are in the map already
                    parent_idx = self.graph.find('objectId', parent)[0]

                    # is there an edge from parent to child?
                    if not idx in self.graph.successors(parent_idx):
                        self.graph.add_relation_edge(parent_idx, idx, 'contains')

                    # is there an edge from child to parent?
                    if not parent_idx in self.graph.successors(idx):
                        self.graph.add_relation_edge(idx, parent_idx, 'in')
                except IndexError:
                    pass
                    # the parent object is not yet in the map, we can't do anything about it now


    # ===============================================================================================
    def explore_environment(self):

        self.graph.clear()

        event = self.controller.step(dict(action='LookDown', degrees=20))
        # get all valid positions
        event = self.controller.step(dict(action='GetReachablePositions'))
        reachable = event.metadata['actionReturn'] # event.metadata['actionReturn'] for new version

        # teleport the robot to all positions, then rotate in place by a fixed angle and remember which objects are visible
        for loc in reachable:
            action = dict(action='Teleport')
            action.update(loc)
            self.controller.step(action)

            for rot in range(0,360,45):   # rotate in 45 deg increments
                event = self.controller.step(dict(action='Rotate', rotation=rot))
                pose = [loc['x'], loc['y'], loc['z'], rot]
                self.last_pose = pose

                for obj in [o for o in event.metadata['objects'] if o['visible']]:
                    self.update_object(obj, pose)

        self.graph.to_torch_graph()
    # ===============================================================================================
    def step(self, idx):
        assert self.graph.nodes[idx]['node_type'] == 'affordance', 'Not an affordance: %d' % idx

        if self.graph.nodes[idx]['affordance'] == 'go':
            # Teleport the agent to the new pose
            pose = self.graph.nodes[idx]['data']['pose']
            event = self.controller.step(dict(action='TeleportFull', x=pose[0], y=pose[1], z=pose[2], rotation=pose[3], horizon=20, standing=True))

            # was this successful?
            if event.metadata['lastActionSuccess']:
                self.last_pose = pose

                # we have to remove the 'at' relations between the robot and the pose affordance and its object
                for node in self.graph.get_relations(self.graph.robot_node, 'at'):
                    self.graph.remove_edge(self.graph.robot_node, node)

                # insert a new 'at' relation between robot and new pose affordance, and the object of that affordance
                obj = next(self.graph.predecessors(idx))
                assert len(list(self.graph.predecessors(idx)))==1

                self.graph.add_relation_edge(self.graph.robot_node, obj, 'at')
                self.graph.add_relation_edge(self.graph.robot_node, idx, 'at')

        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'pick':
            # which object are we trying to pick?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step(dict(action='PickupObject', objectId=objectId))

            # was this successful?
            if event.metadata['lastActionSuccess']:
                # if yes, we have to remove the edges to the affordances
                self.graph.remove_nodes_from(self.graph.get_affordances(obj))

                # remove relation edges (from this object to other objects)
                succ, pred = self.graph.get_related_objects(obj)
                for s in succ:
                    self.graph.remove_edge(obj, s)
                for p in pred:
                    self.graph.remove_edge(p, obj)

                # add new relation edges to the robot node
                self.graph.add_relation_edge(self.graph.robot_node, obj, 'contains')
                self.graph.add_relation_edge(obj, self.graph.robot_node, 'in')

                self.graph.to_torch_graph()

        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'put':
            # which object are we attempting to put down?
            inhand = self.graph.get_relations(self.graph.robot_node, 'contains')

            if len(inhand)<1:
                if self.verbose:
                    print('! Error when executing PUT affordance: Agent does not carry an object.', inhand)
                return False
            elif len(inhand)>1:
                if self.verbose:
                    print('! Error when executing PUT affordance: Agent carries multiple objects.')
                    print(inhand, [self.graph.nodes[x]['data']['name'] for x in inhand])
                return False
            else:
                inhand=inhand[0]

            # make sure we have an object in our hand!
            if self.graph.nodes[inhand]['node_type'] != 'object':
                if self.verbose:
                    print('! Error when executing PUT affordance: Not an object.', obj, self.graph)
                return False

            # which object are we trying to put something on?
            target = next(self.graph.predecessors(idx))
            objectId_target = self.graph.nodes[target]['data']['objectId']
            objectId_inhand = self.graph.nodes[inhand]['data']['objectId']
            
            # get spawn coordinates
            event = self.controller.step(dict(action='PutObject', objectId=objectId_inhand, receptacleObjectId=objectId_target, forceAction=False, placeStationary=True))

            # was this successful?
            if event.metadata['lastActionSuccess']:

                self.controller.step('DropHandObject')
                event = self.controller.step(action='PlaceObjectAtPoint', objectId=objectId_inhand, position=spawn_point)

                # remove connection between robot and dropped inhand object
                self.graph.remove_edge(inhand, self.graph.robot_node)
                self.graph.remove_edge(self.graph.robot_node, inhand)

                # we need to update both the target and the dropped object
                obj = [o for o in event.metadata['objects'] if o['name']==self.graph.nodes[inhand]['data']['name']][0]
                self.update_object(obj, pose = self.last_pose)

                obj = [o for o in event.metadata['objects'] if o['name']==self.graph.nodes[target]['data']['name']][0]
                self.update_object(obj)
            else:
                if self.verbose:
                    print(event.metadata['lastAction'], event.metadata['lastActionSuccess'], event.metadata['errorMessage'])

        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'open':
            # which object are we trying to open?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step('OpenObject', objectId=objectId)
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'close':
            # which object are we trying to close?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step('CloseObject', objectId=objectId)
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'slice':                 
            # which object are we trying to slice?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step(dict(action='SliceObject', objectId=objectId))  
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'heat':                 
            # which object are we trying to heat?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step(dict(action='HeatObject', objectId=objectId))  
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'cool':                 
            # which object are we trying to cool?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step(dict(action='CoolObject', objectId=objectId))  
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'clean':                 
            # which object are we trying to clean?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step(dict(action='CleanObject', objectId=objectId)) 
        # +++++++++       
        elif self.graph.nodes[idx]['affordance'] == 'toggle':                 
            # which object are we trying to toggle?
            obj = next(self.graph.predecessors(idx))
            objectId = self.graph.nodes[obj]['data']['objectId']
            event = self.controller.step('ToggleObjectOn', objectId=objectId) # WARNING: ToggleOff if already ON
        
        # robot affordances (moving, looking up and down, crouching, standing)
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'crouch':
            event = self.controller.step(action='Crouch')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'stand':
            event = self.controller.step(action='Stand')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'up':
            event = self.controller.step(action='LookUp')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'down':
            event = self.controller.step(action='LookDown')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'left':
            event = self.controller.step(action='MoveLeft')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'right':
            event = self.controller.step(action='MoveRight')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'ahead':
            event = self.controller.step(action='MoveAhead')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'back':
            event = self.controller.step(action='MoveBack')
        # +++++++++
        elif self.graph.nodes[idx]['affordance'] == 'done':
            raise NotImplementedError('Done affordance was called.')

        else:
            raise NotImplementedError('Unknown affordance', self.graph.nodes[idx]['affordance'] )

        if event.metadata['lastActionSuccess']:
            result = True
        else:
            if self.verbose:
                print(event.metadata['lastAction'], event.metadata['lastActionSuccess'], event.metadata['errorMessage'])
            result = False


        # update visible objects (except the picked up one, if any)
        for obj in [o for o in event.metadata['objects'] if o['visible'] and not o['isPickedUp']]:
            self.update_object(obj, self.last_pose)

        return result
