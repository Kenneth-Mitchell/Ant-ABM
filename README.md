# Mesa-Ant-ABM-
This is an agent based model of turtle ants created with funding by Harvey Mudd College and the National Science Foundation. The model is made with Mesa and Networkx, using Matplotlib and Pandas for graphs.

The model exists in two forms. 

The Individual Model (MesAntIndividual.py) assumes that ants do not interact with their environment, and they stop when reaching the tip of a tree.

The Colony Model (MesAntColony.py) assumes that ants do interact with their environment, leaving pheromone trails and changing states.

Either model can be ran by using the command experiment(n,x), which runs the model with n time steps and x ants (default if not specified is 50 time steps with 1000 ants). Results will be displayed as a network plot and area plot.

# Recent updates:

The two versions of the model exist in a single jupyter notebook file now. Also a new version of the model was made for use in our in progress publication. This version makes use of a statistical model from our experimental data (not included).


