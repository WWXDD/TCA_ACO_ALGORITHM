/**
 * WANG, DAWEI
 * AMARO AULAR, YSABEL CRISTINA
 * MARTÍNEZ VIDAL, ANTONIO
 */


package com.net2plan.examples.netDesignAlgorithm.tca;
import com.net2plan.interfaces.networkDesign.IAlgorithm;
import com.net2plan.interfaces.networkDesign.Net2PlanException;
import com.net2plan.interfaces.networkDesign.NetPlan;
import com.net2plan.utils.Constants;
import com.net2plan.utils.Pair;
import com.net2plan.utils.RandomUtils;
import com.net2plan.utils.Triple;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * This algorithm computes the bidirectional ring which minimizes the total ring 
 * length, using an ACO (Ant Colony Optimization) heuristic, described as
 * Ant System in the literature: M. Dorigo, V. Maniezzo, A. Colorni, "Ant system: 
 * optimization by a colony of cooperating agents", IEEE T on Cybernetics, 1996.
 * The cost of a link equals the euclidean distance between link end nodes. The 
 * algorithm executes ACO iterations until the maxExecTime is reached.
 * In each ACO iteration, a loop for each ant is executed. Each ant, creates a 
 * greedy-randomized solution using the pheromones information of each potential link, 
 * and each link length, as follows. Each ant starts in a node chosen randomly. 
 * At each iteration, an ant in node n decides the next node to visit 
 * randomly among the non-visited nodes, being the probability of choosing node 
 * n' proportional to ph_nn'^alpha b_nn'^beta. ph_nn' is the amount of pheromones 
 * associated to link nn', b_nn' is the inverse of the distance between both 
 * nodes. alpha and beta are parameters tuning the importance of pheromones 
 * and link distances respectively. After all ants have finished, an evaporation 
 * strategy is executed, where each link nn' looses pheromones multiplicatively 
 * (pheromones are multiplied by 1-r, where r is a 0...1 evaporation factor). 
 * After evaporation phase, a reinforcement step is completed, where each ant a adds 
 * a quantity 1/La to the pheromones of all links traversed, being La the total 
 * distance of its ring.
 */
public class TCA_ACO_TSP implements IAlgorithm
{
	private final static boolean DEBUG = false;
	
	@Override
	public String executeAlgorithm(NetPlan netPlan, Map<String, String> algorithmParameters, Map<String, String> net2planParameters)
	{
		/* Basic checks */
		final int N = netPlan.getNumberOfNodes();
		if (N == 0) throw new Net2PlanException("The input design must have nodes");

		/* Initialize some variables */
		final double maxExecTimeSecs = Double.parseDouble(algorithmParameters.get("maxExecTimeSecs"));
		final int numAnts = Integer.parseInt(algorithmParameters.get("numAnts"));
		final double alpha = Double.parseDouble(algorithmParameters.get("alpha"));
		final double beta = Double.parseDouble(algorithmParameters.get("beta"));
		final double evaporationFactor = Double.parseDouble(algorithmParameters.get("evaporationFactor"));
		final double linkCapacities = Double.parseDouble(algorithmParameters.get("linkCapacities"));
		final long randomSeed = Long.parseLong(algorithmParameters.get("randomSeed"));
		final long algorithmEndTime = System.nanoTime() + (long) (1E9 * maxExecTimeSecs);
		Random r = new Random(randomSeed);

		/* Check input parameters */
		if (numAnts <= 0) throw new Net2PlanException("A positive number of ants is needed");
		if ((evaporationFactor < 0) || (evaporationFactor > 1)) throw new Net2PlanException("The evaporationFactor must be between 0 and 1");
		if (linkCapacities < 0) throw new Net2PlanException("Link capacities must be a non-negative number");
		if (maxExecTimeSecs <= 0) throw new Net2PlanException("Algorithm running time must be a non-negative number");

		/* Compute the distances between each node pair */
		double[][] c_ij = netPlan.getNodeEuclideanDistanceMatrix().toArray();

		/* The best solution found so far (incumbent solution) is stored in these variables */
		ArrayList<Integer> best_nodeSequence = new ArrayList<Integer>();
		double best_cost = Double.MAX_VALUE;

		/* Initialize some ACO control variables: the pheromones */
		double[][] pheromones_ij = new double[N][N];
		for (int n1 = 0; n1 < N; n1++)
			for (int n2 = 0; n2 < N; n2++)
				if (n1 != n2)
					pheromones_ij[n1][n2] = 1;

		/* Main loop. Stop when maximum execution time is reached */
		while (System.nanoTime() < algorithmEndTime)
		{
			ArrayList<Pair<ArrayList<Integer>, Double>> antSolutions = new ArrayList<Pair<ArrayList<Integer>, Double>>(numAnts);
			for (int a = 0; a < numAnts; a++)
			{
				/* Build a greedy-random solution using pheromones info */
				Pair<ArrayList<Integer>, Double> sol = computeAntSolution(r, alpha, beta, pheromones_ij, c_ij);
				if (DEBUG) checkRing(sol, c_ij, "SOLUTION FOUND");

				/* Update incumbent solution */
				if (sol.getSecond() < best_cost)
				{
					best_cost = sol.getSecond();
					best_nodeSequence = (ArrayList<Integer>) sol.getFirst().clone();
					if (DEBUG) checkRing(sol, c_ij, "-- IMPROVING SOLUTION");
				}
				
				antSolutions.add(sol);
			}

			/* Apply evaporation strategy */
			for (int n1 = 0; n1 < N; n1++)
				for (int n2 = 0; n2 < N; n2++)
					if (n1 != n2)
						pheromones_ij[n1][n2] *= (1 - evaporationFactor);

			/* Apply reinforcement strategy */
			for (int a = 0; a < numAnts; a++)
			{
				ArrayList<Integer> sol = antSolutions.get(a).getFirst();
				double benefit = 1 / antSolutions.get(a).getSecond();
				for (int cont = 0; cont < N - 1; cont++)
					pheromones_ij[sol.get(cont)][sol.get(cont + 1)] += benefit;

				pheromones_ij[sol.get(N - 1)][sol.get(0)] += benefit;
			}
		}

		/* Save the best solution found into the netPlan object */
		long[] nodeIdsVector = netPlan.getNodeIdsVector();
		netPlan.removeAllLinks();
		netPlan.setRoutingType(Constants.RoutingType.SOURCE_ROUTING);
		for (int cont = 0; cont < N - 1; cont++)
		{
			final int n1 = best_nodeSequence.get(cont);
			final int n2 = best_nodeSequence.get(cont + 1);
			final long nodeId1 = nodeIdsVector[n1];
			final long nodeId2 = nodeIdsVector[n2];
			netPlan.addLinkBidirectional(nodeId1, nodeId2, linkCapacities, c_ij[n1][n2], null);
		}
		
		final int firstRingN = best_nodeSequence.get(0);
		final int lastRingN = best_nodeSequence.get(best_nodeSequence.size() - 1);
		final long firstRingNodeId = nodeIdsVector[firstRingN];
		final long lastRingNodeId = nodeIdsVector[lastRingN];
		netPlan.addLinkBidirectional(firstRingNodeId, lastRingNodeId, linkCapacities, c_ij[firstRingN][lastRingN], null);

		/* Return printing the total distance of the solution */
		return "Ok! Cost : " + best_cost;
	}

	@Override
	public String getDescription()
	{
		return "This algorithm computes the bidirectional ring which minimizes the total ring length, using an ACO (Ant Colony Optimization) heuristic, described as"
			+ " Ant System in the literature: M. Dorigo, V. Maniezzo, A. Colorni, \"Ant system: optimization by a coloyn of cooperating agents\", IEEE T on Cybernetics, 1996."
			+ " The cost of a link equals the euclidean distance between link end nodes. The algorithm executes ACO iterations until the maxExecTime is reached."
			+ " In each ACO iteration, a loop for each ant is executed. Each ant, creates a greedy-randomized solution using the pheromones information of each potential link,"
			+ " and each link length, as follows. Each ant starts in a node chosen randomly. At each iteration, an ant in node n decides the next node to visit "
			+ " randomly among the non-visited nodes, being the probability of choosing node n' proportional to ph_nn'^alpha b_nn'^beta. ph_nn' is the amount of pheromones"
			+ " associated to link nn', b_nn' is the inverse of the distance between both nodes. alpha and beta are parameters tuning the importance of pheromones "
			+ " and link distances respectively. After all ants have finished, an evaporation strategy is executed, where each link nn' looses pheromones multiplicatively"
			+ " (pheromones are multiplied by 1-r, where r is a 0...1 evaporation factor). After evaporation phase, a reinforcement step is completed, where each ant a adds "
			+ " a quantity 1/La to the pheromones of all links traversed, being La the total distance of its ring.";
	}

	@Override
	public List<Triple<String, String, String>> getParameters()
	{
		List<Triple<String, String, String>> algorithmParameters = new ArrayList<Triple<String, String, String>>();
		algorithmParameters.add(Triple.of("maxExecTimeSecs", "10", "Execution time of the algorithm. The algorithm will stop after this time, returning the best solution found"));
		algorithmParameters.add(Triple.of("numAnts", "10", "Number of ants in the colony"));
		algorithmParameters.add(Triple.of("alpha", "1", "Alpha factor tuning the pheromone influence in the ant movement"));
		algorithmParameters.add(Triple.of("beta", "1", "Beta factor tuning the link distance influence in the ant movement"));
		algorithmParameters.add(Triple.of("evaporationFactor", "0.5", "Factor controlling the evaporation of pheromones"));
		algorithmParameters.add(Triple.of("linkCapacities", "100", "Capacities to set to the links"));
		algorithmParameters.add(Triple.of("randomSeed", "1", "Seed for the random number generator"));

		return algorithmParameters;
	}

	/**
	 * This function implements the greedy-randomized computation of a ring by 
	 * an ant, as described in the class documentation.
	 *
	 * @since 1.0
	 */
	private Pair<ArrayList<Integer>, Double> computeAntSolution(Random r, double alpha, double beta, double[][] pheromones_ij, double[][] c_ij)
	{
		/* Initialize some variables */
		final int N = pheromones_ij.length;

		/* Sequence of nodes traversed: first node is chosen randomly */
		ArrayList<Integer> nodeSequence = new ArrayList<Integer>(N); /* this will hold the ring: sequence of nodes traversed */
		final int initialNode = r.nextInt(N);
		nodeSequence.add(initialNode);

		/* Keep a set of non-visited nodes. Initially, the list contains all nodes but the first */
		HashSet<Integer> notVisitedNode = new LinkedHashSet<Integer>();
		for (int i = 0; i < N; i++)
			if (i != initialNode)
				notVisitedNode.add(i);

		/* In each iteration, a node is added to the ring */
		double nodeSequenceCost = 0;
		while (nodeSequence.size() < N)
		{
			/* Create a list with the probabilities of possible next nodes  */
			final int currentNode = nodeSequence.get(nodeSequence.size() - 1);
			double[] p_next = new double[notVisitedNode.size()];
			int[] nextNodes = new int[notVisitedNode.size()];
			int counter = 0;
			for (int n : notVisitedNode)
			{
				nextNodes[counter] = n;
				p_next[counter] = Math.pow(pheromones_ij[currentNode][n], alpha) * Math.pow(1 / c_ij[currentNode][n], beta);
				counter++;
			}

			/* Choose next node randomly according to the probabilities computed */
			final int nextNodeIndex = RandomUtils.selectWeighted(p_next, r);
			final int nextNode = nextNodes[nextNodeIndex];

			/* Add the node to the rings */
			nodeSequence.add(nextNode);
			notVisitedNode.remove(nextNode);
			nodeSequenceCost += c_ij[currentNode][nextNode];
		}

		/* Update the cost with the link from last node to first node */
		nodeSequenceCost += c_ij[nodeSequence.get(nodeSequence.size() - 1)][nodeSequence.get(0)];

		return Pair.of(nodeSequence, nodeSequenceCost);
	}

	/**
	 * This function receives a solution (node sequence and cost), and checks it 
	 * is a valid ring, and that the cost is well calculated.
	 * 
	 * @since 1.0
	 */
	private static void checkRing(Pair<ArrayList<Integer>, Double> sol, double[][] c_ij, String message)
	{
		System.out.println(message + ": Sequence of nodes: " + sol.getFirst());
		final int N = c_ij.length;
		boolean[] inRing = new boolean[N];
		final ArrayList<Integer> sol_ns = sol.getFirst();
		double checkCost = 0;
		if (sol_ns.size() != N)
		{
			throw new RuntimeException("The solution is not a ring");
		}
		for (int cont = 0; cont < N; cont++)
		{
			int a_e = sol_ns.get(cont);
			int b_e = (cont == N - 1) ? sol_ns.get(0) : sol_ns.get(cont + 1);
			if (inRing[a_e] == true)
			{
				throw new RuntimeException("The solution is not a ring");
			}
			else
			{
				checkCost += c_ij[a_e][b_e];
				inRing[a_e] = true;
			}
		}
		
		if (Math.abs(checkCost - sol.getSecond()) > 1e-3)
			throw new RuntimeException("The solution cost is not well calculated. Solution true cost: " + checkCost + ", solution advertised cost: " + sol.getSecond());
		
		System.out.println(message + " -- Cost: " + checkCost + ": Sequence of nodes: " + sol.getFirst());
	}
}
