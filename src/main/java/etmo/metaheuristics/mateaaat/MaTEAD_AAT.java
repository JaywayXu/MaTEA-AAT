package etmo.metaheuristics.mateaaat;

import etmo.core.*;
import etmo.util.JMException;
import etmo.util.KLD;
import etmo.util.PseudoRandom;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.Vector;

public class MaTEAD_AAT extends MtoAlgorithm {
	private int populationSize;
	private SolutionSet[] population;

	double[][] z;
	double[][][] lambda;
	int T;
	int[][][] neighborhood;
	double delta;
	int nr;
	Solution[][] indArray;
	String functionType;

	int normalStep;
	int extraStep;
	double[][] transferP;
	double[] improvements;

	int evaluations;
	int maxEvaluations;

	Operator crossover;
	Operator crossover2;
	Operator mutation;

	String dataDirectory;

	int taskNum;

	public MaTEAD_AAT(ProblemSet problemSet) {
		super(problemSet);

		functionType = "_TCHE1";

	} // MOEAD

	private void initState() {
		taskNum = problemSet_.size();
		evaluations = 0;
		maxEvaluations = (Integer) this.getInputParameter("maxEvaluations");
		populationSize = (Integer) this.getInputParameter("populationSize");
		dataDirectory = this.getInputParameter("dataDirectory").toString();
		T = (Integer) this.getInputParameter("T");
		nr = (Integer) this.getInputParameter("nr");
		delta = (Double) this.getInputParameter("delta");

		normalStep = (Integer) this.getInputParameter("normalStep");
		extraStep =  (Integer) this.getInputParameter("extraStep");
		double initTransferP = (Double) this.getInputParameter("initTransferP");

		crossover = operators_.get("crossover"); // default: DE crossover
		crossover2 = operators_.get("crossover2");
		mutation = operators_.get("mutation"); // default: polynomial mutation

		population = new SolutionSet[taskNum];
		indArray = new Solution[taskNum][];
		neighborhood = new int[taskNum][populationSize][T];
		z = new double[taskNum][];
		lambda = new double[taskNum][][];

		transferP = new double[taskNum][taskNum];

		improvements = new double[taskNum];

		for (int k = 0; k < taskNum; k++){
			indArray[k] = new Solution[problemSet_.get(k).getNumberOfObjectives()];
			z[k] = new double[problemSet_.get(k).getNumberOfObjectives()];
			lambda[k] = new double[populationSize][problemSet_.get(k).getNumberOfObjectives()];

			Arrays.fill(transferP[k], initTransferP);
		}
	}

	public SolutionSet[] execute() throws JMException, ClassNotFoundException {
		initState();
		initUniformWeight();
		initNeighborhood();
		initPopulation();
		initIdealPoint();

		while (evaluations < maxEvaluations) {
			iterate();
		}
		return population;
	}

	public void iterate() throws JMException {
		calculatedConverge(normalStep);
		
		for (int taskId = 0; taskId < taskNum; taskId++) {
			int assistTask = getSourceTaskId(taskId, "rndwd");
			if (PseudoRandom.randDouble() < transferP[taskId][assistTask] && taskId != assistTask){
				int t = 0;
				while (improvements[assistTask] > improvements[taskId] && t < extraStep){
					improvements[assistTask] = solelyConverge(assistTask, 1);
					t ++;
				}
				transferConverge(taskId, assistTask);
			}
		}
	}

	void calculatedConverge(int times) throws JMException {
		Arrays.fill(improvements, 0);
		for (int k = 0; k < taskNum; k++){
			improvements[k] = solelyConverge(k, times);
		}
	}

	public double solelyConverge(int taskId, int times) throws JMException {
		double improvement = 0;
		for (int t = 0; t < times; t ++){
			for (int i = 0; i < populationSize; i++) {
				int type = PseudoRandom.randDouble() < delta ? 1 : 2;
				Solution child = normalReproduce(taskId, i, type);
				problemSet_.get(taskId).evaluate(child);
				evaluations++;

				updateReference(child, taskId);
				improvement += updateProblemAndGetImprove(child, i, type, taskId);
			}
		}
		return improvement;
	}

	public void transferConverge(int targetTaskId, int sourceTaskId) throws JMException {
		int betterCount = 0;

		for (int i = 0; i < populationSize; i++) {
			int type = 2;
			Solution child;
			child = transferReproduce(targetTaskId, sourceTaskId, i);

			problemSet_.get(targetTaskId).evaluate(child);
			evaluations++;

			updateReference(child, targetTaskId);
			betterCount += updateProblem(child, i, type, targetTaskId);
		}

		if (betterCount <= populationSize * 0.1 * transferP[targetTaskId][sourceTaskId]) {
			transferP[targetTaskId][sourceTaskId] = Math.max(transferP[targetTaskId][sourceTaskId] * 0.9, 0.1);
		} else if (betterCount > populationSize * 0.5 * transferP[targetTaskId][sourceTaskId]){
			transferP[targetTaskId][sourceTaskId] = Math.min(transferP[targetTaskId][sourceTaskId] / 0.9, 1);
		}
	}

	private Solution normalReproduce(int taskId, int currentId, int type) throws JMException {
		Solution child;

		Vector<Integer> p = new Vector<>();
		matingSelection(p, currentId, 2, type, taskId);

		Solution[] parents = new Solution[3];

		parents[0] = population[taskId].get(p.get(0));
		parents[1] = population[taskId].get(p.get(1));
		parents[2] = population[taskId].get(currentId);

		child = (Solution) crossover.execute(new Object[]{
				population[taskId].get(currentId), parents});
		mutation.execute(child);
		return child;
	}

	private Solution transferReproduce(int targetTaskId, int sourceTaskId, int currentId) throws JMException {
		Solution child;

		// SBX
		Solution[] parents = new Solution[2];
		parents[0] = population[sourceTaskId].get(currentId);
		parents[1] = population[targetTaskId].get(PseudoRandom.randInt(0, populationSize - 1));
		child = ((Solution[]) crossover2.execute(parents))[PseudoRandom.randInt(0, 1)];

		return child;
	}

	public int getSourceTaskId(int targetTaskId, String type) throws JMException {
		int sourceTaskId = targetTaskId;

		if (type.equalsIgnoreCase("random")) {
			// random
			while (sourceTaskId == targetTaskId)
				sourceTaskId = PseudoRandom.randInt(0, taskNum - 1);
		}
		else if (type.equalsIgnoreCase("wd")) {
			// Wasserstein Distance
			double[] distance = new double[taskNum];
			double[] finalScore = new double[taskNum];
			double maxScore = 0;
			double minDistance = Double.MAX_VALUE;
			double maxDistance = 0;
			for (int k = 0; k < taskNum; k++) {
				if (k == targetTaskId) continue;
				distance[k] = WassersteinDistance.getWD2(
						population[targetTaskId].getMat(),
						population[k].getMat());
				minDistance = Math.min(distance[k], minDistance);
				maxDistance = Math.max(distance[k], maxDistance);
			}
			for (int k = 0; k < taskNum; k++){
				if (k == targetTaskId) continue;
				distance[k] = (distance[k] - minDistance) / (maxDistance - minDistance);
				finalScore[k] = (1 - distance[k]) * transferP[targetTaskId][sourceTaskId];
				maxScore = Math.max(maxScore, finalScore[k]);
			}

			if (maxScore > 0)
				sourceTaskId = Utils.rouletteExceptZero(finalScore);
			else
				sourceTaskId = targetTaskId;
		} 
		else if (type.equalsIgnoreCase("kl")) {
			double[] distance;
			double[] finalScore = new double[taskNum];
			double maxScore = 0;
			KLD kld = new KLD(problemSet_, population);
			distance = kld.getKDL(targetTaskId);
			for (int k = 0; k < taskNum; k++){
				finalScore[k] = transferP[targetTaskId][k] / distance[k];
				maxScore = Math.max(maxScore, finalScore[k]);
			}
			if (maxScore > 0)
				sourceTaskId = Utils.rouletteExceptZero(finalScore);
			else
				sourceTaskId = targetTaskId;
		}
		else if (type.equalsIgnoreCase("rndwd")) {
			int size = (int)(Math.sqrt(taskNum));
			int[] taskOrder = new int[taskNum];
			int[] perm = new int[size];
			Utils.randomPermutation(taskOrder, taskNum);
			System.arraycopy(taskOrder, 0, perm, 0, size);

			double[] distance = new double[size];
			double minDistance = Double.MAX_VALUE;
			double maxDistance = 0;
			for (int i = 0; i < size; i++){
				if (perm[i] == targetTaskId) continue;
				distance[i] = WassersteinDistance.getWD2(
						population[targetTaskId].getMat(),
						population[perm[i]].getMat());
				minDistance = Math.min(minDistance, distance[i]);
				maxDistance = Math.max(maxDistance, distance[i]);
			}

			double[] finalScore = new double[size];
			double maxScore = 0;
			for (int i = 0; i < size; i++){
				if (perm[i] == targetTaskId) continue;
				finalScore[i] = transferP[targetTaskId][perm[i]] / distance[i];
				maxScore = Math.max(finalScore[i], maxScore);
			}
			if (maxScore > 0){
				sourceTaskId = perm[Utils.rouletteExceptZero(finalScore)];
			}
		}
		else {
			System.out.println("Unsupported type: " + type);
		}

		return sourceTaskId;
	}

	public void initUniformWeight() { // init lambda vectors
		for (int k = 0; k < taskNum; k++) {
			if (problemSet_.get(k).getNumberOfObjectives() == 2) {
				for (int n = 0; n < populationSize; n++) {
					double a = 1.0 * n / (populationSize - 1);
					lambda[k][n][0] = a;
					lambda[k][n][1] = 1 - a;
				} // for
			} // if
			else {
				String dataFileName;
				dataFileName = "W" + problemSet_.get(k).getNumberOfObjectives() + "D_"
						+ populationSize + ".dat";

				try {
					// Open the file
					FileInputStream fis = new FileInputStream(dataDirectory + "/"
							+ dataFileName);
					InputStreamReader isr = new InputStreamReader(fis);
					BufferedReader br = new BufferedReader(isr);

					int i = 0;
					int j;
					String aux = br.readLine();
					while (aux != null) {
						StringTokenizer st = new StringTokenizer(aux);
						j = 0;
						while (st.hasMoreTokens()) {
							double value = Double.parseDouble(st.nextToken());
							lambda[k][i][j] = value;
							j++;
						}
						aux = br.readLine();
						i++;
					}
					br.close();
				} catch (Exception e) {
					System.out
							.println("initUniformWeight: failed when reading for file: "
									+ dataDirectory + "/" + dataFileName);
					e.printStackTrace();
				}
			} // else
		}
	} // initUniformWeight

	/**
   * 
   */
	public void initNeighborhood() {
		for (int k = 0; k < taskNum; k++) {
			double[] x = new double[populationSize];
			int[] idx = new int[populationSize];

			for (int i = 0; i < populationSize; i++) {
				for (int j = 0; j < populationSize; j++) {
					x[j] = Utils.distVector(lambda[k][i], lambda[k][j]);
					idx[j] = j;
				}

				Utils.minFastSort(x, idx, populationSize, T);
				System.arraycopy(idx, 0, neighborhood[k][i], 0, T);
			}
		}
	}

	public void initPopulation() throws JMException, ClassNotFoundException {
		for (int k = 0; k < taskNum; k++) {
			population[k] = new SolutionSet(populationSize);
			for (int i = 0; i < populationSize; i++) {
				Solution newSolution = new Solution(problemSet_);

				problemSet_.get(k).evaluate(newSolution);
				evaluations++;
				population[k].add(newSolution);
			} // for
		}
	} // initPopulation

	/**
   * 
   */
	void initIdealPoint() throws JMException, ClassNotFoundException {
		for (int k = 0; k < taskNum; k++) {
			for (int i = 0; i < problemSet_.get(k).getNumberOfObjectives(); i++) {
				z[k][i] = 1.0e+30;
				indArray[k][i] = new Solution(problemSet_);
				problemSet_.get(k).evaluate(indArray[k][i]);
				evaluations++;
			} // for

			for (int i = 0; i < populationSize; i++) {
				updateReference(population[k].get(i), k);
			} // for
		}
	} // initIdealPoint

	/**
   * 
   */
	public void matingSelection(Vector<Integer> list, int cid, int size, int type, int taskId) {
		int ss;
		int r;
		int p;

		ss = neighborhood[taskId][cid].length;
		while (list.size() < size) {
			if (type == 1) {
				r = PseudoRandom.randInt(0, ss - 1);
				p = neighborhood[taskId][cid][r];
				// p = population[cid].table[r];
			} else {
				p = PseudoRandom.randInt(0, populationSize - 1);
			}
			boolean flag = true;
			for (Integer integer : list) {
				if (integer == p) // p is in the list
				{
					flag = false;
					break;
				}
			}

			// if (flag) list.push_back(p);
			if (flag) {
				list.addElement(p);
			}
		}
	} // matingSelection

	boolean updateReference(Solution individual, int taskId) {
		boolean isUpdate = false;
		int objOffset = problemSet_.get(taskId).getStartObjPos();
		for (int n = 0; n < problemSet_.get(taskId).getNumberOfObjectives(); n++) {
			if (individual.getObjective(n + objOffset) < z[taskId][n]) {
				isUpdate = true;
				z[taskId][n] = individual.getObjective(n + objOffset);

				indArray[taskId][n] = individual;
			}
		}
		return isUpdate;
	} // updateReference

	double updateProblemAndGetImprove(Solution indiv, int id, int type, int taskId) {
		double improvement = 0;
		int size;
		int time;

		time = 0;

		if (type == 1) {
			size = neighborhood[taskId][id].length;
		} else {
			size = population[taskId].size();
		}
		int[] perm = new int[size];

		Utils.randomPermutation(perm, size);

		for (int i = 0; i < size; i++) {
			int k;
			if (type == 1) {
				k = neighborhood[taskId][id][perm[i]];
			} else {
				k = perm[i]; // calculate the values of objective function
								// regarding the current subproblem
			}
			double f1, f2;

			f1 = fitnessFunction(population[taskId].get(k), lambda[taskId][k], taskId);
			f2 = fitnessFunction(indiv, lambda[taskId][k], taskId);

			if (f2 < f1) {
				population[taskId].replace(k, new Solution(indiv));
				improvement += ((f1 - f2) / f1);
				time++;
			}
			// the maximal number of solutions updated is not allowed to exceed
			// 'limit'
			if (time >= nr) {
				break;
			}
		}

		return improvement;
	} // updateProblem

	int updateProblem(Solution indiv, int id, int type, int taskId) {
		int size;
		int time;

		time = 0;

		if (type == 1) {
			size = neighborhood[taskId][id].length;
		} else {
			size = population[taskId].size();
		}
		int[] perm = new int[size];

		Utils.randomPermutation(perm, size);

		for (int i = 0; i < size; i++) {
			int k;
			if (type == 1) {
				k = neighborhood[taskId][id][perm[i]];
			} else {
				k = perm[i]; // calculate the values of objective function
								// regarding the current subproblem
			}
			double f1, f2;

			f1 = fitnessFunction(population[taskId].get(k), lambda[taskId][k], taskId);
			f2 = fitnessFunction(indiv, lambda[taskId][k], taskId);

			if (f2 < f1) {
				population[taskId].replace(k, new Solution(indiv));
				time++;
			}
			// the maximal number of solutions updated is not allowed to exceed
			// 'limit'
			if (time >= nr) {
				break;
			}
		}

		return time;
	} // updateProblem

	double fitnessFunction(Solution individual, double[] lambda, int taskId) {
		double fitness;
		fitness = 0.0;

		int objOffset = problemSet_.get(taskId).getStartObjPos();
		if (functionType.equals("_TCHE1")) {
			double maxFun = -1.0e+30;

			for (int n = 0; n < problemSet_.get(taskId).getNumberOfObjectives(); n++) {
				double diff = Math.abs(individual.getObjective(n + objOffset) - z[taskId][n]);

				double feval;
				if (lambda[n] == 0) {
					feval = 0.0001 * diff;
				} else {
					feval = diff * lambda[n];
				}
				if (feval > maxFun) {
					maxFun = feval;
				}
			} // for

			fitness = maxFun;
		} // if
		else {
			System.out.println("MOEAD.fitnessFunction: unknown type "
					+ functionType);
			System.exit(-1);
		}
		return fitness;
	} // fitnessEvaluation
} // MOEAD

