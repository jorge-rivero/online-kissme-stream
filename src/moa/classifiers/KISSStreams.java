/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.classifiers;

import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import moa.classifiers.core.driftdetection.DriftDetectionMethod;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.*;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import java.io.*;

/**
 *
 * @author Jorge Luis Rivero Perez
 */
public class KISSStreams extends AbstractClassifier {

    private static final int DEFAULT_SIZE_WINDOW = 3000;
    private static final int DEFAULT_SIZE_KNN = 5;
    private static final double EPSILON = 1e-7;
    private static final double LAMBDA = 0.001;
    public IntOption widthInitOption = new IntOption("InitialWidth",
            'i', "Size of first Window for training learner.", DEFAULT_SIZE_WINDOW, 1, Integer.MAX_VALUE);
    public IntOption kNearestNeighborsOption = new IntOption("KNN", 'k',
            "Number of nearest neighbours (k) used in classification.", DEFAULT_SIZE_KNN, 1, Integer.MAX_VALUE);
    public MultiChoiceOption weighFunctionOption = new MultiChoiceOption(
            "weighFunction", 'w', "The weigh function to use.", new String[]{
        "WEIGHT_NONE", "WEIGHT_INVERSE", "WEIGHT_SIMILARITY"}, new String[]{
        "No distance weighting",
        "Weight neighbours by the inverse of their distance",
        "Weight neighbours by 1 - their distance"}, 1);
    /**
     * Concept Drift
     */
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", DriftDetectionMethod.class, "DDM");
    protected int changeDetected;
    protected int warningDetected;
    protected int ddmLevel;
    protected DriftDetectionMethod driftDetectionMethod;
    /**
     * matrix of difference
     */
    protected Matrix similar_matrix;
    protected Matrix dissimilar_matrix;
    protected Matrix mahalanobis_matrix;
    /**
     * Function of distance
     */
    protected MahalanobisDistance distanceFunction;
    /**
     * Size of matrices of difference
     */
    protected int similar_size;
    protected int dissimilar_size;
    /**
     * Number of attribute
     */
    protected int m_attributes;
    /**
     * Index of class
     */
    protected int classIndex;
    /**
     * Number of class
     */
    protected int classCount;
    /**
     * Type of class value
     */
    protected int classType;
    /**
     * Size of window
     */
    protected int window_size;
    /**
     * K-nearest neighbors
     */
    protected int KNN;
    /**
     * for nearest-neighbor search.
     */
    protected NearestNeighbourSearch search;
    /**
     * window_training
     */
    protected Instances learner;
    /**
     * First time for training
     */
    protected boolean first_learned;
    /**
     * Type of distance function
     */
    protected int m_DistanceWeighting;
    /**
     * no weighting.
     */
    public static final int WEIGHT_NONE = 1;
    /**
     * weight by 1/distance.
     */
    public static final int WEIGHT_INVERSE = 2;
    /**
     * weight by 1-distance.
     */
    public static final int WEIGHT_SIMILARITY = 4;
    /**
     * possible instance weighting methods.
     */
    public static final Tag[] TAGS_WEIGHTING = {
        new Tag(WEIGHT_NONE, "No distance weighting"),
        new Tag(WEIGHT_INVERSE, "Weight by 1/distance"),
        new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance")
    };
    private boolean BUG = true;

    
    public int cantInstancias;
    public int cantMiss;
    
    
    @Override
    public void resetLearningImpl() {
        try {
            //this.cantInstancias=0;
            this.cantMiss=0;
            this.learner = null;
            this.search = new LinearNNSearch();
            this.window_size = this.widthInitOption.getValue();
            this.KNN = this.kNearestNeighborsOption.getValue();
            this.first_learned = false;
            this.similar_size = this.dissimilar_size = 0;
            this.changeDetected = this.warningDetected = 0;
            this.distanceFunction = new MahalanobisDistance();
            this.driftDetectionMethod = ((DriftDetectionMethod) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
            switch (this.weighFunctionOption.getChosenIndex()) {
                case 0:
                    this.m_DistanceWeighting = WEIGHT_NONE;
                    break;
                case 1:
                    this.m_DistanceWeighting = WEIGHT_INVERSE;
                    break;
                default:
                    this.m_DistanceWeighting = WEIGHT_SIMILARITY;
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        System.out.println("Training on instance: " + inst);        
        try {
            this.m_attributes = inst.numAttributes();
            this.classIndex = inst.classIndex();
            this.classCount = inst.numClasses();

            if (inst.classIsMissing()) {
                return;
            }

            if (!this.first_learned) {
                if (this.learner == null) {
                    int N = this.m_attributes - 1;
                    this.similar_matrix = new Matrix(N, N);
                    this.dissimilar_matrix = new Matrix(N, N);
                    this.mahalanobis_matrix = Matrix.identity(N, N);
                    this.distanceFunction.setMatrix(this.mahalanobis_matrix);
                    this.search.setDistanceFunction(this.distanceFunction);
                    this.learner = new Instances(inst.dataset(), 0, 0);
                }
                if (this.learner.size() < window_size) {
                    this.insertInstance(inst);
                }
                if (this.learner.size() == window_size) {
                    this.updateDistanceFunction();
                    this.first_learned = true;
                }
            } else {

                Instances neighbours = this.search.kNearestNeighbours(inst, KNN);

                // drift concept detection
                int trueClass = (int) (inst.classValue() + 1e-2);
                boolean prediction = (Utils.maxIndex(makeDistribution(neighbours, search.getDistances())) == trueClass);
                
                this.ddmLevel = this.driftDetectionMethod.computeNextVal(prediction);

                switch (ddmLevel) {
                    case DriftDetectionMethod.DDM_WARNING_LEVEL:
                        updateDistanceFunction();
                        this.driftDetectionMethod = ((DriftDetectionMethod) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
                        this.warningDetected++;
                        break;
                    case DriftDetectionMethod.DDM_OUTCONTROL_LEVEL:
                        this.resetLearning();
                        this.changeDetected++;
                        break;
                }

                if (prediction) {
                    int cant = 0;
                    for (int i = 0; i < neighbours.numInstances(); ++i) {
                        if (Math.abs(neighbours.get(i).classValue() - inst.classValue()) > EPSILON) {
                            deleteInstance(neighbours.get(i));
                            ++cant;
                            
                        }
                    }
                                                          
                }
                this.insertInstance(inst);
                while (learner.size() > window_size) {
                    this.deleteInstance(learner.get(0));
                }
            }
            this.search.setInstances(learner);
        } catch (Exception ex) {
            Logger.getLogger(KISSStreams.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * This method is demo, not used yet
     *
     * @param inst
     * @return
     */
    private int getIndexOfInstance(Instance inst) {
        for (int i = 0; i < learner.numInstances(); ++i) {
            if (equalsInstance(inst, learner.get(i))) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Delete an instance from base case and update parameters in classifier
     *
     * @param inst instance to be erased.
     */
    private void deleteInstance(Instance inst) {
        int position = getIndexOfInstance(inst);
        if (position >= 0) { // It's in learner                        
            for (int i = 0; i < learner.numInstances(); ++i) {
                if (Math.abs(inst.classValue() - learner.get(i).classValue()) < EPSILON) { // same class
                    this.outerVectorProduct(inst, learner.get(i), similar_matrix.getArray(), -1);
                    --similar_size;
                } else {
                    this.outerVectorProduct(inst, learner.get(i), dissimilar_matrix.getArray(), - 1);
                    --dissimilar_size;
                }
            }
            distanceFunction.update(inst, -1);
            this.learner.delete(position);
        } else {
            System.err.println("Delete fail !");
        }
    }

    /**
     * Insert one instance to case base and update parameters in classifier
     *
     * @param inst
     */
    private void insertInstance(Instance inst) {
        if (this.learner == null) {
            this.learner = new Instances(inst.dataset());
        }
        this.learner.add(inst);
        distanceFunction.update(inst, 1);
        for (int i = 0; i < learner.numInstances(); ++i) {
            if (Math.abs(inst.classValue() - learner.get(i).classValue()) < 1e-2) { // same class
                this.outerVectorProduct(inst, learner.get(i), similar_matrix.getArray(), 1);
                ++similar_size;
            } else {
                this.outerVectorProduct(inst, learner.get(i), dissimilar_matrix.getArray(), 1);
                ++dissimilar_size;
            }
        }
    }

    /**
     * Check if two instances are iquals
     *
     * @param first
     * @param second
     * @return true if they are iquals
     */
    private boolean equalsInstance(Instance first, Instance second) {
        if (first.equalHeaders(second)) {
            double[] first_value = first.toDoubleArray();
            double[] second_value = second.toDoubleArray();
            for (int i = 0; i < first_value.length; ++i) {
                if (i != first.classIndex() && Double.compare(first_value[i], second_value[i]) != 0) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
    int update = 0;

    private void updateDistanceFunction() {
        if (BUG) {
            System.out.println("updating: " + update);
            update++;
        }
        try {
            Matrix S = (this.similar_matrix.copy()).timesEquals(1.0 / similar_size);
            Matrix D = (this.dissimilar_matrix.copy()).timesEquals(1.0 / dissimilar_size);
            
            if (S.det() < 1e-20) {
                regularizedMatrix(S);
            }
            if (D.det() < 1e-20) {
                regularizedMatrix(D);
            }

            Matrix inv_similar = S.inverse();
            Matrix inv_dissimilar = D.inverse();

            for (int i = 0; i < this.mahalanobis_matrix.getRowDimension(); ++i) {
                for (int j = 0; j < this.mahalanobis_matrix.getColumnDimension(); ++j) {
                    this.mahalanobis_matrix.getArray()[i][j] = inv_similar.get(i, j) - inv_dissimilar.get(i, j);
                }
            }

            EigenvalueDecomposition eig = this.mahalanobis_matrix.eig();
            Matrix eigensVectors = eig.getV();

            /* Matrix diagonal contains eigenvalues */
            Matrix diagonalMatrix = eig.getD();

            /* Check if eigenvalue is negative */
            for (int i = 0; i < this.mahalanobis_matrix.getRowDimension(); i++) {
                if (diagonalMatrix.get(i, i) < 1e-10) {
                    diagonalMatrix.getArray()[i][i] = 1e-6;
                }
            }
            this.mahalanobis_matrix = eigensVectors.times(diagonalMatrix.times(eigensVectors.transpose()));
            this.distanceFunction.setMatrix(this.mahalanobis_matrix);

            if (BUG) {
                ((MahalanobisDistance) search.getDistanceFunction()).getMatrix().write(new PrintWriter(System.err));
            }

        } catch (Exception ex) {
            Logger.getLogger(KISSStreams.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private boolean isPositiveMatrix(Matrix m) {
        EigenvalueDecomposition eig = m.eig();
        /* Matrix diagonal contains eigenvalues */
        Matrix diagonalMatrix = eig.getD();
        /* Check if eigenvalue is negative */
        for (int i = 0; i < m.getRowDimension(); i++) {
            if (diagonalMatrix.get(i, i) < 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Outer product of two instance (first, second) = (first - second) * (first
     * - second)'.
     *
     * @param first first instance
     * @param second second instance
     * @param accumulate value accumulate
     * @param constante scale accumulate
     */
    private void outerVectorProduct(Instance first, Instance second, double[][] accumulate, double constante) {
        double[] diff = distanceFunction.difference(first, second);
        for (int i = 0; i < diff.length; i++) {
            for (int j = 0; j < diff.length; j++) {
                accumulate[i][j] += diff[i] * diff[j] * constante;
            }
        }
    }

    /**
     * Regularized matrix singular
     *
     * @param matrix
     */
    private void regularizedMatrix(Matrix matrix) {
        int m_rows = matrix.getRowDimension();
        int m_columns = matrix.getColumnDimension();

        for (int i = 0; i < m_rows; i++) {
            for (int j = 0; j < m_columns; j++) {
                double val = (i == j) ? matrix.getArray()[i][j] : 0;
                matrix.getArray()[i][j] = matrix.getArray()[i][j] * (1 - LAMBDA) - val * LAMBDA;
            }
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<Measurement>();
        measurementList.add(new Measurement("Change detected", this.changeDetected));
        measurementList.add(new Measurement("Warning detected", this.warningDetected));
        this.changeDetected = 0;
        this.warningDetected = 0;
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        
        try {
            
            this.cantInstancias++;
            System.out.println("Classifying the instance: " + this.cantInstancias);
            classCount = inst.numClasses();
            classType = inst.classAttribute().type();
            if (learner == null) {
                return new double[inst.numClasses()];
            }
            if (!first_learned) {
                
                search.setInstances(learner);
            }
            Instances neighbours = search.kNearestNeighbours(inst, KNN); 
            for (int i = 0; i < neighbours.size(); ++i) {
                if (equalsInstance(neighbours.get(i), inst)) {   
                    double[] ans = new double[inst.numClasses()];
                    ans[(int) (neighbours.get(i).classValue() + 1e-2)] = 1;

                    if(Utils.maxIndex(ans)!=(int)(inst.classValue())){ 
                       this.cantMiss=this.cantMiss+1;
                       
                       
                    }
                  
                    return ans;
                }
            }
           
            
            
            
            if(Utils.maxIndex(makeDistribution(neighbours, search.getDistances())) != (int) (inst.classValue() + 1e-2))
            {
                this.cantMiss=this.cantMiss+1;
                
                
                try {
                      BufferedWriter out = new BufferedWriter(new FileWriter("kissstream_correctly_classified_instances.txt", true));
                      out.write(this.cantInstancias+"\n");
                      out.close();
                      
                    } catch (IOException e) {
                      System.out.println("Error writing in the file");   
                    }
                
            }
            
            
           
            return makeDistribution(neighbours, search.getDistances());
        } catch (Exception ex) {
            return new double[inst.numClasses()];
        }
    }

    /**
     * Turn the list of nearest neighbors into a probability distribution.
     *
     * @param neighbours the list of nearest neighboring instances
     * @param distances the distances of the neighbors
     * @return the probability distribution
     * @throws Exception if computation goes wrong or has no class attribute
     */
    protected double[] makeDistribution(Instances neighbours, double[] distances)
            throws Exception {
        double total = 0, weight;
        double[] distributionV = new double[classCount];
        
        if (classType == Attribute.NOMINAL) {
            for (int i = 0; i < classCount; i++) {
                distributionV[i] = 1.0 / Math.max(1, learner.numInstances());
            }
            total = (double) classCount / Math.max(1, learner.numInstances());
        }
        for (int i = 0; i < neighbours.numInstances(); i++) {
            
            Instance current = neighbours.instance(i);
            distances[i] = distances[i] * distances[i];
            distances[i] = Math.sqrt(distances[i] / (learner.numAttributes() - 1));
            switch (m_DistanceWeighting) {
                case WEIGHT_INVERSE:
                    weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
                    break;
                case WEIGHT_SIMILARITY:
                    weight = 1.0 - distances[i];
                    break;
                default:                                 // WEIGHT_NONE:
                    weight = 1.0;
                    break;
            }
            weight *= current.weight();
            try {
                switch (classType) {
                    case Attribute.NOMINAL:
                        distributionV[(int) (current.classValue() + 1e-2)] += weight;
                        break;
                    case Attribute.NUMERIC:
                        distributionV[0] += current.classValue() * weight;
                        break;
                }
            } catch (Exception ex) {
                throw new Error("Data has no class attribute!");
            }
            total += weight;
        }
        
        if (total > 0) {
            Utils.normalize(distributionV, total);
        }
        
        return distributionV;
    }
}
