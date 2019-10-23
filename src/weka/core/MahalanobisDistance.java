
package weka.core;

import weka.core.matrix.Matrix;
import weka.core.neighboursearch.PerformanceStats;

/**
 <!-- globalinfo-start -->
 * Implementing Mahalanobis distance (or similarity) function.<br/>
 * <br/>
 * One object defines not one distance but the data model in which the distances between objects of that data model can be computed.<br/>
 * <br/>
 * Attention: For efficiency reasons the use of consistency checks (like are the data models of the two instances exactly the same), is low.<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Wikipedia. Euclidean distance. URL http://en.wikipedia.org/wiki/Mahalanobis_distance.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;misc{
 *    author = {Wikipedia},
 *    title = {Mahalanobis distance},
 *    URL = {http://en.wikipedia.org/wiki/Mahalanobis_distance}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  Turns off the normalization of attribute 
 *  values in distance calculation.</pre>
 * 
 * <pre> -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to used in the calculation of the 
 *  distance. 'first' and 'last' are valid indices.
 *  (default: first-last)</pre>
 * 
 <!-- options-end --> 
 *
 * @author Jorge Luis Rivero Perez
 * @version $Revision: 1111 $
 */

public class MahalanobisDistance extends NormalizableDistance {
    
    /** for serialization. */
    private static final long serialVersionUID = 1068606253458807909L;    
    
    /** matrix used for calculating distance between two instances */
    protected Matrix mahalanobisMatrix = null;   
    
    /** Distribution of attributes nominal [<class>][<k-attribute>][<k-value>]*/    
    protected double[][][] distribution = null;
    
    protected boolean started = false;
    protected int m_attibutes;
    protected int classIndex;
    protected int classCount;    
    protected int [] indexOfNominal;
    
    
    
    /**
    * Constructs an Mahalanobis Distance object.
    * @param mahalanobisMatrix the matrix of the distance function should work on.
    */
    public MahalanobisDistance(Matrix A) {        
        super();
        this.mahalanobisMatrix = A;
        this.started = false;
    }    
    public MahalanobisDistance() {        
        super();
        this.mahalanobisMatrix = null;
        this.started = false;
    }
    
    
    /**
     * Calculates the distance (or similarity) between two instances. Need to
     * pass this returned distance later on to postprocess method to set it on
     * correct scale. <br/> P.S.: Please don't mix the use of this function with
     * distance(Instance first, Instance second), as that already does post
     * processing. Please consider passing Double.POSITIVE_INFINITY as the
     * cutOffValue to this function and then later on do the post processing on
     * all the distances.
     *
     * @param first the first instance
     * @param second the second instance
     * @param stats the structure for storing performance statistics.
     * @return the distance between the two given instances or
     * Double.POSITIVE_INFINITY.
     */
    @Override
    public double distance(Instance first, Instance second, PerformanceStats stats) { //debug method pls remove after use
        return distance(first, second, Double.POSITIVE_INFINITY, stats);
    }

    /**
     * Calculates the distance between two instances.
     *
     * @param first the first instance
     * @param second the second instance
     * @return the distance between the two given instances
     */
    @Override
    public double distance(Instance first, Instance second) {
        return distance(first, second, Double.POSITIVE_INFINITY);
    }

    /**
     * Calculates the distance between two instances. Offers speed up (if the
     * distance function class in use supports it) in nearest neighbour search
     * by taking into account the cutOff or maximum distance. Depending on the
     * distance function class, post processing of the distances by
     * postProcessDistances(double []) may be required if this function is used.
     *
     * @param first the first instance
     * @param second the second instance
     * @param cutOffValue If the distance being calculated becomes larger than
     * cutOffValue then the rest of the calculation is discarded.
     * @param stats the performance stats object
     * @return the distance between the two given instances or
     * Double.POSITIVE_INFINITY if the distance being calculated becomes larger
     * than cutOffValue.
     */
    @Override
    public double distance(Instance first, Instance second, 
            double cutOffValue, PerformanceStats stats) {             
        double[] diff = difference(first, second);                
        double ret = 0;        
        for( int i = 0; i < diff.length; i ++)
            for( int j = 0; j < diff.length; j ++)
                ret += mahalanobisMatrix.getArray()[i][j] * diff[i] * diff[j];        
        if ( ret < 0)
            System.err.println( "Error of distance function , negative distance "  + ret);
        return ret;
    }
    
  /**
   * Returns a string describing this object.
   * 
   * @return 		a description of the evaluator suitable for
   * 			displaying in the explorer/experimenter gui
   */
    @Override
    public String globalInfo() {
        return 
        "Implementing Mahalanobis distance (or similarity) function.\n\n"
      + "One object defines not one distance but the data model in which "
      + "the distances between objects of that data model can be computed.\n\n"
      + "Attention: For efficiency reasons the use of consistency checks "
      + "(like are the data models of the two instances exactly the same), "
      + "is low.\n\n"
      + "For more information, see:\n\n"
      + getTechnicalInformation().toString();
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        double result;
        result = currDist;
        result += diff * diff;
        return result;
    }
    
    
    public double[] difference(Instance first, Instance second){
        
        int index_class = first.classIndex();
        int m_attributes = first.numAttributes();
        double [] firstValues = first.toDoubleArray();
        double [] secondValues = second.toDoubleArray();       
        double [] diff = new double[m_attributes - 1];
        double p1, p2;
                
        for( int i = 0, k = 0; i < m_attributes; i ++){     
            int size = first.attribute(i).numValues();
           
            if ( i == index_class)
                continue;            
            switch(first.attribute(i).type()){
                case Attribute.NOMINAL:
                    if (Utils.isMissingValue(firstValues[i]) ||
                        Utils.isMissingValue(secondValues[i])){                    
                        diff[k] = 1;
                    }else {
                        for( int j = 0; j < first.numClasses(); ++ j){
                            p1 = distribution[j][indexOfNominal[i]][(int)(firstValues[i] + 1e-2)] * 1.0 / 
                                    Math.max(1, distribution[j][indexOfNominal[i]][size]);
                            
                            

                            
                            
                            p2 = distribution[j][indexOfNominal[i]][(int)(secondValues[i] + 1e-2)] * 1.0 / 
                                    Math.max(1, distribution[j][indexOfNominal[i]][size]);
                            diff[k] += Math.abs(p1 - p2);                         
                        }                                               
                    }                    
                    break;
                case Attribute.NUMERIC:
                    if (Utils.isMissingValue(firstValues[i]) ||
                        Utils.isMissingValue(secondValues[i])) {                                                    
                           diff[k] = 1;                            
                    }else{
                           diff[k] = Math.abs(firstValues[i] - secondValues[i]);
                    }
                    break;
            }
            k ++;
        }          
        return diff;
    }  
    
    /**
     * initial campus of distribution
     * @param inst 
     */     
    private void init(Instance inst){
        int index = 0;
        indexOfNominal = new int[m_attibutes];
        for(int i = 0; i < m_attibutes; ++ i){
                if ( inst.attribute(i).isNominal() ){
                    indexOfNominal[i] = index ++; 
                }
        }
        distribution = new double[classCount][index][];
        for(int i = 0; i < m_attibutes; ++ i){  
                if ( inst.attribute(i).isNominal() ){
                    for( int j = 0; j < classCount; ++ j){
                        int size = inst.attribute(i).numValues();
                        distribution[j][indexOfNominal[i]] = new double[size + 1];
                    }
                }
        }
        started = true;                
    }
    /**
     * update campus of distribution.
     * 
     * @param inst
     * @param val is parameter for insert (val = 1) or remove (val = -1)
     */
    public void update(Instance inst, int val){
        
                
        classIndex = inst.classIndex();
        m_attibutes = inst.numAttributes();
        classCount = inst.numClasses();        
        
        if ( !started ){
            init(inst);
        }
        for( int attr = 0; attr < inst.numAttributes(); ++ attr){                   
            if ( inst.attribute(attr).isNominal()){
                int cls = (int)(inst.classValue() + 1e-2); 
                int index = (int)( inst.value(attr) + 1e-2);

                int pos = indexOfNominal[attr];

                int size = inst.attribute(attr).numValues();

                
                this.distribution[cls][pos][index] += val;
                this.distribution[cls][pos][size] += val;
            }
        }                
    }
    /**
    * Returns the revision string.
    * 
    * @return		the revision
    */
    @Override
    public String getRevision() {
        return "Revision 1111 $";
    }
    /**
    * Returns an instance of a TechnicalInformation object, containing 
    * detailed information about the technical background of this class,
    * e.g., paper reference or book this class is based on.
    * 
    * @return 		the technical information about this class
    */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;

        result = new TechnicalInformation(TechnicalInformation.Type.MISC);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Wikipedia");
        result.setValue(TechnicalInformation.Field.TITLE, "Mahalanobis distance");
        result.setValue(TechnicalInformation.Field.URL, "http://en.wikipedia.org/wiki/Mahalanobis_distance");

        return result;
    } 
    /**
     * Set value for matrix mahalanobis
     * @param mahalanobisMatrix matrix of distance
     */
    public void setMatrix(Matrix mahalanobisMatrix){
        this.mahalanobisMatrix = mahalanobisMatrix;
    }
    public Matrix getMatrix(){
        return this.mahalanobisMatrix;
    }
    public void setAttributeDistribution(double[][][] distribution){
        this.distribution = distribution;
    }    
    
    public double getDistributionClass( Instance inst){        
        int cls =(int)( inst.classValue() + 1e-2 );
        double total = 0;
        for( int i = 0; i < classCount; ++ i){            
            total += distribution[i][indexOfNominal[classIndex]][inst.numClasses()];            
        }
        return distribution[cls][indexOfNominal[classIndex]][inst.numClasses()] / Math.max(1, total);
    }
}
