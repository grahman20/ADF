/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.tasks;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.LinkedList;
import java.util.StringTokenizer;
import java.util.logging.Level;
import java.util.logging.Logger;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.learners.Learner;
import moa.options.ClassOption;
import moa.streams.ExampleStream;
import moa.streams.InstanceStream;
import static moa.tasks.MainTask.INSTANCES_BETWEEN_MONITOR_UPDATES;
import moa.streams.ArffFileStream;
//import moa.classifiers.meta.AdaptiveDecisionForestSysFor;
//import moa.classifiers.meta.AdaptiveDecisionForestHT;
//import moa.classifiers.meta.AdaptiveDecisionForestRF;
import moa.classifiers.meta.AdaptiveDecisionForest;
import moa.core.TimingUtils;

/**
 *
 * @author grahman
 */
public class EvaluateModelsForBatchData extends ClassificationMainTask implements CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Evaluation of models of Incremental Learning for Batch training and testing Data by Rahman and Islam (2020).";
    }

    private static final long serialVersionUID = 1L;
//    public ClassOption modelOption = new ClassOption("treeLearner", 'l',
//            "Decision Forest Algorithm.", MultiClassClassifier.class,
//            "moa.classifiers.meta.WEKASysFor");
    public ClassOption modelOption = new ClassOption("ChooseAClassifier", 'M',
            "Classifier to train.", MultiClassClassifier.class, "moa.classifiers.meta.AdaptiveDecisionForest");

//    public ClassOption batchOption = new ClassOption("ChooseBatchTraningAndTestData", 's',
//            "Batch data to learn from.", ExampleStream.class,
//            "TrainingAndTestBatchFiles");
    public FileOption trainArffOption = new FileOption("ChooseBatchLogFile", 'D',
            "Log contains training and testing batch files.",
            null,"LogBatchFiles", true);
//    public FileOption testArffOption = new FileOption("ChooseTestArffFile", 'd',
//            "Training data to learn from.",
//            null,"TrainingBatchFile", true);
    public IntOption classIndexOption = new IntOption("classIndex", 'I',
            "Class index of data. 0 for none or -1 for last attribute in file.",
            -1, -1, Integer.MAX_VALUE);
//    public ClassOption evaluatorOption = new ClassOption("ChooseAnEvaluator", 'e',
//            "Classification performance evaluation method.",
//            LearningPerformanceEvaluator.class,
//            "BasicClassificationPerformanceEvaluator");
//    public FileOption outputPredictionFileOption = new FileOption("SelectOutputPredictionFile", 'p',
//            "File to append output classifier to.", null, "pred", true);
    
//    private String status;
//    private long parallelMaxTime;
//    private long totalTime;
//    private long []exetime;//=new long[4];
//    private float []trainAccuracy;//=new float[4];
    
    public EvaluateModelsForBatchData() {
    }

    public EvaluateModelsForBatchData(Classifier model, InstanceStream stream,
            LearningPerformanceEvaluator evaluator, int maxInstances) {
        this.modelOption.setCurrentObject(model);
//        this.batchOption.setCurrentObject(stream);
//        this.evaluatorOption.setCurrentObject(evaluator);
//        this.maxInstancesOption.setValue(maxInstances);
    }

    @Override
    public Class<?> getTaskResultType() {
        return LearningEvaluation.class;
    }

//    @Override
//    public Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
//        Learner model = (Learner) getPreparedClassOption(this.modelOption);
//        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.batchOption);
//        
//        LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
//        LearningCurve learningCurve = new LearningCurve("learning evaluation instances");
//        int maxInstances =1000000;// this.maxInstancesOption.getValue();
//        long instancesProcessed = 0;
//        monitor.setCurrentActivity("Evaluating model...", -1.0);
//
//        //File for output predictions
//        File outputPredictionFile = this.outputPredictionFileOption.getFile();
//        PrintStream outputPredictionResultStream = null;
//        if (outputPredictionFile != null) {
//            try {
//                if (outputPredictionFile.exists()) {
//                    outputPredictionResultStream = new PrintStream(
//                            new FileOutputStream(outputPredictionFile, true), true);
//                } else {
//                    outputPredictionResultStream = new PrintStream(
//                            new FileOutputStream(outputPredictionFile), true);
//                }
//            } catch (Exception ex) {
//                throw new RuntimeException(
//                        "Unable to open prediction result file: " + outputPredictionFile, ex);
//            }
//        }
//        while (stream.hasMoreInstances()
//                && ((maxInstances < 0) || (instancesProcessed < maxInstances))) {
//            Example testInst = (Example) stream.nextInstance();//.copy();
//            int trueClass = (int) ((Instance) testInst.getData()).classValue();
//            //testInst.setClassMissing();
//            double[] prediction = model.getVotesForInstance(testInst);
//            //evaluator.addClassificationAttempt(trueClass, prediction, testInst
//            //		.weight());
//            if (outputPredictionFile != null) {
//                outputPredictionResultStream.println(Utils.maxIndex(prediction) + "," +(
//                        ((Instance) testInst.getData()).classIsMissing() == true ? " ? " : trueClass));
//            }
//            evaluator.addResult(testInst, prediction);
//            instancesProcessed++;
//
//            if (stream.hasMoreInstances() == false) {
//	            learningCurve.insertEntry(new LearningEvaluation(
//	                    new Measurement[]{
//	                        new Measurement(
//	                        "learning evaluation instances",
//	                        instancesProcessed)
//	                    },
//	                    evaluator, model));
//            }
//            if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
//                if (monitor.taskShouldAbort()) {
//                    return null;
//                }
//                long estimatedRemainingInstances = stream.estimatedRemainingInstances();
//                if (maxInstances > 0) {
//                    long maxRemaining = maxInstances - instancesProcessed;
//                    if ((estimatedRemainingInstances < 0)
//                            || (maxRemaining < estimatedRemainingInstances)) {
//                        estimatedRemainingInstances = maxRemaining;
//                    }
//                }
//                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
//                        : (double) instancesProcessed
//                        / (double) (instancesProcessed + estimatedRemainingInstances));
//                if (monitor.resultPreviewRequested()) {
//                    monitor.setLatestResultPreview(learningCurve.copy());
//                }
//            }
//        }
//        if (outputPredictionResultStream != null) {
//            outputPredictionResultStream.close();
//        }
//        return learningCurve;
//    }
   
   @Override
    public Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
         Classifier model = (Classifier) getPreparedClassOption(this.modelOption);
         File logFile = this.trainArffOption.getFile();
         int classIndex = this.classIndexOption.getValue();
         String modelName ="";
         String []options=null;
         try{
         options=moa.core.Utils.splitOptions(modelOption.getValueAsCLIString());
         modelName = options[0];
         }
         catch(Exception e)
         {
             System.out.println("Error: "+e.toString());
         }
         
         System.out.println("Model options: "+this.modelOption.getValueAsCLIString());
         System.out.println("Log file: "+logFile.getAbsolutePath());
         if(modelName.equals("meta.AdaptiveDecisionForest"))
         {
         AdaptiveDecisionForest adf=new AdaptiveDecisionForest();
         adf.learnFromBatchDataset(logFile.getAbsolutePath(), classIndex, options,"");
         }
         else{
         String [][]bFile=readFileAs2DArray(logFile);
         String header="Batch, Accuracy, ExecutionTime";
//         if(modelName.equals("meta.AdaptiveDecisionForestRF") || modelName.equals("meta.AdaptiveDecisionForestHT"))
//         {
//             header="Batch, TestAccuracy, TotalTime, PFAccuracy, PFTime, AFAccuracy,AFTime, TFAccuracy, TFTIme, Staus";
//         }
         String accuracyFile=changedFileName(logFile.getAbsolutePath(),"_accuracy");
         File outF=new File(accuracyFile);
         writeToFile(outF, header);
         String path=logFile.getParent();
         int noB=bFile.length;
         boolean isFB;
         String batchStatus="";
         for(int i=0;i<noB;i++)
            {
                isFB = i==0;
                String trainFile=path+"\\"+bFile[i][0];
                String learnerFile=changedFileExtension(changedFileName(trainFile,"_finalclassifier"),"txt");
                String testFile=path+"\\"+bFile[i][1];
                System.out.println("Processing file: "+trainFile);
                long time=0;
                float accuracy=0.0f;
//                if(modelName.equals("meta.AdaptiveDecisionForest"))
//                {
//                    time=buildADFClassifier(isFB,trainFile,classIndex,learnerFile,options,0);
//                    accuracy=adfAccuracy(learnerFile,testFile);
//                }
//                else if(modelName.equals("meta.AdaptiveDecisionForestHT"))
//                {
//                    time=buildADFClassifier(isFB,trainFile,classIndex,learnerFile,options,2);                                       
//                    accuracy=adfRFAccuracy(learnerFile,testFile,classIndex,2);
//                    time=totalTime;
//                    batchStatus=", "+trainAccuracy[3]+","+exetime[3]+", "
//                            +trainAccuracy[1]+","+exetime[1]+", "
//                            +trainAccuracy[2]+","+exetime[2]+", "+status;
//                }
//                else if(modelName.equals("meta.AdaptiveDecisionForestRF"))
//                {
//                    time=buildADFClassifier(isFB,trainFile,classIndex,learnerFile,options,1);                                       
//                    accuracy=adfRFAccuracy(learnerFile,testFile,classIndex,1);
//                    time=totalTime;
//                    batchStatus=", "+trainAccuracy[3]+","+exetime[3]+", "
//                            +trainAccuracy[1]+","+exetime[1]+", "
//                            +trainAccuracy[2]+","+exetime[2]+", "+status;
//                }
//                else{
                     long sTime=0, eTime=0;
                     sTime = System.currentTimeMillis();
                     ArffFileStream trainArffs=new ArffFileStream(trainFile,classIndex);
                     model=buildClassifier(isFB,model,trainArffs);
                     writeToFile(new File(learnerFile),model.toString());
                     ArffFileStream testArffs=new ArffFileStream(testFile,classIndex);
                     accuracy=calculateAccuracy(model, testArffs);
                     eTime = System.currentTimeMillis();
                     time =eTime-sTime;
//                }
                System.out.println("Accuracy: "+accuracy+", time: "+time);
                NumberFormat formatter = new DecimalFormat("#0.000");  
                String acc="\n"+bFile[i][0]+", "+formatter.format(accuracy)+"%,"+time+batchStatus;
                appendToFile(outF, acc);
            }
         }
         LearningCurve learningCurve = new LearningCurve("learning evaluation instances");
         return learningCurve;
    }
   public Classifier buildClassifier(boolean isFB,Classifier learner, ArffFileStream trainArffs)
   {
       trainArffs.prepareForUse();
       if(isFB){
       learner.setModelContext(trainArffs.getHeader());
       learner.prepareForUse();
       }
       while (trainArffs.hasMoreInstances()) {
           Instance trainInst = trainArffs.nextInstance().getData();
           learner.trainOnInstance(trainInst);
         }       
       return learner;
   }
    public float calculateAccuracy(Classifier learner, ArffFileStream testArffs)
   {
       int numberSamplesCorrect = 0;
       int numberSamples = 0;
       while (testArffs.hasMoreInstances()) 
       {
            Instance testInst = testArffs.nextInstance().getData();
            if (learner.correctlyClassifies(testInst)){
                    numberSamplesCorrect++;
            }
            numberSamples++;

        }
        float accuracy = 100.0f * (float) numberSamplesCorrect/ (float) numberSamples;
        return accuracy;
   } 
//    public long buildADFClassifier(boolean isFB, String trainArffs, int classIndex, String outputFile, String []options, int adfTask)
//   {
//       long t=0;
//       if(adfTask==0)
//       {
//           AdaptiveDecisionForestSysFor adf=new AdaptiveDecisionForestSysFor();
//           t=adf.learnFromDataSet(isFB, trainArffs, classIndex,outputFile,options);
//       }
//       else if(adfTask==1){
//            AdaptiveDecisionForestRF adfrf=new AdaptiveDecisionForestRF();
//            t=adfrf.learnFromDataSet(isFB, trainArffs, classIndex,outputFile,options);
//            status=adfrf.getStatus();
//            totalTime=adfrf.getTotalTime();
//            exetime=adfrf.getTime();
//            trainAccuracy=adfrf.getAccuracy();
//       }
//       else{
//            AdaptiveDecisionForestHT adfht=new AdaptiveDecisionForestHT();
//            t=adfht.learnFromDataSet(isFB, trainArffs, classIndex,outputFile,options);
//            status=adfht.getStatus();
//            totalTime=adfht.getTotalTime();
//            exetime=adfht.getTime();
//            trainAccuracy=adfht.getAccuracy();
//       }
//       return t;
//   }
//   
//   public float adfAccuracy(String treeFile, String testDataFile)
//   {
//       AdaptiveDecisionForestSysFor adf=new AdaptiveDecisionForestSysFor(); 
//       return adf.calculateAccuracy(treeFile, testDataFile);
//   } 
//   
//   public float adfRFAccuracy(String treeFile, String testDataFile, int classIndex, int adfTask)
//   {
//       if(adfTask==1)
//       {
//        AdaptiveDecisionForestRF adf_rf=new AdaptiveDecisionForestRF(); 
//        return adf_rf.calculateAccuracy(treeFile, testDataFile,classIndex);
//       }
//       else
//       {
//        AdaptiveDecisionForestHT adf_ht=new AdaptiveDecisionForestHT(); 
//        return adf_ht.calculateAccuracy(treeFile, testDataFile,classIndex);
//       }
//   } 
   
    /**
     * Reads the contents of the file to a <code>String</code> array, where
     * each line of the file is in an array cell. In the event of an error
     * the array will be null.
     *
     * @param file file to be read, assumes full path details
     * @return each attribute value of the file in new cell in the array
     */
    public String [][] readFileAs2DArray(File file)
    {
        /** on reading the file each line is temporarily stored in a
         * LinkedList to determine the size of the array, then elements of the
         * LinkedList are copied to the String array.
         *
         * O(n), where n is number of lines in the file.
         */
        LinkedList <String> tempList = new LinkedList<String>();
        String [][] fileArr=null;
        int totalAttr=0;
        StringTokenizer tokenizer;
        String currLine;
        /** open the file and simply add each new line to the end of the list
         */
        try {
            FileReader fr = new FileReader(file);
            BufferedReader inFile = new BufferedReader(fr);
            currLine = inFile.readLine();
            tokenizer= new StringTokenizer(currLine, " ,\t\n\r\f");
            totalAttr=tokenizer.countTokens();
            //keep reading until file is empty
            while(currLine!=null)
            {
                tempList.add(currLine);
                currLine = inFile.readLine();
            }
            inFile.close();
        } catch (FileNotFoundException e) {
            String error="Error: The File " + file + " was not found, " + e;

        } catch (IOException e) {
            String error = "Error: IO Exception occured " + e;

        }
        /** now copy the List elements over to the array. */
        int listSize = tempList.size();
        if(listSize>0)
        {
            fileArr = new String[listSize][totalAttr];
            for(int i=0; i<listSize; i++)
            {
                currLine=tempList.removeFirst();
                tokenizer= new StringTokenizer(currLine, " ,\t\n\r\f");
                for(int j=0;j<totalAttr;j++)
                    fileArr[i][j] = tokenizer.nextToken();
            }
        }
        return fileArr;
    }
    public String changedFileExtension(String filename, String Ext)
    {
        String returnStr = "";
       /** simply rename a filename*/
        File outFile = new File(filename);
        String outFilePath = outFile.getPath();
        int indexOfDot = outFilePath.lastIndexOf("."); //to get position of file extension
        String pathBeforeExtension = outFilePath.substring(0,indexOfDot);
        returnStr = pathBeforeExtension +"."+Ext;
        return returnStr;
    }
   /**
     * This method will be used to change a file name by a supplied padding .

     *Author Geaur
     * @param filename file name to be changed
     * @param padding to add at the right of the filename
     * @return a changed file name, or an error message if there is a
     * problem writing to file
     */
    public String changedFileName(String filename, String padding)
    {
        String returnStr = "";
       /** simply rename a filename*/
        File outFile = new File(filename);
        String outFilePath = outFile.getPath();
        int indexOfDot = outFilePath.lastIndexOf("."); //to get position of file extension
        String pathBeforeExtension = outFilePath.substring(0,indexOfDot);
        String pathAfterExtension = outFilePath.substring(indexOfDot+1,outFilePath.length());
        returnStr = pathBeforeExtension + padding+"."+pathAfterExtension;
        return returnStr;
    }
    /**
     * Write output <code>String</code> to the passed <code>File</code>. This
     * method will replace any current contents of the file rather than append.
     *
     * @param file output file the contents are being written to
     * @param output the new contents of the file
     * @return a message indicating success, or an error message if there is a
     * problem writing to file
     */
    public String writeToFile(File file, String output)
    {
       /** simply open a file writer and write contents to the file */
       String returnStr = "";
       try{
           FileWriter fileWriter = new FileWriter(file);
           fileWriter.write(output);
           fileWriter.flush();
           fileWriter.close();
           returnStr= "output sucessfully written to " + file.toString();
       }
       catch(IOException ex)
       {
           returnStr="Error: IO Exception occured " + ex;

       }
       return returnStr;
    }
    /**
     * Append output <code>String</code> to the passed <code>File</code>. This
     * method will retain any current contents of the file and then append the
     * output text to the end of the file.
     *
     * @param file output file the contents are being written to
     * @param output the new contents to be added to the file
     * @return a message indicating success, or an error message if there is a
     * problem writing to file
     */
    public String appendToFile(File file, String output)
    {
       /** simply open a file writer and write contents to the file */
       String returnStr = "";
       try{
           FileWriter fileWriter = new FileWriter(file,true);
           fileWriter.write(output);
           fileWriter.flush();
           fileWriter.close();
           returnStr= "output sucessfully append to " + file.toString();
       }
       catch(IOException ex)
       {
           returnStr="Error: IO Exception occured " + ex;
       }
       return returnStr;
    }
    
    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == EvaluateModelsForBatchData.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
