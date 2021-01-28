/*
 *    AdaptiveDecisionForest.java
 *    Copyright (C) 2020 Charles Sturt University, Bathurst, NSW, Australia
 *    @author Md Geaur Rahman (grahman@csu.edu.au)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import moa.capabilities.CapabilitiesHandler;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.WEKAClassOption;
import moa.streams.ArffFileStream;
import weka.classifiers.Classifier;
/**
 *
 * @author Md Geaur Rahman
 */
public class AdaptiveDecisionForest extends AbstractClassifier implements MultiClassClassifier,
                                                                        CapabilitiesHandler{
    @Override
    public String getPurposeString() {
        return "Adaptive Decision Forest (ADF) algorithm for Incremental Learning on Batch Data by Rahman and Islam (2020).";
    }
    
    private static final long serialVersionUID = 1L;
    
//        public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'F',
//            "Decision Forest Algorithm.", MultiClassClassifier.class,
//            "moa.classifiers.meta.WekaRF");
        public MultiChoiceOption treeLearnerOption = new MultiChoiceOption("baseLearner", 'F', 
        "choose an algorithm.",
        new String[]{"RF", "SysFor", "HT"},
        new String[]{"RF", "SysFor", "HT"}, 0);
        
        public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'E',
            "The number of trees.", 1, 1, Integer.MAX_VALUE);        
        public IntOption minRecordsOption = new IntOption("minimumRecords", 'm',
            "Minimum number of records.", 100, 1, Integer.MAX_VALUE);

        public FloatOption repairOption = new FloatOption("repairableThreshold", 'R',
            "The repairable threshold. Default value is 0.4", 0.2, 0.0, 1.0);
        public FloatOption pertubedETOption = new FloatOption("pertubedETThreshold", 'B',
            "Error tolerance for identifying a perturbed leaf. Default value is 0.02", 0.01, 0.00, 0.10);
        public IntOption windowSizeOption = new IntOption("windowSize", 'Z',
            "The number of batches training data that are saved in the window. default value is 3", 3, 1, Integer.MAX_VALUE);
        public IntOption conceptDriftOption = new IntOption("conceptDriftThreshold", 'D',
            "The concept drift threshold. default value is 3", 3, 1, Integer.MAX_VALUE);

        protected BasicClassificationPerformanceEvaluator evaluator;
        private int minRecords;
        private int cdfThreshold;
        private float repairableThreshold;    
        private float errorTolerance; 
        private int windowThreshold;
        private int ensembleSize;
        private int method;
        @Override
    
        public void resetLearningImpl() {
            // Reset attributes
            this.minRecords=this.minRecordsOption.getValue();
            this.ensembleSize=this.ensembleSizeOption.getValue();
            this.cdfThreshold=this.conceptDriftOption.getValue();
            this.repairableThreshold=(float)this.repairOption.getValue();
            this.windowThreshold=this.windowSizeOption.getValue();
            this.errorTolerance=(float)this.pertubedETOption.getValue();
        }

        @Override
        public void trainOnInstanceImpl(Instance instance) {

        }

        @Override
        public double[] getVotesForInstance(Instance instance) {
            DoubleVector combinedVote = new DoubleVector();
            return combinedVote.getArrayRef();
        }

        @Override
        public boolean isRandomizable() {
            return true;
        }

        @Override
        public void getModelDescription(StringBuilder arg0, int arg1) {
        }

        @Override
        protected Measurement[] getModelMeasurementsImpl() {
            return null;
        }

        @Override
        public ImmutableCapabilities defineImmutableCapabilities() {
            if (this.getClass() == AdaptiveDecisionForest.class)
                return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
            else
                return new ImmutableCapabilities(Capability.VIEW_STANDARD);
        }
        
    public void learnFromBatchDataset(String batchFiles,int classIndex, String []options, String padding)
    {
       setOptions(options);
       String md=modelDescription();
       if(padding.equals("")) padding=getMethodName();
       File logFile=new File(batchFiles);
       String [][]bFile=ForestFunctions.readFileAs2DArray(logFile);
       String header="Batch, TestAccuracy, TotalTime, PFAccuracy, PFTime, AFAccuracy,AFTime, TFAccuracy, TFTime, Staus";       
       String accuracyFile=ForestFunctions.changedFileName(ForestFunctions.changedFileExtension(logFile.getAbsolutePath(),"csv"),"_accuracy"+padding);       
       File outF=new File(accuracyFile);
       ForestFunctions.writeToFile(outF, md+"\n"+header);
       String path=logFile.getParent();       
       int noB=bFile.length;
       String batchStatus="";
       ADFLearner adfl=new ADFLearner(this.ensembleSize,this.minRecords,classIndex,this.method
       ,this.cdfThreshold,this.repairableThreshold,this.errorTolerance,this.windowThreshold);                
       for(int i=0;i<noB;i++)
            {
                String trainFile=path+"\\"+bFile[i][0];
                String learnerFile=ForestFunctions.changedFileExtension(ForestFunctions.changedFileName(trainFile,"_finalclassifier"),"txt");
                String testFile=path+"\\"+bFile[i][1];
                System.out.println("\n\nProcessing file: "+trainFile);
                long time=0;
                double accuracy=0.0;
                long sTime=0, eTime=0;
                sTime = System.currentTimeMillis();                
                adfl.learnClassifiers(trainFile);
//                adfl.displayAllForests();
                Forest forest= adfl.getClassifier();  
                String f=forest.toString(); 
                ForestFunctions.writeToFile(new File(learnerFile), f);    
                eTime = System.currentTimeMillis();
                time =eTime-sTime;
                if(f.equals(""))
                {
                     System.out.println("No forest is built");
                }
                else{

                    accuracy=forest.forestAccuracy(testFile,classIndex);
                }                
                long []trainTime=adfl.getExeTime();
                float []trainAccuracy=adfl.getAccuracy();
                time=adfl.getTotalTime();
                batchStatus=", "+trainAccuracy[1]+","+trainTime[1]+", "
                            +trainAccuracy[2]+","+trainTime[2]+", "
                            +trainAccuracy[3]+","+trainTime[3]+", "+adfl.getStatus();
                
                NumberFormat formatter = new DecimalFormat("#0.000");  
                String acc="\n"+bFile[i][0]+", "+formatter.format(accuracy)+"%,"+time+batchStatus;
                System.out.println(acc);
                ForestFunctions.appendToFile(outF, acc);
            }
    }   
    
    public String modelDescription()
    {
        String md="Base classifier: ";
        md+=getMethodName();
        md+=", Ensemble size: "+this.ensembleSize;
        md+=", Min leaf size: "+this.minRecords;
        md+=", Repairable threshold: "+this.repairableThreshold;
        md+=", Error tolerance threshold: "+this.errorTolerance;
        md+=", SCD threshold: "+this.cdfThreshold;
        md+=", Window threshold: "+this.windowThreshold;
        return md;
    }
    public String getMethodName()
    {
        if(method==1)
        {
            return "ADF-SysFor";
        }
        else if(method==3)
        {
            return "ADF-HT";
        }
        else
        {
            return "ADF-RF";
        }        
    }
    
    public void setOptions(String []options)
    {
      this.cdfThreshold=this.conceptDriftOption.getValue();
      this.repairableThreshold=(float)this.repairOption.getValue();
      this.errorTolerance=(float)this.pertubedETOption.getValue();
      this.windowThreshold=this.windowSizeOption.getValue();
      this.minRecords=this.minRecordsOption.getValue();
      this.ensembleSize=this.ensembleSizeOption.getValue();
      this.method=2;
      //reset the parameters with current settings
      for(int i=0;i<options.length;i++)  
      {
//          if(options[i].equals("-T"))
//            {i++;this.method=Integer.parseInt(options[i]);}
          if(options[i].equals("-D"))
            {i++;this.cdfThreshold=Integer.parseInt(options[i]);}
          if(options[i].equals("-R"))
            {i++;this.repairableThreshold=(float)Float.parseFloat(options[i]);}     
          if(options[i].equals("-B"))
            {i++;this.errorTolerance=(float)Float.parseFloat(options[i]);}
          if(options[i].equals("-Z"))
            {i++;this.windowThreshold=Integer.parseInt(options[i]);}
          if(options[i].equals("-E"))
            {i++;this.ensembleSize=Integer.parseInt(options[i]);}
          if(options[i].equals("-m"))
            {i++;this.minRecords=Integer.parseInt(options[i]);}
          if(options[i].equals("-F"))
            {i++;
            if(options[i].equals("SysFor"))
                this.method=1;
            else if(options[i].equals("HT"))
                this.method=3;
            else
                this.method=2;
            }
      }
      
      
    }
    public void display(String tree)
    {
        System.out.println(tree);
    }
    private class ADFLearner{
        private int cdf;
        private boolean isPF_Update=true;
        private int cdfThreshold;
        private float repairableThreshold;    
        private float errorTolerance; 
        private int windowThreshold;
        private int numTree;
        private int minLeafSize;
        private int classIndex;
        private int method;        
        private String dataFile;
        private String windowFile;
        private DatasetStats dss;
        private ArffFileStream ARFFdataFile;
        private Forest PF=new Forest();
        private Forest AF=new Forest();
        private Forest TF=new Forest();
        private Forest bestF=new Forest();
        private float []accuracy=new float[4];
        private long []exeTime=new long[4];        
        private String status;
        private boolean isBuiltTF=false;
        ADFLearner(int numTree, int minLeafSize,int classIndex,int method
        ,int cdfThreshold,float repairableThreshold,float errorTolerance,int windowThreshold)
        {
            this.numTree=numTree;
            this.minLeafSize=minLeafSize;
            this.classIndex=classIndex;
            this.method=method;    
            this.cdfThreshold=cdfThreshold;
            this.repairableThreshold=repairableThreshold;
            this.errorTolerance=errorTolerance;       
            this.windowThreshold=windowThreshold;
            this.cdf=0;
            windowFile="";
            status="";
            dss=new DatasetStats();
        }
        public long[]getExeTime()
        {
            return this.exeTime;
        }
        public long getMaxExeTime()
        {
            return this.exeTime[0];
        }
        public long getTotalTime()
        {
            long t=0;
            for(int i=1;i<4;i++){
                if(exeTime[i]>0)                
                {
                    if((i==2 && isBuiltTF)||i!=2)
                        t+=exeTime[i];
                }
            }
            return t;
        }
        public float[]getAccuracy()
        {
            return this.accuracy;
        }
        public String getStatus()
        {
            return status;
        }
        public void learnClassifiers(String dataFile)
        {
            long sTime=0, eTime=0;
            status="";
            for(int i=0;i<4;i++)
            {
                exeTime[i]=-1;
                accuracy[i]=0;
            }
            this.dataFile=dataFile;
            ARFFdataFile=new ArffFileStream(this.dataFile,this.classIndex);             
            dss.processARFFDataFile(ARFFdataFile);
            String [][]bData=dss.getData();
            int cIndex=dss.getClassIndex();
            if(PF.getForestSize()==0)
            {   status+="(PFNE-BPF-CAF)-";
                updateWindow(true);
                sTime = System.currentTimeMillis();             
                PF.buildForest(ARFFdataFile, dss, numTree, minLeafSize, method);
                PF.setClassValues(dss.getClassValues());
                eTime = System.currentTimeMillis();
                exeTime[1]=eTime-sTime;
                sTime = System.currentTimeMillis();
                AF.constructRuleToForest(PF.toString(), dss);
                AF.setClassValues(dss.getClassValues());
                eTime = System.currentTimeMillis();
                exeTime[2]=eTime-sTime;
            }
            else{ 
                updateWindow(false);
                if(isPF_Update)
                {
                    sTime = System.currentTimeMillis();
                    boolean ret=repairForest(PF,"PF");
                    eTime = System.currentTimeMillis();
                    exeTime[1]=eTime-sTime;
                }
                sTime = System.currentTimeMillis();
                boolean isAFRepairable=repairForest(AF,"AF");
                eTime = System.currentTimeMillis();
                exeTime[2]=eTime-sTime;
                if(isAFRepairable)
                {
                    this.cdf=0; 
                    TF=new Forest();
                }
                else    
                {
                    isBuiltTF=true;
                    this.cdf++;
                    if(TF.getForestSize()==0)
                    {   status+="(TFNE-TFW)-";
                        createWindowFile();                        
                        TF=new Forest();
                        sTime = System.currentTimeMillis();     
                        TF.buildForest(ARFFdataFile, dss, numTree, minLeafSize, method);
                        TF.setClassValues(dss.getClassValues());
                        eTime = System.currentTimeMillis();
                        exeTime[3]=eTime-sTime;
                    }
                    else{  
                        sTime = System.currentTimeMillis();                        
                        boolean isTFRepairable=repairForest(TF,"TF");
                        eTime = System.currentTimeMillis();
                        exeTime[3]=eTime-sTime;
                        if(isTFRepairable==false)
                        {
                            status+="(BTFW)-";
                            createWindowFile();
                            TF=new Forest();
                            sTime = System.currentTimeMillis();             
                            TF.buildForest(ARFFdataFile, dss, numTree, minLeafSize, method);
                            TF.setClassValues(dss.getClassValues());
                            eTime = System.currentTimeMillis();
                            exeTime[3]=eTime-sTime;
                            if(this.cdf>this.cdfThreshold)
                            {
                                this.cdf=0; 
                                AF=new Forest();
                                AF.constructRuleToForest(TF.toString(), dss);
                                AF.setClassValues(dss.getClassValues());
                                TF=new Forest();
                                status+="(CDF-AF=TF)-";
                            }
                        }
                    }                    
                }
            }            
            findBestClassifier(bData,cIndex);
        }
        public boolean repairForest(Forest F,String msg)
        {
            boolean isRepairable=true;
            F.setClassValues(dss.getClassValues());
            int totalTree=F.getForestSize();
            int[]treePerturbed=new int[totalTree];
            isRepairable=isForestRepairable(F,treePerturbed,msg);
            String []tClassValues=ClassObserver.updateClassValues(F.getClassValues(), dss.getClassValues());  
            int ncv=ClassObserver.getNumNewCV();
            if(isRepairable ||(msg.equals("PF")&&isPF_Update) || ncv>0){
                ISAT isat=new ISAT();
                isat.expandForestByISAT(F, dss, this.dataFile, minLeafSize, method,
                        this.classIndex,ARFFdataFile,treePerturbed,ncv);                            
                isRepairable=true;
                status+="("+msg+"-isat)-";
            }
            return isRepairable;
        }
        public boolean isForestRepairable(Forest F, int[]treePerturbed,String msg)
        {
            boolean isRepairable=true;
            int totalTree=F.getForestSize();
            int[]leaves=new int[totalTree];            
            int totalPerturbed=0;
            int totalLeaves=0;
            int maxTreeSize=0;
            List<Tree> Trees=F.getForest();
            for(int i=0;i<totalTree;i++)
            {
                Tree T=Trees.get(i);
                treePerturbed[i]=T.identifyPerturbedLeaves(dss, (double)errorTolerance);
                totalPerturbed+=treePerturbed[i];
                leaves[i]=T.getTotalLeafCount();
                totalLeaves+=leaves[i];      
                if(leaves[i]>maxTreeSize)maxTreeSize=leaves[i];
            }
            float perturbedRatio=0.0f;
            if(totalLeaves>0)
            {
                perturbedRatio=(float)totalPerturbed/(float)totalLeaves;
            }
            System.out.println(msg+": Total perturbed leaves: "+totalPerturbed+",  total leaves: "+totalLeaves+
                    ", perturbed ratio: "+perturbedRatio+" and max tree size: "+maxTreeSize);
            if(perturbedRatio>repairableThreshold)
            {
                isRepairable=false;
            }
            return isRepairable;
        }
        public void findBestClassifier(String [][]bData,int cIndex)
        {
            calculateAccuracy(bData,cIndex);
            float maxAcc=0.0f;
            int maxIndex=1;
            for(int i=1;i<accuracy.length;i++)
            {
                if(accuracy[i]>maxAcc)
                {
                    maxAcc=accuracy[i];
                    maxIndex=i;
                }
            }
            accuracy[0]=maxAcc;
            if(maxIndex==1)
            {
                bestF=PF;status+="(PF)";
            }
            else if(maxIndex==2)
            {
                bestF=AF;status+="(AF)";
            }
            else if(maxIndex==3)
            {
               bestF=TF;status+="(TF)";
            }
        }
        private void updateWindow(boolean isFirstBatch)
        {
            if(this.windowFile.equals("")||isFirstBatch)
            {
                windowFile=ForestFunctions.changetoNewFileName(dataFile,"\\"+dss.getDataSetName()+"_window.txt");
                ForestFunctions.writeToFile(new File(this.windowFile), this.dataFile+"\n");
            }
            else
            {
                String []w=ForestFunctions.readFileAsArray(new File(this.windowFile));
                if(w.length<windowThreshold)
                {
                    ForestFunctions.appendToFile(new File(this.windowFile), this.dataFile+"\n");
                }
                else
                {
                    String rec="";
                    for(int i=1;i<w.length;i++)
                    {
                        rec=rec+w[i]+"\n";
                    }
                    rec=rec+dataFile+"\n";
                    ForestFunctions.writeToFile(new File(this.windowFile), rec);
                }
            }       
        }
        
        private void createWindowFile()
        {   
            List<String[]> recordList = new ArrayList<String[]>();            
            String []wFile=ForestFunctions.readFileAsArray(new File(windowFile));
            for(int i=0;i<wFile.length;i++)
            {
               String []tmpdata =ForestFunctions.readFileAsArray(new File(wFile[i]));
               for(int j=0;j<tmpdata.length;j++)
               {
                if(tmpdata[j].equals("@data"))  
                {
                    for(j=j+1;j<tmpdata.length;j++)
                     {
                        if(!tmpdata[j].equals(""))  
                        {
                           String []rec=tmpdata[j].split(", |,| ");
                           recordList.add(rec);
                        }
                     }
                }
               }
            }
            String tmpArffFile=ForestFunctions.changedFileName(dataFile, "-tmpWin");
            ForestFunctions.createArffFile(dss.getDataSetName(),dss.getAttrNames(),dss.getAttrType(),
                    recordList.toArray(new String[][] {}),tmpArffFile);                    
            ARFFdataFile=new ArffFileStream(tmpArffFile,classIndex);                                      
            dss=new DatasetStats(ARFFdataFile,dss.getDataSetName(),dss.getAttrNames(),dss.getAttrType(),
                    recordList.toArray(new String[][] {}),dss.getClassIndex());       
            ForestFunctions.removeFile(tmpArffFile);
        }
        public Forest getClassifier()
        {
            return bestF;
        }
        public void calculateAccuracy(String [][]records,int cIndex)
        {            
            if(PF.getForestSize()>0)
            {
                accuracy[1]=PF.forestAccuracy(records,cIndex);
            }
            if(AF.getForestSize()>0)
            {
                accuracy[2]=AF.forestAccuracy(records,cIndex);
            }
            if(TF.getForestSize()>0)
            {
                accuracy[3]=TF.forestAccuracy(records,cIndex);
            }            
        }
        public void displayAllForests()
        {
            System.out.println("\nPF\n"+PF.toString());
            System.out.println("\nAF\n"+AF.toString());
            System.out.println("\nTF\n"+TF.toString());
        }
    }
    
    private class ISAT{
       private int []attrType;
       private String[]attrNames;
       private String dataFile;
       private String dsName;
       private int minLeafSize;
       private int method;
       private int cIndex;
       private ArffFileStream ARFFdataFile;
       private static final int MAX_DEPTH=20;
       
       public void expandForestByISAT(Forest F, DatasetStats d, String dataFile,int minLeafSize,
               int method, int cIndex,ArffFileStream ARFFdataFile,int[]treePerturbed, int newCV)
       {
           this.dataFile=dataFile;
           this.attrType=d.getAttrType();
           this.attrNames=d.getAttrNames();
           this.dsName=d.getDataSetName();
           this.minLeafSize=minLeafSize;
           this.method=method;
           this.cIndex=cIndex;
           this.ARFFdataFile=ARFFdataFile;

           String mCV=ClassObserver.findMajorityClassValue(d.getClassValues(), d.getClassDistribution());           
           int i=0;
           for(Tree t:F.getForest()){
               if(treePerturbed[i]>0 ||newCV>0)
               {
                   expandTreeByISAT(t,d,mCV,t.getRoot().isLeaf());
               }
               t.updateLeafStats(d, false);
               if(treePerturbed[i]>0 ||newCV>0)
               {
                   expandTreeByEntropy(t,d);
               }
               t.updateTreeDepth();
               t.updateTreeMinMax(d.getMin(),d.getMax());
               i++;
           }
       }
       public void expandTreeByISAT(Tree T, DatasetStats d, String mCV, boolean isLeaf)
       {            
            double []treeMin=T.getTreeMin();
            double []treeMax=T.getTreeMax();
            double []rangeMin1=d.getMin();
            double []rangeMax1=d.getMax();
            double []dist1=ForestFunctions.calculateDistance(rangeMin1,treeMax,attrType);
            double []dist2=ForestFunctions.calculateDistance(treeMin,rangeMax1,attrType);
            double max1=ForestFunctions.findMaxValue(dist1,attrType);
            int index1=ForestFunctions.findMaxIndex(dist1,attrType);
            double max2=ForestFunctions.findMaxValue(dist2,attrType);
            int index2=ForestFunctions.findMaxIndex(dist2,attrType);
            
            //no intersection, SAT
            if((max1>0 && max1>=max2)|| (max2>0 && max2>max1)) 
            {
                // create root node and right child
                if(max1>0 && max1>=max2)
                {
                    Node newChild=null;
                    double splitVal=(rangeMin1[index1]+treeMax[index1])/2.0;
                    String []cv=d.getClassValues();
                    if(cv.length>1 && d.getNumRecords()>this.minLeafSize)
                    {
                        Tree t=new Tree();
                        newChild=t.constructSubTree(this.ARFFdataFile,this.attrNames,this.attrType,this.minLeafSize,this.method);
                    }
                    T.addRootAndChild(newChild,attrNames[index1], attrType[index1], splitVal+"", mCV, false);
                }
                //no intersection, create root node and left child
                else if(max2>0 && max2>max1)
                {
                    Node newChild=null;
                    double splitVal=(treeMin[index2]+rangeMax1[index2])/2.0;
                    String []cv=d.getClassValues();

                    if(cv.length>1 && d.getNumRecords()>this.minLeafSize)
                    {
                        Tree t=new Tree();
                        newChild=t.constructSubTree(this.ARFFdataFile,this.attrNames,this.attrType,this.minLeafSize,this.method);
                    }
                    T.addRootAndChild(newChild,attrNames[index2], attrType[index2], splitVal+"", mCV, true);
                }
            }
            else //intersection (left or right or both), iSAT implementation
            {                
                dist1=ForestFunctions.calculateDistance(rangeMax1,treeMax,attrType);
                dist2=ForestFunctions.calculateDistance(treeMin,rangeMin1,attrType);
                max1=ForestFunctions.findMaxValue(dist1,attrType);
                index1=ForestFunctions.findMaxIndex(dist1,attrType);
                max2=ForestFunctions.findMaxValue(dist2,attrType);
                index2=ForestFunctions.findMaxIndex(dist2,attrType);                                
                if(max2>0) //intersection left side, create root node and left child
                {
                    String [][]batchData=d.getData();
                    double splitVal=treeMin[index2];
                    String [][] satData=partitionData(batchData,splitVal,index2,"L");
                    if(satData.length>0)
                    {
                        Node newChild=null;
                        String []cv=ForestFunctions.findDomainValues(satData, d.getClassIndex());
                        if(cv.length>1)
                        {
                            Tree t=new Tree();
                            newChild=t.constructSubTree(dsName, attrNames, attrType, minLeafSize, method, dataFile, satData, cIndex);
                        }
                        T.addRootAndChild(newChild,attrNames[index2], attrType[index2], splitVal+"", mCV, true);                        
                    }
                } 
                
                if(max1>0) //right side intersection, create root node and right child
                {
                    String [][]batchData=d.getData();
                    double splitVal=treeMax[index1];
                    String [][]satData=partitionData(batchData,splitVal,index1,"R");
                    if(satData.length>this.minLeafSize)
                    {
                        Node newChild=null;
                        String []cv=ForestFunctions.findDomainValues(satData, d.getClassIndex());
                        if(cv.length>1)
                        {
                            Tree t=new Tree();
                            newChild=t.constructSubTree(dsName, attrNames, attrType, minLeafSize, method, dataFile, satData, cIndex);
                        }
                        T.addRootAndChild(newChild,attrNames[index1], attrType[index1], splitVal+"", mCV, false);                        
                    }
                }                                               
            }
       }
       public void expandTreeByEntropy(Tree T, DatasetStats d)
       {
           List<Node> treeLeaves=T.getLeaves();
           for(Node node:treeLeaves)
           {
               if(node.isLeaf() && !node.isPure() && node.isPerturbed() && node.getTreeDepth()<MAX_DEPTH)
               {
                   List<String[]> data=node.getLeafData();
                   String [][]leafData=data.toArray(new String[][] {});
                   if(leafData.length>this.minLeafSize)
                   {
                   Node newChild=null;
                    String []cv=ForestFunctions.findDomainValues(leafData, d.getClassIndex());
                    if(cv.length>1)
                    {
                        Tree t=new Tree();
                        newChild=t.constructSubTree(dsName, attrNames, attrType, minLeafSize, method, dataFile, leafData, cIndex);
                        if(newChild!=null && !newChild.isLeaf() && newChild.getNumberOfChildren()>1)
                        {
                           t.updateSubTreeStats(leafData,node.getLeafClassValues(),d.getClassIndex()); 
                           //set child to parent link
                           if(node.getParent()!=null)
                           {
                           Node parent=node.getParent();
                           String sOP=node.getSplitOp();
                           String sVal=node.getSplitValue();
                           int index=node.getNodeIndex();
                           
                           newChild.setParent(parent);
                           newChild.setSplitInfo(sOP, sVal);
                           //set parent to child link
                           if(parent.getNumberOfChildren()>1)
                           {
                           int oldIndex=-1;
                           for(Node nde:parent.getChildren())
                           {    
                               oldIndex++;
                               if(nde.getNodeIndex()==index && nde.getSplitOp().equals(sOP) && nde.getSplitValue().equals(sVal))
                               {
                                   break;
                               }
                           }
                           if(oldIndex>=0)
                           {
                               parent.replaceChild(newChild, oldIndex);
                           }
                           }
                           }
                        }
                    }
               }
               }
           }
       }
       private String [][]partitionData(String [][]data, double splitVal, int index, String leftOrRight)
        {    
            String [][]pData=null;
            int records=data.length;
            ArrayList<Integer>tmpList = new ArrayList<Integer>();
            for(int r=0;r<records;r++)
            {
                double val=Double.parseDouble(data[r][index]);
                boolean flag=false;
                if(leftOrRight.equals("L"))
                {
                    if(val<=splitVal) flag=true;
                }
                else{
                    if(val>splitVal) flag=true;
                }
                if(flag)
                {
                    tmpList.add(r);
                }    
            }
            int nRecs=tmpList.size();
            int noa=attrType.length;        
            pData=new String[nRecs][noa];
            for(int r=0;r<nRecs;r++)
            {           
                System.arraycopy(data[tmpList.get(r)], 0, pData[r], 0, noa);
            }
            return pData;
        }
    }
    
    private class Forest{
        private int numTree;
        private int minLeafSize;
        private int classIndex;
        private List<Tree> trees;
        private List <String>classValues;
        Forest()
        {
            trees=new ArrayList<Tree>();
            classValues=new ArrayList<String>();
        }

        
        public int getMinLeafSize()
        {
            return this.minLeafSize;
        }       
        public void setClassValues(String []cv)
        {
            for(String c:cv)
                if(!this.classValues.contains(c))this.classValues.add(c);
        }
        public String[] getClassValues()
        {
            return this.classValues.toArray(new String[this.classValues.size()]);
        }
        public void buildForest(ArffFileStream ARFFdataFile,DatasetStats dss,int numTree, int minLeafSize,int method)
        {
            this.numTree=numTree;            
            this.minLeafSize=minLeafSize;
            this.classIndex=dss.getClassIndex();
            setClassValues(dss.getClassValues());
            String treeStr=ForestFunctions.buildClassifier(method, ARFFdataFile, numTree, minLeafSize);
            if(method==2)treeStr=ForestFunctions.preprocessTree(treeStr);
            if(treeStr.equals(""))
            {
                String majorityCV=ClassObserver.findMajorityClassValue(dss.getClassValues(), dss.getClassDistribution());                
                for(int i=1;i<=numTree;i++) 
                {
                    String []tmpConditions=new String[1];
                    tmpConditions[0]=":"+majorityCV;  
                    Tree tree=new Tree();
                    tree.buildInitialTree(tmpConditions, dss);                    
                    tree.updateTreeSingleLeafStats(dss);
                    trees.add(tree);
                }
            } 
            else{
            constructRuleToForest(treeStr,dss);   
            }
        }
        public void constructRuleToForest(String treeStr,DatasetStats dss)
        {
//           System.out.println("\n"+treeStr+"\n");
           String []rules=treeStr.split("\n");//ff.readFileAsArray(new File(treeFile));
           if(rules.length>0)           
           {
               numTree=ForestFunctions.countTrees(rules);
               int []Loc=new int[numTree];
               String []conditions=ForestFunctions.processTrees(rules, Loc);
               int totalCons=conditions.length;
               for(int tc=0;tc<numTree;tc++)
               {
                    int ln;
                    if(tc==numTree-1)
                    {
                         ln=totalCons-Loc[tc];
                    }
                    else
                    {
                         ln=Loc[tc+1]-Loc[tc];
                    }
                    int tln=ln+Loc[tc];

                    String []tmpConditions=new String[ln];
                    for(int r=Loc[tc], i=0;r<tln;r++,i++)
                    {
                        tmpConditions[i]=conditions[r];
                    }
                    
                    Tree tree=new Tree();
                    tree.buildInitialTree(tmpConditions, dss);                    
                    tree.updateLeafStats(dss, true);
                    trees.add(tree);
                 }
           }
        }
        public List<Tree> getForest()
        {
            return this.trees;
        }
        public int getForestSize()
        {
            return this.trees.size();
        }
        public String forestPrediction(String []record)
        {                        
            String []CVs=this.classValues.toArray(new String[this.classValues.size()]);
            int ncv=CVs.length;
            int []vote=new int [ncv];
            for(Tree tree:trees)
            {
               String pv =tree.getClassValueForInstance(record);
               for(int i=0;i<ncv;i++)
               {
                   if(CVs[i].equals(pv))
                   {
                        vote[i]++;break;
                   }
               }
            }
            int majorityIndex=0;
            for(int i=1;i<ncv;i++)
            {
                if(vote[i]>vote[majorityIndex])
                {
                    majorityIndex=i;
                }
            }            
            return CVs[majorityIndex];
        }
        public boolean isCorrectyClassified(Instance inst, int ci)
        {
            String s=inst.toString();
            String []record=s.split(",");
            String pcv=forestPrediction(record);
            return record[ci].equals(pcv);
        }
        public boolean isCorrectyClassified(String []record, int ci)
        {
            String pcv=forestPrediction(record);
            return record[ci].equals(pcv);
        }
        public float forestAccuracy(String testDataFile, int ci)
        {
            float accuracy=0.0f;
            ArffFileStream testData=new ArffFileStream(testDataFile,ci);
            int cIndex=testData.getHeader().classIndex();
            int numberSamplesCorrect = 0;
            int numberSamples = 0;
            while(testData.hasMoreInstances())
            {
               Instance inst=testData.nextInstance().getData();
               numberSamples++;
               if(isCorrectyClassified(inst,cIndex))numberSamplesCorrect++;                   
            }
            if(numberSamples>0)
                accuracy = 100.0f * (float) numberSamplesCorrect/ (float) numberSamples;
           return accuracy;
        }
        public float forestAccuracy(String [][]records,int cIndex)
        {
            float accuracy=0.0f;
            int numberSamplesCorrect = 0;
            int numberSamples = records.length;
            for(int i=0;i<numberSamples;i++){
               if(isCorrectyClassified(records[i],cIndex))numberSamplesCorrect++;                   
            }
            if(numberSamples>0)
                accuracy = 100.0f * (float) numberSamplesCorrect/ (float) numberSamples;
           return accuracy;
        }
        
        @Override
        public String toString()
        {
            String out="";
            int t=0;
            for(Tree tree:trees)
            {
                t++;
                out+="\nTree: "+t+", Total nodes:"+tree.getTotalNodeCount()
                        +", Total leaves:"+tree.getTotalLeafCount()
                        +", Tree depth:"+tree.getTreeDepth()+"\n";
                out+=tree.toString()+"\n";                
            }
            return out;
        }
        
    }
    
    private class Tree{
        final String levelPadding="|   ";
        private Node root;  
        private String []classValues;
        private int []classDist;
        private int totalNodeCount;
        private int totalLeafCount;
        private boolean leafFoundFlag=false;
        private Node findLeaf;
        private int []atype;
        private String []aNames;
        private double []treeMax;
        private double []treeMin;
        private int treeDepth;
        private List<Node> leafCollection = new ArrayList<>();
        public Tree()
        {
            root=new Node(null,0,false,0);
            totalNodeCount=0;
            treeDepth=0;
            totalLeafCount=0;
        }

        public int getTreeDepth()
        {
            return treeDepth;
        }
        public List<Node> getLeaves()
        {
            return leafCollection;
        }
        public void buildInitialTree(String []conditions, DatasetStats dss)
        {

            atype=dss.getAttrType();
            aNames=dss.getAttrNames();
            treeMax=dss.getMax();
            treeMin=dss.getMin();
            classValues=dss.getClassValues();
            classDist=dss.getClassDistribution();
            constructTree(conditions,aNames,atype);            
        }
        public void constructTree(String []conditions)//tree with just a single leaf
        {
                String majorityCV="";
                if(conditions[0].contains(":")&& conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(conditions[0].indexOf(":")+1, conditions[0].indexOf("("));
                }
                else if(conditions[0].contains(":")&& !conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(conditions[0].indexOf(":")+1, conditions[0].length());
                }
                else if(!conditions[0].contains(":")&& conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(0, conditions[0].indexOf("("));
                }
                else
                {
                    majorityCV=conditions[0];
                }
             root=new Node(null,0,true,0);                         
             totalLeafCount++;totalNodeCount++;             
             root.setLeafPrediction(majorityCV.trim());
             leafCollection.add(root);
        }
        public void constructTree(String []conditions,String []attrNames,int []attrType)
        {
            int n=conditions.length;  
            if(conditions.length==1)
            {
                constructTree(conditions);
            }
            else{
            Node currentNode=root;
            int nodeIndex=0;
            boolean rootSet=false;
            ArrayList<String> lAttr=new ArrayList<String>();
            ArrayList<String> sVal=new ArrayList<String>();
            ArrayList<String> sOp=new ArrayList<String>();
                      
            for(int c=0;c<n;c++) 
             { 
                 String []cac=ForestFunctions.separateConditionClass(conditions[c]);
                 String []strVal=ForestFunctions.convertStringToArray(cac[0]);
                 int nstr=strVal.length;
                 int cl=0,tl=0;
                 //find levels
                 for (int x=0;x<nstr;x++)
                 {
                     if(strVal[x].equals("|"))
                     {
                         cl++;
                     }
                 }
                 //delete old node information 
                 tl=lAttr.size();
                 if(tl>cl)
                 {
                     while(tl>cl)
                     {
                         tl--;
                         lAttr.remove(tl);
                         sVal.remove(tl);
                         sOp.remove(tl);
                         if(currentNode.getParent()!=null)
                         {
                             currentNode=currentNode.getParent();
                         }
                     }
                 }
                 //add new node information
                 if((cl+2)<nstr)
                 {
                     lAttr.add(strVal[cl]);
                     sOp.add(strVal[cl+1]);
                     sVal.add(strVal[cl+2]);
                 }
                 tl=lAttr.size();
                 if(tl==1)
                 {
                    if(!rootSet) {                                                      
                       currentNode.setNodeInfo(ForestFunctions.findAttrType(lAttr.get(0),attrNames,attrType), lAttr.get(0));
                       rootSet=true;
                    }
                 }
                 else if(tl>1){
                     int depth=currentNode.getTreeDepth();
                     if(depth<tl-1)
                     {
                         treeDepth=depth+1;
                        Node child=new Node(currentNode,treeDepth,false,++nodeIndex); 
                        int tmp=tl-1;
                        child.setNodeInfo(ForestFunctions.findAttrType(lAttr.get(tmp),attrNames,attrType), lAttr.get(tmp));                    
                        child.setSplitInfo(sOp.get(tmp-1), sVal.get(tmp-1));
                        currentNode.addChild(child);
                        currentNode=currentNode.getLastChild();
                     }
                 }
                 if(!cac[1].isEmpty())
                 {
                     tl=lAttr.size();
                     if(tl>0)
                     {
                         treeDepth=currentNode.getTreeDepth()+1;
                         Node child=new Node(currentNode,treeDepth,true,++nodeIndex);                         
                         totalLeafCount++;
                         tl--;
                         child.setSplitInfo(sOp.get(tl), sVal.get(tl));
                         child.setLeafPrediction(cac[1]);
                         currentNode.addChild(child);
                         leafCollection.add(child);
                         lAttr.remove(tl);
                         sVal.remove(tl);
                         sOp.remove(tl);
                     }
                 }                  
            }
            totalNodeCount=nodeIndex+1;
            }
        }
        
        
        public void updateLeafStats(DatasetStats dss, boolean isFirstBatch)
        { 
            int n=dss.getNumRecords();
            
            if(n>0)
            {
                int ncv=0,tcv=0;
                
                if(!isFirstBatch)
                {
                    String []tmpoldCV=classValues.clone();
                    String []tmpClassValues=dss.getClassValues();
                    int []tmpClassDist=dss.getClassDistribution();
                    classValues=ClassObserver.updateClassValues(classValues, tmpClassValues);  
                    ncv=ClassObserver.getNumNewCV();
                    classDist=ClassObserver.updateClassDistribution(classValues,tmpoldCV,classDist.clone(), tmpClassValues, tmpClassDist);
                }
                tcv=classValues.length;
                int [][]leafCVDist=new int[totalLeafCount][tcv];                
                int []leafID=getLeafIndex();
                String [][]dataset=dss.getData();
                int ci=dss.getClassIndex();
                for(int r=0;r<n;r++)
                {
                    Node foundNode=findLeafForInstance(dataset[r]);
                    if(foundNode!=null)
                    {
                        foundNode.addInstance(dataset[r]);
                        int lid=ClassObserver.findArrayIndex(leafID,foundNode.getNodeIndex());
                        int cid=ClassObserver.getCVIndex(classValues, dataset[r][ci]);
                        if(lid>=0 && cid>=0)leafCVDist[lid][cid]++;
                    }
                }
                for(int l=0;l<totalLeafCount;l++)
                {
                    Node leaf=leafCollection.get(l);
                    if(isFirstBatch)
                    {
                        ClassObserver.calculateConfidence(classValues, leafCVDist[l]);
                    }
                    else {
                        if(leaf.isClassDistributionSet())
                        {
                            String []oldLeafCVs=leaf.getLeafClassValues();
                            int []oldLeafDist=leaf.getLeafClassDistribution();
                            ClassObserver.updateClassDistribution(oldLeafCVs, oldLeafDist, classValues, leafCVDist[l]);                        
                        }
                        else{
                            ClassObserver.updateClassDistribution(classValues, leafCVDist[l]);                           
                        }
                    }                     
                    leaf.setClassDistribution(classValues, leafCVDist[l]);
                    leaf.setConfidence(ClassObserver.getConfidence());
                    leaf.setLeafPrediction(ClassObserver.getMajorityCV());                    
                }
                
            }
        }
        
        public void updateSubTreeStats(String [][]dataset,String []leafCV,int ci)
        { 
            int n=dataset.length;
            
            if(n>0)
            {
                int tcv=leafCV.length;
                int [][]leafCVDist=new int[totalLeafCount][tcv];                
                int []leafID=getLeafIndex();
                for(int r=0;r<n;r++)
                {
                    Node foundNode=findLeafForInstance(dataset[r]);
                    if(foundNode!=null)
                    {
                        foundNode.addInstance(dataset[r]);
                        int lid=ClassObserver.findArrayIndex(leafID,foundNode.getNodeIndex());
                        int cid=ClassObserver.getCVIndex(leafCV, dataset[r][ci]);
                        if(lid>=0 && cid>=0)leafCVDist[lid][cid]++;
                    }
                }
                for(int l=0;l<totalLeafCount;l++)
                {
                    Node leaf=leafCollection.get(l);
                    ClassObserver.updateClassDistribution(leafCV, leafCVDist[l]);                           
                    leaf.setClassDistribution(leafCV, leafCVDist[l]);
                    leaf.setConfidence(ClassObserver.getConfidence());
                    leaf.setLeafPrediction(ClassObserver.getMajorityCV());                    
                }
                
            }
        }
        
        public void updateTreeSingleLeafStats(DatasetStats dss)
        { 
            String [][]dataset=dss.getData();            
            int n=dataset.length;
            
            if(n>0)
            {
                for(int r=0;r<n;r++)
                {
                    root.addInstance(dataset[r]);                    
                }
                ClassObserver.updateClassDistribution(dss.getClassValues(), dss.getClassDistribution());                           
                root.setClassDistribution(dss.getClassValues(), dss.getClassDistribution());
                root.setConfidence(ClassObserver.getConfidence());
                root.setLeafPrediction(ClassObserver.getMajorityCV());                                 
            }
        }
        
        public int identifyPerturbedLeaves(DatasetStats dss, double errorTolerance)
        { 
            int totalPerturbed=0;
            int n=dss.getNumRecords();            
            if(n>0)
            {                
                String []tmpClassValues=dss.getClassValues();
                int []tmpClassDist=dss.getClassDistribution();
                String []newClassValues=ClassObserver.updateClassValues(classValues, tmpClassValues);  
                int [] newClassDist=ClassObserver.updateClassDistribution(newClassValues,classValues,classDist.clone(), tmpClassValues, tmpClassDist);                
                int tcv=newClassValues.length;
                int [][]leafCVDist=new int[totalLeafCount][tcv];                
                int []leafID=getLeafIndex();
                String [][]dataset=dss.getData();
                int ci=dss.getClassIndex();
                for(int r=0;r<n;r++)
                {
                    Node foundNode=findLeafForInstance(dataset[r]);
                    if(foundNode!=null)
                    {                        
                        int lid=ClassObserver.findArrayIndex(leafID,foundNode.getNodeIndex());
                        int cid=ClassObserver.getCVIndex(newClassValues, dataset[r][ci]);
                        if(lid>=0 && cid>=0)leafCVDist[lid][cid]++;
                    }
                }
                for(int l=0;l<totalLeafCount;l++)
                {
                    Node leaf=leafCollection.get(l);                    
                    if(leaf.isClassDistributionSet())
                    {
                        String []oldLeafCVs=leaf.getLeafClassValues();
                        int []oldLeafDist=leaf.getLeafClassDistribution();
                        double preConfidence=leaf.getConfidence();
                        ClassObserver.updateClassDistribution(oldLeafCVs, oldLeafDist, newClassValues, leafCVDist[l]);                        
                        double newConfidence=ClassObserver.getConfidence();
                        if(preConfidence>(newConfidence+errorTolerance))
                        {
                            leaf.setPerturbed();
                            totalPerturbed++;
                        }
                        else
                        {
                            leaf.setNotPerturbed();
                        }
                    }
                    else{
                        leaf.setNotPerturbed();
                    }
                   
                }
                
            }
            return totalPerturbed;
        }
        
        public void updateTreeMinMax(double []newMin, double []newMax)
        {
            for(int i=0;i<atype.length;i++)
            {
                if(atype[i]==1){
                    if(newMin[i]<treeMin[i])treeMin[i]=newMin[i];
                    if(newMax[i]>treeMax[i])treeMax[i]=newMax[i];
                }
            }
        }
        public Node constructSubTree(String dsName,String []attrNames,int []attrType, int minLeafSize,int method,String batchFile,String [][]data, int classIndex)
        {   
            this.aNames=attrNames;
            this.atype=attrType;
            String tmpArffFile=ForestFunctions.changedFileName(batchFile, "-tmp");
            ForestFunctions.createArffFile(dsName,attrNames,attrType,data,tmpArffFile);                    
            ArffFileStream ARFFdataFile=new ArffFileStream(tmpArffFile,classIndex);              
            Node node=constructSubTree(ARFFdataFile,attrNames,attrType,minLeafSize,method);  
            ForestFunctions.removeFile(tmpArffFile);
            return node;                  
        }
        
        public Node constructSubTree(ArffFileStream ARFFdataFile,String []attrNames,int []attrType, int minLeafSize,int method)
        {                     
            String treeStr=ForestFunctions.buildClassifier(method, ARFFdataFile, 1, minLeafSize);
            if(treeStr.equals(""))
            {
                 return null;
            }
            else{
               if(method==2)treeStr=ForestFunctions.preprocessTree(treeStr);
               constructRuleToTree(treeStr,attrNames,attrType);
               return root;
            }            
        }
        
        private void constructRuleToTree(String treeStr,String []attrNames,int []attrType)
        {
           String []rules=treeStr.split("\n");
           if(rules.length>2) //if it is not just a tree with a single leaf          
           {
               int numTree=ForestFunctions.countTrees(rules);
               int []Loc=new int[numTree];
               constructTree(ForestFunctions.processTrees(rules, Loc), attrNames,attrType);
           }
           else
               root=null;
        }
        public void addRootAndChild(Node newChild,String nodeName,int nodeType,String splitValue,String nodePrediction,boolean isLeft)
        {
            
            Node oldChild=root;
            root=new Node(null,0,false,totalNodeCount++);
            root.setNodeInfo(nodeType, nodeName);            
            oldChild.setParent(root);
            if(newChild==null){
                newChild=new Node(root,root.getTreeDepth()+1,true,totalNodeCount++);                                     
                newChild.setLeafPrediction(nodePrediction);  
            }
            if(isLeft)
            {
                newChild.setSplitInfo("<=", splitValue);
                root.addChild(newChild);
                oldChild.setSplitInfo(">", splitValue);
                root.addChild(oldChild);
            }
            else{
                oldChild.setSplitInfo("<=", splitValue);
                root.addChild(oldChild);
                newChild.setSplitInfo(">", splitValue);
                root.addChild(newChild);                
            }

            this.totalNodeCount=0;
            leafCollection.removeAll(leafCollection);
            totalLeafCount=0;
            updateTreeDepthAndIndex(root,0);
        }
        public void updateTreeDepth()
        {
            this.totalNodeCount=0;
            leafCollection.removeAll(leafCollection);
            totalLeafCount=0;
            treeDepth=0;
            updateTreeDepthAndIndex(root,0);            
        }
        
        public int[]getLeafIndex()
        {
            int []leafIndex=new int[totalLeafCount];
            int i=0;
            for(Node leaf:leafCollection)
            {
                leafIndex[i]=leaf.getNodeIndex();
                i++;
            }
            return leafIndex.clone();
        }
        
        public void displayLeafInfo()
        {
            for(Node leaf:leafCollection)
            {
                System.out.println(leaf.toString());
            }
        }
        public void updateTreeDepthAndIndex(Node node, int depth)
        {           
                node.setNodeIndex(this.totalNodeCount++);
                node.setTreeDepth(depth);
                if(node.isLeaf()){
                    leafCollection.add(node);
                    this.totalLeafCount++;
                    if(depth>treeDepth)treeDepth=depth;
                }
                List<Node> children=node.getChildren();
                for (Node child : children) {                   
                    updateTreeDepthAndIndex(child,node.getTreeDepth()+1);
                }
            
        }
        public int getTotalNodeCount()
        {
            return this.totalNodeCount;
        }
        public int getTotalLeafCount()
        {
            return this.totalLeafCount;
        }
        public double []getTreeMin()
        {
            return treeMin.clone();
        }
        public double []getTreeMax()
        {
            return treeMax.clone();
        }
        public Node getRoot()
        {
            return this.root;
        }
        public Node findLeafForInstance(String []record)
        {
           leafFoundFlag=false;
           if(this.root!=null)
           {
               searchTree(record,this.root);
           }
           return findLeaf;
        }
        public String getClassValueForInstance(String []record)
        {
           Node foundNode=findLeafForInstance(record);
           return foundNode.getLeafPrediction();
        }
        private void searchTree(String []record,Node node)
        {
            if(node.isLeaf())
            {
                findLeaf=node;
                leafFoundFlag=true;
            }
            else
            {
                String currentStr=record[ForestFunctions.findAttrIndex(node.getSplitName(),aNames)];
                List<Node> children=node.getChildren();
                for (int i=0;i<children.size() && leafFoundFlag==false;i++) 
                {
                    Node child =children.get(i);
                    String splitValue=child.getSplitValue();
                    if(node.isNumeric())
                    {
                        boolean flag=false;
                        double rval=Double.parseDouble(currentStr);
                        double sval=Double.parseDouble(splitValue);
                        String splitOp=child.getSplitOp();
                        if(splitOp.equals("<="))
                           {
                               if(rval<=sval) flag=true;
                           }
                        else if(splitOp.equals("<"))
                           {
                               if(rval<sval) flag=true;
                           }                
                        else if(splitOp.equals(">"))
                           {
                               if(rval>sval) flag=true;
                           }
                        else if(splitOp.equals(">="))
                           {
                               if(rval>=sval) flag=true;
                           }
                        if(flag)
                        {
                            searchTree(record,child);
                        }
                    }
                    else
                    {
                        if(splitValue.equals(currentStr))
                        {
                            searchTree(record,child);
                        }
                    }
                }
                
            }
        }
        
        
        public void describeSubTree(Node node, StringBuilder out, int indent)
        {
            if(node.isLeaf())
            {
                out.append(": "+node.getLeafPrediction()+" "+node.getClassDistribution());
            }
            else
            {
                List<Node> children=node.getChildren();
                for (Node child : children) {
                    out.append("\n");
                    for(int j=0;j<indent;j++)
                    {
                        out.append(levelPadding);
                    }
                    out.append(node.getSplitName()+" "+child.getSplitOp()+" "+child.getSplitValue());
                    describeSubTree(child,out, child.getTreeDepth());
                }
            }
        }
        @Override
        public String toString()
        {
            StringBuilder out=new StringBuilder();
            describeSubTree(root, out,0);
            return out.toString();
        }
    }
    
    private static class Node //implements Cloneable
    {
        private Node parent;
        private List<Node> children = new ArrayList<>();
        List<String[]> recordList = new ArrayList<String[]>();
        private boolean isLeafNode;
        private int nodeIndex;
        private int nodeType;
        private int treeDepth;
        private String splitAttrName;
        private String splitOp;
        private String splitValue;
        private String majorityClassValue;
        private String classValuesDistribution;
        private String []leafClassValues;
        private int []leafClassDistribution;
        private boolean leafPerturbed;
        private double confidence;
        
        Node(Node parent,int treeDepth,boolean isLeafNode,int nodeIndex)
        {
           this.parent=parent; 
           this.treeDepth=treeDepth;
           this.isLeafNode=isLeafNode;
           this.nodeIndex=nodeIndex;
           this.nodeType=2;
           this.leafPerturbed=false;
           leafClassValues=null;
        }

        public void setNodeInfo(int nodeType, String nodeName)
        {
            this.nodeType=nodeType;
            this.splitAttrName=nodeName;
        }
        public void setSplitInfo(String splitOp,String splitValue)
        {
            this.splitOp=splitOp;
            this.splitValue=splitValue;
        }
        public int getTreeDepth()
        {
            return this.treeDepth;
        }
        public void setTreeDepth(int treeDepth)
        {
           this.treeDepth=treeDepth;
        }
        public String getSplitName()
        {
            return this.splitAttrName;
        }
        public String getSplitOp()
        {
            return this.splitOp;
        }
        public String getSplitValue()
        {
            return this.splitValue;
        }
        public boolean isNumeric()
        {
            return this.nodeType==1;
        }
        public void setLeafPrediction(String mCV)
        {
            this.majorityClassValue=mCV;
        }
        public String getLeafPrediction()
        {
            return this.majorityClassValue;
        }
        public void addChild(Node child)
        {
            this.children.add(child);
        }
        public void replaceChild(Node newChild, int index)
        {
            this.children.set(index, newChild);
        }
        public List<Node> getChildren()
        {
            return this.children;
        }
        public int getNumberOfChildren()
        {
             return this.children.size();
        }
        public void setParent(Node parent)
        {
           this.parent=parent;
        }
        public Node getParent()
        {
            return this.parent;
        }
        public Node getFirstChild()
        {
            if(this.children.size()>0)
                return this.children.get(0);
            else
                return null;
        }
        public Node getLastChild()
        {
            int noc=this.children.size();
            if(noc>0)
                return this.children.get(noc-1);
            else
                return null;
        }
        public void setPerturbed()
        {
            this.leafPerturbed=true;
        }
        public void setNotPerturbed()
        {
            this.leafPerturbed=false;
        }
        public boolean isPerturbed()
        {
            return this.leafPerturbed;
        }
        public boolean isLeaf()
        {
            return this.isLeafNode;
        }
        public void setNodeIndex(int nodeIndex)
        {
            this.nodeIndex=nodeIndex;            
        }
        public int getNodeIndex()
        {
            return this.nodeIndex;            
        }
        public void addInstance(String []record)
        {
            recordList.add(record);
        }
        public List<String []> getLeafData()
        {
           return recordList;
        }
        public String[] getLeafData(int index)
        {
           return recordList.get(index);
        }               
        public void setClassDistribution(String []classValues, int []classDistribution)
        {
            this.leafClassValues=classValues.clone();
            this.leafClassDistribution=classDistribution.clone();
            classValuesDistribution="{";
            for(int i=0;i<leafClassValues.length;i++)
            {
                if(i==0)
                    classValuesDistribution+=leafClassValues[i]+":"+leafClassDistribution[i];
                else
                    classValuesDistribution+=", "+leafClassValues[i]+":"+leafClassDistribution[i];
            }
            classValuesDistribution+="}";
        }
        public boolean isClassDistributionSet()
        {
            if (this.leafClassValues==null)
            {
                return false;                
            }
            else{
                if (this.leafClassValues.length>0)
                    return true; 
                else
                    return false; 
            }
        }
        public String getClassDistribution()
        {            
            return this.classValuesDistribution;
        }
        public String []getLeafClassValues()
        {
            return this.leafClassValues.clone();
        }
        public int []getLeafClassDistribution()
        {
            return this.leafClassDistribution.clone();
        }
        public double getConfidence()
        {
            return this.confidence;
        }
        public void setConfidence(double confidence)
        {
            this.confidence=confidence;
        }
        public boolean isPure()
        {   
            return this.confidence==1.0;
        }
        
        @Override
        public String toString()
        {
            String nStr="";
            nStr+="Node Index:"+this.nodeIndex+", data size:"+recordList.size()+"\n";
            nStr+="Class distribution:\n"+getClassDistribution();
            nStr+="\nIsPure:"+isPure()+", Class prediction:"+this.majorityClassValue;
            nStr+=", Confidence:"+this.confidence+"\n";
            return nStr;
        }
    }
    
    private static class ClassObserver{
        private static double confidence=0.0;
        private static String majorityCV;
        private static int numnewcv;
        private static String[] updateClassValues(String []OCV, String []NCV)
        {
            int ol=OCV.length;
            int nl=NCV.length;
            numnewcv=0;
            ArrayList<String> oldCV=new ArrayList<String>();
            for(int i=0;i<ol;i++)
            {
                oldCV.add(OCV[i]);
            }
            for(int i=0;i<nl;i++)
            {
                if(!oldCV.contains(NCV[i]))
                {
                    oldCV.add(NCV[i]);
                    numnewcv++;
                }
            }
            return oldCV.toArray(new String[oldCV.size()]);
        }
        private static int getNumNewCV()
        {
            return numnewcv;
        }
        private static int []updateClassDistribution(String []TCV,
                String []OCV,int []odist, String []NCV, int []ndist)
        {
            int tl=TCV.length;
            int ol=OCV.length;
            int nl=NCV.length;
            int []tdist=new int[tl];
            
            for(int i=0;i<tl;i++)
            {
//               tdist[i]=0;
               int index=getCVIndex(OCV,TCV[i]);
               if(index>=0)tdist[i]+=odist[index];
               index=getCVIndex(NCV,TCV[i]);
               if(index>=0)tdist[i]+=ndist[index];
            }                        
            return tdist.clone();
        }
        private static void updateClassDistribution(String []OCV,int []odist, String []NCV, int []ndist)
        {
            int nl=NCV.length;
            for(int i=0;i<nl;i++)
            {
               int index=getCVIndex(OCV,NCV[i]);
               if(index>=0)ndist[i]+=odist[index];
            }  

            int maxIndex=0;
            int total=ndist[0];
            for(int i=1;i<nl;i++)
            {
                total+=ndist[i];
                if(ndist[i]>ndist[maxIndex])
                    maxIndex=i;
            }
            majorityCV=NCV[maxIndex];            
            //confidence
            confidence=(double)ndist[maxIndex]/(double)total;
        }
        private static void updateClassDistribution(String []NCV, int []ndist)
        {
            int nl=NCV.length;            
            int maxIndex=0;
            int total=ndist[0];
            for(int i=1;i<nl;i++)
            {
                total+=ndist[i];
                if(ndist[i]>ndist[maxIndex])
                    maxIndex=i;
            }
            majorityCV=NCV[maxIndex];            
            //confidence
            confidence=(double)ndist[maxIndex]/(double)total;
        }
        private static void calculateConfidence( String []NCV, int []ndist)
        {
            int nl=NCV.length;
            int maxIndex=0;
            int total=ndist[0];
            for(int i=1;i<nl;i++)
            {
                total+=ndist[i];
                if(ndist[i]>ndist[maxIndex])
                    maxIndex=i;
            }
            majorityCV=NCV[maxIndex];            
            //confidence
            confidence=(double)ndist[maxIndex]/(double)total;
        }    
        private static String findMajorityClassValue(String []NCV, int []ndist)
        {
            int nl=NCV.length;
            int maxIndex=0;
            for(int i=1;i<nl;i++)
            {
                if(ndist[i]>ndist[maxIndex])
                    maxIndex=i;
            }
            return NCV[maxIndex];            
        }
        private static double getConfidence()
        {
            return confidence;
        }
        private static String getMajorityCV()
        {
            return majorityCV;
        }
        private static int getCVIndex(String []values, String cv)
        {
            int index=-1;
            int l=values.length;
            for(int i=0;i<l;i++)
            {
                if(values[i].equals(cv))
                {
                    index=i;break;
                }
            }
            return index;
        }
        private static int findArrayIndex(int []leafIDs, int leafIndex)
        {
            int index=-1;
            int l=leafIDs.length;
            for(int i=0;i<l;i++)
            {
                if(leafIDs[i]==leafIndex)
                {
                    index=i;break;
                }
            }
            return index;
        }
    }
    
    
    private static class DatasetStats{
       private int numRecords;
       private int numAttr;
       private String [][]data;
       private String []classValues;
       private int []classDist;
       private int classIndex;
       private double []Max;
       private double []Min;
       ArffFileStream dataFile;
       private String []attrNames;
       private int []attrType;
       private String dsName;
       public DatasetStats()
       {
           
       }
       public DatasetStats(ArffFileStream dataFile,String dsName,String []attrNames,int []attrType,String [][]data,int classIndex)
       {
           this.dataFile=dataFile;
           this.dsName=dsName;
           this.attrNames=attrNames;
           this.attrType=attrType;
           this.data=data;
           this.classIndex=classIndex;
           this.numRecords=data.length;
           this.numAttr=attrType.length;
           findDatasetStats();
       }
       private void findDatasetStats()
       {
           classValues=ForestFunctions.findDomainValues(data, classIndex);
           Max=new double[numAttr];
           Min=new double[numAttr];
           for(int i=0;i<numAttr;i++)
           {Max[i]=Double.NEGATIVE_INFINITY;Min[i]=Double.POSITIVE_INFINITY;}
           int numcv=classValues.length;
           classDist=new int[numcv];
           for(int i=0;i<numcv;i++)
           {classDist[i]=0;}
           
           for(int i=0;i<numRecords;i++)
           {
               for(int j=0;j<numAttr;j++)
               {
                   if(attrType[j]==1)
                    {
                        double cval=Double.parseDouble(data[i][j]);
                        if(cval>Max[j])Max[j]=cval;
                        if(cval<Min[j])Min[j]=cval;
                    }
                    else
                    {
                        if(j==classIndex)
                        {
                            int cvi=0;
                            for(int k=0;k<numcv;k++)
                            {
                                if(classValues[k].equals(data[i][j]))
                                {
                                    cvi=k;break;
                                }
                            }
                            classDist[cvi]++;                            
                        }

                    }
               }
           }            
       }
       
       
       public void processARFFDataFile(ArffFileStream ARFFdataFile)
       {
           calculateDatasetStats(ARFFdataFile);
       }
       
       public void calculateDatasetStats(ArffFileStream ARFFdataFile)
       {
           this.dataFile=ARFFdataFile;     
           this.dsName=ARFFdataFile.getHeader().getRelationName();
           this.classIndex=ARFFdataFile.getHeader().classIndex();
           if (ARFFdataFile.hasMoreInstances()) 
           {
            Instance Inst = ARFFdataFile.nextInstance().getData();
            this.numAttr=Inst.numAttributes();
            this.attrType=new int[this.numAttr];
            this.attrNames=new String[this.numAttr];
            for(int i=0;i<this.numAttr;i++)
                {
                    if(i==this.classIndex)
                         this.attrType[i]=2;
                    else if(Inst.attribute(i).isNumeric())
                         this.attrType[i]=1;
                    else 
                         this.attrType[i]=0;
                    this.attrNames[i]=Inst.attribute(i).name();
                }
            }

           
           Max=new double[numAttr];
           Min=new double[numAttr];
           String []tdata=CountRecords();
           numRecords=tdata.length;           
           data=new String[numRecords][numAttr];
           dataFile.restart();
           for(int i=0;i<numAttr;i++)
           {Max[i]=Double.NEGATIVE_INFINITY;Min[i]=Double.POSITIVE_INFINITY;}
           int numcv=classValues.length;
           classDist=new int[numcv];
           for(int i=0;i<numcv;i++)
           {classDist[i]=0;}
           
           for(int i=0;i<numRecords;i++)
           {
               String []t=tdata[i].split(",");
               for(int j=0;j<numAttr;j++)
               {
                   data[i][j]=t[j];
                   if(attrType[j]==1)
                    {
                        double cval=Double.parseDouble(t[j]);
                        if(cval>Max[j])Max[j]=cval;
                        if(cval<Min[j])Min[j]=cval;
                    }
                    else
                    {
                        if(j==classIndex)
                        {
                            int cvi=0;
                            for(int k=0;k<numcv;k++)
                            {
                                if(classValues[k].equals(t[j]))
                                {
                                    cvi=k;break;
                                }
                            }
                            classDist[cvi]++;                            
                        }

                    }
               }
           }            
       }
       private String[] CountRecords()
       {
           dataFile.restart();
           List<String> domainValues = new ArrayList<>();
           List<String> tmpdata= new ArrayList<>();
           while(dataFile.hasMoreInstances())
           {
               Instance inst=dataFile.nextInstance().getData();               
               String s=inst.toString();
               tmpdata.add(s);
               String []tData=s.split(",");       
               if(!domainValues.contains(tData[classIndex]))
               {
                    domainValues.add(tData[classIndex]);
               }                    
           }
           classValues=domainValues.toArray(new String[domainValues.size()]);
           return tmpdata.toArray(new String[tmpdata.size()]);
       }
       public int getNumRecords()
       {
               return numRecords;
       }
       public double[] getMax()
       {
           return Max.clone();
       }
       public double[] getMin()
       {
           return Min.clone();
       }
       public String[] getClassValues()
       {
           return classValues.clone();
       }
       public String[][] getData()
       {
           return data.clone();
       }
       public int[] getClassDistribution()
       {
           return classDist.clone();
       }
       
       public String getDataSetName()
        {
            return this.dsName;
        }
      public int getClassIndex()
        {
            return classIndex;
        }
   
        public String []getAttrNames()
        {
            return attrNames.clone();
        }      
        public int []getAttrType()
        {
            return attrType.clone();
        } 
        public int getNumOfAttrs()
        {
            return numAttr;
        }      
   }  
   private static class ForestFunctions {

    final String newline = "\n";
    
    private static String buildClassifier(int method,ArffFileStream trainArffs, int numTrees,int leafSize)
    {
        if(method==1 || method==2)
        {
            return buildWekaForest(method,trainArffs,numTrees,leafSize);
        }
        else{
            return buildForestHT(trainArffs,numTrees,leafSize);
        }
    }
    
    private static String buildWekaForest(int method,ArffFileStream trainArffs, int numTrees,int leafSize)
    {
        WEKAClassOption wekaLearnerOption;
        weka.core.Instances instancesBuffer=null;
        if(method==1)
        {
            wekaLearnerOption= new WEKAClassOption("baseLearner", 'T',
                "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.trees.SysFor -L 50 -N 10 -G 0.3 -C 0.25 -S 0.3");
        }
        else{
            wekaLearnerOption= new WEKAClassOption("baseLearner", 'l',
                "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.trees.RandomForest -P 100 -print -I 1 -num-slots 1 -K 0 -M 2 -V 0.001 -S 1");  
        }    
        Classifier classifier;
        SamoaToWekaInstanceConverter instanceConverter;
        String tree="";
        try{
            instanceConverter = new SamoaToWekaInstanceConverter();
            String[] options = weka.core.Utils.splitOptions(wekaLearnerOption.getValueAsCLIString());
            String classifierName = options[0];
            String[] newoptions = options.clone();
            newoptions[0] = "";
            if(classifierName.equals("weka.classifiers.trees.RandomForest"))
            {
                newoptions[5] = numTrees+"";
                newoptions[11] = leafSize+"";                
            }
            else if(classifierName.equals("weka.classifiers.trees.SysFor"))
            {
                newoptions[2] = leafSize+"";
                newoptions[4] = numTrees+"";
            }            
            classifier = weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);
            int bflag=0;
            while (trainArffs.hasMoreInstances()) {
                   Instance trainInst = trainArffs.nextInstance().getData();
                   weka.core.Instance inst= instanceConverter.wekaInstance(trainInst);
                   if(bflag==0)
                   {
                       instancesBuffer = new weka.core.Instances(inst.dataset());
                       bflag=1;
                   }
                   instancesBuffer.add(inst);
                 }
           weka.classifiers.Classifier auxclassifier = weka.classifiers.AbstractClassifier.makeCopy(classifier);
           auxclassifier.buildClassifier(instancesBuffer);
           classifier = auxclassifier;
           tree= classifier.toString();
        }
        catch(Exception e)
        {
            tree="";
        }
        return tree;
    }
    
  public static String buildForestHT(ArffFileStream ARFFdataFile, int numTrees,int leafSize)
    {
        float lamda=6.0f;
        ARFFdataFile.prepareForUse();
        moa.classifiers.Classifier []forest = new moa.classifiers.Classifier[numTrees];

        for(int t=0;t<numTrees;t++)
        {
               HoeffdingTree ht=new HoeffdingTree();
               ht.gracePeriodOption.setValue(leafSize);
               ht.leafpredictionOption.setChosenIndex(0);
               forest[t]=ht;
               forest[t].setModelContext(ARFFdataFile.getHeader());
               forest[t].prepareForUse();
        }
        while (ARFFdataFile.hasMoreInstances()) {
            for(int t=0;t<numTrees;t++)
            {
                Random r=new Random();
                int k = MiscUtils.poisson(lamda, r);
                if(k>0){
                       Instance trainInst = ARFFdataFile.nextInstance().getData();
                       Instance weightedInstance = trainInst.copy();
                       weightedInstance.setWeight(trainInst.weight() * k);
                       forest[t].trainOnInstance(weightedInstance);
                     } 
            }
        }
        
        String rec="";
        for(int t=0;t<numTrees;t++)
        {
//            System.out.println(forest[t].toString());
            rec=rec+"Tree "+(t+1)+":\n";
            rec=rec+preprocessTreeHT(forest[t].toString())+"\n";           
        }
        return rec;
    } 
   private static String preprocessTreeHT(String tree)
    {
        if(tree.equals(""))
        {
            return "";
        }
        String []rules=tree.split("\r\n");
        String rec="";
        String tab="|   ";
        if(rules.length>4)
        {
        for(int i=0;i<rules.length;i++)
        {
            if(rules[i].equals("Model description:"))
            {
               i++;
               while(i<rules.length)
               {
                   String rule=rules[i].trim();
                   
                   if(rule.startsWith("if"))
                   {
                       int ntab=(int)(rules[i].substring(0, rules[i].indexOf("if")).length())/2;
                       for(int l=0;l<ntab;l++)
                       {
                           rec=rec+tab;
                       }
                       int j=i+1;
                       if(j<rules.length)
                       {
                           
                           String srule=rules[j].trim();
                           String nl=""; boolean isleaf=false;
                           if(srule.startsWith("if"))
                           {
                               nl="\n";
                           }
                           else
                           {
                               isleaf=true;
                           }
                           
                           rec=rec+processConditionPart(rule)+nl;
                           if(isleaf)
                           {
                               rec=rec+": "+processLeafPart(srule)+"\n";
                               i=j;
                           }
                       }
                              
                   }
                   else if (rule.startsWith("Leaf"))
                   {
                            rec=rec+": "+processLeafPart(rule)+"\n";
                   }
                   i++;
               }
            }
        }
        }
        return rec;
    }
   private static String processConditionPart(String con)
   {
       String var=con.substring(con.indexOf(":")+1, con.indexOf("]"));
       int l=con.indexOf("]")+1;
       String op=con.substring(l, l+3).trim();
       String val=con.substring(con.indexOf(op)+op.length(), con.length()-1).trim();
       if(val.contains(":")){
           val=val.substring(val.indexOf(":")+1, val.indexOf("}"));
       }
       return var+" "+op+" "+val;
   }
   private static String processLeafPart(String con)
   {
       String rec=con.substring(con.indexOf("<")+1, con.indexOf(">"));
       return rec.substring(rec.indexOf(":")+1);
   }
   
    
    
    public static void createArffFile(String datasetName,String []attrNames,int []attrType,String [][]inDataset, String outFile)
    {        
        int noOfAttr=attrType.length;
        int numRecords=inDataset.length;
        File outF=new File(outFile);
        String rec="@relation "+datasetName+"\n\n";
        writeToFile(outF, rec);
        for(int j=0;j<noOfAttr;j++)
        {   
            rec="@attribute "+attrNames[j]+" ";
            if(attrType[j]==1)
            {
                rec=rec+ "real\n";
            }
            else
            {
                String []domainVal=findDomainValues(inDataset,j);
                rec=rec+ "{";
                for(int i=0;i<domainVal.length;i++)
                {
                    if(i==domainVal.length-1)
                        rec=rec+ domainVal[i];
                    else
                        rec=rec+ domainVal[i]+", ";
                }
                rec=rec+ "}\n";
            }
            appendToFile(outF, rec);
        }
        rec="\n@data\n";
        appendToFile(outF, rec);
        for(int j=0;j<numRecords;j++)
        {
            rec=String.join(",", inDataset[j]);
            appendToFile(outF, "\n"+rec);
        }
    }
    public static String[] findDomainValues(String [][]inDataset, int attrIndex)
    {
        List<String> domainValues = new ArrayList<>();
        int numRec=inDataset.length;
        String cval;
        for(int j=0;j<numRec;j++)
        {
            cval=inDataset[j][attrIndex];
            if(!domainValues.contains(cval))
            {
                domainValues.add(cval);
            }            
        }
        return domainValues.toArray(new String[0]);
    }
    //Find the number of conditions of each trees
     public static String[] processTrees(String []trees, int []Loc)
        {
            int numTree=Loc.length;
            for(int i=0;i<numTree;i++)
            {
                Loc[i]=0;
            }
            int count=0;
            int pos=0;
            int tlines=trees.length;
            ArrayList<String> Conditions=new ArrayList<String>();
            for(int i=0;i<tlines;i++)
            {
                if(!trees[i].equals("")){
                if(trees[i].startsWith("Tree"))
                {
               
                    Loc[count]=pos;
                    count++;    
                }
                else
                {
                    String []cac=separateConditionClass(trees[i]);
                    if(!cac[1].isEmpty())
                     {
                         StringTokenizer tokenizer= new StringTokenizer(cac[1], " {}():\t\n\r\f");
                         if(tokenizer.hasMoreTokens()) cac[1]=tokenizer.nextToken();
                         cac[0]=cac[0].trim()+": "+cac[1];
                     }                
                    Conditions.add(cac[0]);
                    pos++;
                }
                }
            }
            return Conditions.toArray(new String[0]);
        }
     
     //count the number of trees
  public static int countTrees(String []trees)
    {
        int count=0;
        int tlines=trees.length;
        for(int i=0;i<tlines;i++)
        {
            if(!trees[i].equals("") && trees[i].length()>4)
            {
                if(trees[i].substring(0, 4).equals("Tree"))
                {
                    count++;
                }
            }
        }
        return count;
    }
    public static int findAttrType(String currentAttr, String []attrNames, int []attrType)
        {
            int t=0;
            for(int i=0;i<attrNames.length;i++)
            {
                if(attrNames[i].equals(currentAttr))
                {
                   t=attrType[i]; break;
                }
            }
            return t;
        }
    public static int findAttrIndex(String currentAttr, String []attrNames)
        {
            int t=0;
            for(int i=0;i<attrNames.length;i++)
            {
                if(attrNames[i].equals(currentAttr))
                {
                   t=i; break;
                }
            }
            return t;
        }
        public static String[]convertStringToArray(String strVal) 
        {
            List<String> arrVal=new ArrayList<String>();
            StringTokenizer tokenizer= new StringTokenizer(strVal, " :\t\n\r\f");
            int n=tokenizer.countTokens();
            for(int i=0;i<n;i++)
            {
                arrVal.add(tokenizer.nextToken());
            }
            return arrVal.toArray(new String[0]);
        }
        public static boolean isConditionALeaf(String condition)
        {
                return condition.contains(":");
        }
        public static String [] separateConditionClass(String condition)
        {
            String []cac=new String[2];
            cac[0]=condition;
            cac[1]="";
            if(isConditionALeaf(condition))
            {
                 cac[1]=condition.substring(condition.indexOf(":")+1, condition.length());
                 cac[1]=cac[1].trim();
                 cac[0]=cac[0].substring(0, cac[0].indexOf(":"));
                 cac[0]=cac[0].trim();
            }
            return cac;
        }
        public static String preprocessTree(String treeStr)
        {            
            String []rules=treeStr.split("\n");
            String rec="";int t=0;
            for(int i=0;i<rules.length;i++)
            {
                if(rules[i].equals("RandomTree"))
                {
                    t++;
                    rec=rec+"Tree "+t+":\n";i++;
                   while(i<rules.length)
                   {
                       if(rules[i].equals("RandomTree"))
                       {
                                   i--;break;
                       }
                       else
                       {
                           int f=0;
                           if(rules[i].length()>16)
                           {
                               if(rules[i].substring(0, 16).equals("Size of the tree"))f=1;
                           }
                           if(f==0&&!rules[i].equals("") && !rules[i].equals("=========="))
                           {
                               rec=rec+rules[i]+"\n";
                           }
                           i++;
                       }
                   }
                }
            }
            return rec;
        }
        
     public static double findMaxValue(double[]data,int []attrType)
        {
            int noa=data.length;
            double tmax=Double.NEGATIVE_INFINITY;
            for(int j=0;j<noa;j++)
            {
                if(attrType[j]==1)
                {
                    if(data[j]>tmax)
                    {

                         tmax=data[j];
                    }
                }
            }
            return tmax;
        }
        public static int findMaxIndex(double[]data,int []attrType)
        {
            int noa=data.length;
            int index=-1;
            int tmpindex=-1;
            double tmax=Double.NEGATIVE_INFINITY;
            for(int j=0;j<noa;j++)
            {
                if(attrType[j]==1)
                {   if(tmpindex<0)tmpindex=j;
                    if(data[j]>tmax)
                    {

                         tmax=data[j];
                         index=j;
                    }
                }
            }
            if(index<0)index=tmpindex;
            return index;
        }

        public static double []calculateDistance(double []data1,double []data2,int []attrType)
        {
            int noa=data1.length;
            double []dist=new double[noa];
            for(int j=0;j<noa;j++)
            {
                if(attrType[j]==1)
                {
                 dist[j]=data1[j]-data2[j];
                }
            }
            return dist;
        }    
    /**
     * Reads the contents of the file to a <code>String</code> array, where
     * each line of the file is in an array cell. In the event of an error
     * the array will be null.
     *
     * @param file file to be read, assumes full path details
     * @return each line of the file in new cell in the array
     */
    public static String [] readFileAsArray(File file)
    {
        /** on reading the file each line is temporarily stored in a
         * LinkedList to determine the size of the array, then elements of the
         * LinkedList are copied to the String array.
         *
         * O(n), where n is number of lines in the file.
         */
        LinkedList <String> tempList = new LinkedList<String>();
        String [] fileArr=null;

        /** open the file and simply add each new line to the end of the list
         */
        try {
            FileReader fr = new FileReader(file);
            BufferedReader inFile = new BufferedReader(fr);
            String currLine = inFile.readLine();
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
            fileArr = new String[listSize];
            for(int i=0; i<listSize; i++)
            {
                fileArr[i] = tempList.removeFirst();
            }
        }
        return fileArr;
    }
    /**
     * Reads the contents of the file to a <code>String</code> array, where
     * each line of the file is in an array cell. In the event of an error
     * the array will be null.
     *
     * @param file file to be read, assumes full path details
     * @return each attribute value of the file in new cell in the array
     */
    public static String [][] readFileAs2DArray(File file)
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
     /**
     * Write output <code>String</code> to the passed <code>File</code>. This
     * method will replace any current contents of the file rather than append.
     *
     * @param file output file the contents are being written to
     * @param output the new contents of the file
     * @return a message indicating success, or an error message if there is a
     * problem writing to file
     */
    public static String writeToFile(File file, String output)
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
    public static String appendToFile(File file, String output)
    {
       /** simply open a file writer and write contents to the file */
       String returnStr = "";
       try{
           FileWriter fileWriter = new FileWriter(file,true);
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
     * This method will be used to change a file name by a supplied padding .

     *Author Geaur
     * @param filename file name to be changed
     * @param padding to add at the right of the filename
     * @return a changed file name, or an error message if there is a
     * problem writing to file
     */
    public static String changedFileName(String filename, String padding)
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
    public static String changedFileExtension(String filename, String Ext)
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
    public static String changetoNewFileName(String oldFile, String newFile)
    {
        File file=new File(oldFile);
        String p=file.getParent();
        return p+newFile;
    }
    /**
     * this function will remove all temporarily created files
     * 
     * @param fileName the name of the file to be removed
     */
    public static void removeFile(String fileName)
    {
        File tFile;boolean bool=false;    
        tFile=new File(fileName);
        if(tFile.exists())
        {
           bool= tFile.delete();
        }
    }
     /**
     * this function will remove all temporarily created files
     *
     * @param fileNames the names of the files to be removed
     * @param noOfFiles the number of files to be removed
     */
    public static void removeListOfFiles(String []fileNames, int noOfFiles)
    {
       File tFile;boolean bool=false;
       for(int i=0; i<noOfFiles;i++)
       {
        tFile=new File(fileNames[i]);
        if(tFile.exists())
        {
           bool= tFile.delete();
        }
       }
    }
   }
}
