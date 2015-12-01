import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public class ID3_DecisionTree {
	/** Entry point of program
	 * @param args
	 */
	public static void main(String[] args) {
		new ID3_DecisionTree();
	}

	int valueOfImpurity = 2;
	
	/** This is the main function which computes the Decision tree 
	 * 	and displays the accuracy of training and validation file.
	 */
	public ID3_DecisionTree() {
		DataInputStream dis = new DataInputStream(System.in);
		String filePathToTraining = null, filePathToValidation = null;
		try {
			
					
			//Change the filepath to run the program
			filePathToTraining = "C:\\Users\\MaGesh\\Desktop\\hw1\\training.txt";
			filePathToValidation = "C:\\Users\\MaGesh\\Desktop\\hw1\\validation.txt";
			System.out.println("Enter 1 for Entropy Calculation and 2 for Misclassification Impurity");
			valueOfImpurity = Integer.parseInt(dis.readLine());

		} catch (Exception e) {
			e.printStackTrace();
		}
		List<InputRecord> trainingLists = recordsList(filePathToTraining);
		
		
		calculateRootNode rootValueNode = new calculateRootNode(-1, "rootValueNode"); //Passing a sample value to store it as the RootNode so that Program ..
		//doesn't return Null pointer Exception
		buildTree(trainingLists,rootValueNode);		
		treeValidation(filePathToTraining, rootValueNode);		
		treeValidation(filePathToValidation, rootValueNode);		

	}

	/**This is the function which classifies the data correctly and produces the accuracy percentage 
	 * @param fileName
	 * @param rootValueNode
	 */
	private void treeValidation(String fileName, calculateRootNode rootValueNode) {
		List<InputRecord> validationRecords = recordsList(fileName);
		int positiveValue=0,errorValue=0;
		for (InputRecord record : validationRecords) {
			if(parseRecordInTree(record,rootValueNode)){
				positiveValue++;
			}else{
				errorValue++;
			}
		}
		System.out.println("ID3 Algorihm Classified datasets for: "+fileName);
		System.out.println("Leaf Node Classification:"+positiveValue+" \nLeaf Nodes Misclassification:"+errorValue);
		double total = positiveValue+errorValue;
		System.out.println("Percentage of accuracy : " + (positiveValue/total*100)+"%");
	}

	/** This function traverse the 'record' through decision tree whose rootValueNode node is 'rootValueNode'
	 * 	
	 * @param record
	 * @param rootValueNode
	 * @return
	 */
	private boolean parseRecordInTree(InputRecord record, calculateRootNode rootValueNode) {
		calculateRootNode parent = rootValueNode;	
		for(;;) {
			boolean flag = true;	
			for (calculateRootNode decisionNode : parent.nextNode) {		
				if (record.getValue(decisionNode.getAttId()).equals(decisionNode.getValue())) { 
					flag=false;		
					if (decisionNode.nextNode.size()==0) {	
						if (decisionNode.getOutput()==record.getSet()) { 
							return true;		
						}else {
							return false;		
						}
					}else{						
						parent = decisionNode;	
					}
					break;
				}				
			}
			if (flag) {			
				return false;
			}
		}
	}

	/** Recursion Method which computes a decision tree by traversing the given 'records' dataset
	 * if Chi Square Statistics value is not greater the threshold then stop growing node
	 * @param records
	 * @param parent
	 */
	private void buildTree(List<InputRecord> records, calculateRootNode parent) {
		double entropy ;
		if( valueOfImpurity == Constant_Value.entropyImpurityValue){
			entropy = calculateEntropy(records);	
		}else {
			entropy = calculateMisclassifiedValue(records);	
		}
		if (entropy == 0) {						 
			parent.setOutput(records.get(0).getSet());	
			return;										
		}else{
			int attributeNumber = findBestAttribute(records);
			if (calculateChiSquareList(records,attributeNumber) > Constant_Value.ChiSquareValue) {	 
				Map<String, List<InputRecord>> map = listOfDiscreteRecords(records, attributeNumber);	
				Iterator it = map.entrySet().iterator();
				while (it.hasNext()) {					
					Map.Entry<String,List<InputRecord>> pairs = (Map.Entry<String,List<InputRecord>>)it.next();
					List<InputRecord> recordClass = (List<InputRecord>) pairs.getValue();
					calculateRootNode child = new calculateRootNode(attributeNumber, pairs.getKey().toString());	
					parent.addNextNode(child);			
					buildTree(recordClass, child);		
				}
			}else{	
				parent.setOutput(decisionLeafNode(records));	
				return;											
			}
		}
	}

	

	/**This function calculates the entropy 
	 * entropy = summation for each nodes -(p/n)log_2(p/n)
	 * as all labels are same for that record set
	 * @param records
	 * @return Entropy
	 */
	public double calculateEntropy(List<InputRecord> records) {
		Map<String, List<InputRecord>> recordMap = sortedMapList(records);		
		double n = records.size();
		double entropy = 0;
		Iterator it = recordMap.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry pairs = (Map.Entry)it.next();
			List<InputRecord> recordClass = (List<InputRecord>) pairs.getValue();
			double p = recordClass.size();
			if (p==0 || p==n) {
				return 0;
			}
			double x=p/n;
			double x1=Math.log(x)/Math.log(2);
			double x2=x*x1;
			entropy -= x2;			
		}
		return entropy;
	}

	/**Calculates Misclassified records
	 * 
	 * @param records
	 * @return 
	 */
	public double calculateMisclassifiedValue(List<InputRecord> records) {
		Map<String, List<InputRecord>> recordMap = sortedMapList(records);		
		double p = recordMap.get("+").size();
		double n = recordMap.get("-").size();	
		if (p > n) {
			return n / (p+n);
		}else{
			return p / (p+n);
		}
	}

	
	/** Loads the records(dataset) to the recordlists 
	 * @param file
	 * @return ArrayList of loaded Records
	 */
	private List<InputRecord> recordsList(String file) {
		List<InputRecord> records = new ArrayList<InputRecord>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));	
			boolean fstRec = true;
			String line;
			while ((line = br.readLine())!=null) {		
				if (fstRec) {
					Constant_Value.NumberOfAttributes = line.substring(0,line.indexOf(" ")).length();	// Set number of Attributes to number of characters in first line of file
					fstRec=false;
				}
				InputRecord record ;
				String value = line.substring(line.indexOf(" ")+1);
				if (value.equals("+")) {
					record = new InputRecord(line.substring(0,line.indexOf(" ")), 1); 
				}else{
					record = new InputRecord(line.substring(0,line.indexOf(" ")), 0); 
				}
				records.add(record);
			}
			System.out.println();
			br.close();
		} catch (FileNotFoundException e) {
			System.out.println("training.txt : File Not found");
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return records;
	}


	/**sortedMaplist sorts out the class with '+' and '-' and returns a map as a Key 
	 * and ArrayList of corresponding records sorted
	 * from a rest of records, passed as a parameter
	 * @param records
	 * @return Map containing labels as key and their corresponding records
	 */
	public static Map<String, List<InputRecord>> sortedMapList(List<InputRecord> records){
		List<InputRecord> positiveRec = new ArrayList<InputRecord>();
		List<InputRecord> negativeRec = new ArrayList<InputRecord>();
		for (InputRecord record : records) {
			if (record.getSet()==1) {				
				positiveRec.add(record);			
			}else{
				negativeRec.add(record);			
			}
		}
		Map<String, List<InputRecord>> map = new LinkedHashMap<String, List<InputRecord>>();
		map.put("+", positiveRec);					
		map.put("-", negativeRec);					
		return map;
	}

	/**Returns which labels are majority, are they '+'(represented by 1) or '-'(represented by 0)
	 * taking in the ArrayList of records as parameter
	 * @param records
	 * @return labelid
	 */
	public static int decisionLeafNode(List<InputRecord> records){
		Map<String, List<InputRecord>> recordMap = sortedMapList(records);
		if (recordMap.get("+").size() > recordMap.get("-").size()) {
			return 1;				
		}else{
			return 0;				
		}
	}
	/**	Calculates information gain of records
	 * @param records
	 * @param attributeNumber
	 * @return informationGain
	 */
	public double calculateInformationGain(List<InputRecord> records, int attributeNumber) {
		Map<String, List<InputRecord>> map = listOfDiscreteRecords(records, attributeNumber);	 
		
		double n = records.size();		
		double informationGainValue;
		if( valueOfImpurity == Constant_Value.entropyImpurityValue){	
			informationGainValue = calculateEntropy(records);			
		}else {
			informationGainValue = calculateMisclassifiedValue(records);	
		}
		Iterator it = map.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<String,List<InputRecord>> pairs = (Map.Entry<String,List<InputRecord>>)it.next();
			List<InputRecord> recordClass = (List<InputRecord>) pairs.getValue();
			double rcSize = recordClass.size();
			
			double entropy ;
			if( valueOfImpurity == Constant_Value.entropyImpurityValue){
				entropy = calculateEntropy(recordClass);		
			}else {
				entropy = calculateMisclassifiedValue(recordClass);	
			}
			informationGainValue -= (rcSize/n)*entropy;	 
			
		}
		return informationGainValue;
	}
	/** finds and returns the best attribute node which has higher information gain
	 *  Takes in the records based on which decision has to be taken using information Gain value
	 * @param records
	 * @return findBestAttribute
	 */
	public int findBestAttribute(List<InputRecord> records) {
		double informationGainValue=-9999999;					
		int splitAttribute=0;							
		boolean fl=true;							
		Map<Integer, String> attributes = records.get(0).getRecord();	
		Iterator it = attributes.entrySet().iterator();	
		while (it.hasNext()) {						
			Map.Entry pairs = (Map.Entry)it.next();
			Integer integer = (Integer) pairs.getKey();	
			if (fl) {
				splitAttribute=integer;
				fl = false;
			}
			double ig = calculateInformationGain(records, integer);	
			if (ig>informationGainValue) {							
				informationGainValue = ig;
				splitAttribute = integer;					
			}
		}
		return splitAttribute;								
	}




	/**
	 * This function adds the records to the tree of nodes.
	 * @param records
	 * @param attributeNumber
	 * @return 
	 */
	public Map<String, List<InputRecord>> listOfDiscreteRecords(List<InputRecord> records, int attributeNumber){
		Map<String, List<InputRecord>> map = new LinkedHashMap<String, List<InputRecord>>();
		for (InputRecord record : records) {
			String str = record.getValue(attributeNumber);
			if (map.get(str) == null) {
				List<InputRecord> discrete = new ArrayList<InputRecord>();
				discrete.add(record);
				map.put(str, discrete);
			}else{
				map.get(str).add(record);
			}
		}
		return map;
	}

	/** The method returns Chi-square stat_value value 
	 * @param records
	 * @param attributeNumber
	 * @return
	 */
	public double calculateChiSquareList(List<InputRecord> records,int attributeNumber) {
		Map<String, List<InputRecord>> map = listOfDiscreteRecords(records, attributeNumber);	 
		Map<String, List<InputRecord>> recordMap = sortedMapList(records);	
		double p = recordMap.get("+").size();		
		double n = recordMap.get("-").size();		
		double stat_value = 0;						
		Iterator it = map.entrySet().iterator();
		while (it.hasNext()) {						
			Map.Entry<String,List<InputRecord>> pairs = (Map.Entry<String,List<InputRecord>>)it.next();
			List<InputRecord> recordClass = (List<InputRecord>) pairs.getValue();
			Map<String, List<InputRecord>> rm = sortedMapList(recordClass);	
			double pi = rm.get("+").size();		
			double ni = rm.get("-").size();		
			double p_i = p*((pi+ni)/(p+n));
			double n_i = n*((pi+ni)/(p+n));
			stat_value += (((pi-p_i)*(pi-p_i)/p_i) + ((ni-n_i)*(ni-n_i)/n_i)  ); 
		}
		return stat_value;
	}
}

/**
 * constant values such as chi_square, impurity values are declared here
 * 
 *
 */
class Constant_Value {
	public static final int entropyImpurityValue = 1;
	public static final int misclassificationImpurityValue = 2;
	public static final double ChiSquareValue = 11.34;
	public static int NumberOfAttributes;
}

/**
 * Object of this Class represents a node of the decision tree
 * 
 *
 */
class calculateRootNode {
	List<calculateRootNode> nextNode;	
	int attributeNumber;						
	String value;					
	int output;						
	public calculateRootNode( int attributeNumber, String value) {
		super();
		nextNode = new ArrayList<calculateRootNode>();
		this.attributeNumber = attributeNumber;
		this.value = value;
		output = -1;
	}
	public int getAttId() {
		return attributeNumber;
	}
	public void setAttId(int attributeNumber) {
		this.attributeNumber = attributeNumber;
	}
	public String getValue() {
		return value;
	}
	public void setValue(String value) {
		this.value = value;
	}
	public calculateRootNode getNextNode(int index) {
		return nextNode.get(index);
	}
	public void addNextNode(calculateRootNode nextNode) {
		this.nextNode.add(nextNode);
	}
	public int getOutput() {
		return output;
	}
	public void setOutput(int value) {
		this.output = value;
	}
	
}

/** To store records in an object format.
 * 
 *
 */
class InputRecord {
	private Map<Integer, String> record;
	private int set;

	/** 
	 * 	 
	 * @param str
	 * @param value
	 */
	public InputRecord(String str, int value) {	
		if (str.length() != Constant_Value.NumberOfAttributes ) {
			System.out.println("invalid record");
		}else{
			record = new LinkedHashMap<Integer, String>();
			for (int i = 0; i < str.length(); i++) {
				record.put(i, ""+str.charAt(i));	
			}
			this.set=value;							
		}
	}

	public Map<Integer, String> getRecord() {
		return record;
	}

	/**
	 * returns value at from map where attribute index is 'key'
	 * @param value
	 */
	public String getValue(int key) {
		return record.get(key);
	}

	public void setRecord(Map<Integer, String> record) {
		this.record = record;
	}
	public int getSet() {
		return set;
	}
	public void setSet(int value) {
		this.set = value;
	}

	/**
	 * remove Attribute index 'attributeNumber' entry from Map
	 * @param attributeNumber
	 */
	public void deleteValue(int attributeNumber) {
		record.remove(attributeNumber);
	}

}

