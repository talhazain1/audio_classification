package src;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import javax.sound.sampled.*;
//import org.apache.commons.math4.transform.*;
import org.jtransforms.fft.DoubleFFT_1D;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;  // You can choose other classifiers from WEKA
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.attributeSelection.*;



public class AudioPlayerGUI extends JFrame {
    private JList<String> fileList;
    private DefaultListModel<String> listModel;
    private JButton playButton, stopButton;
    private Clip currentClip;
    private JComboBox<String> modelSelector;
    private JTextArea resultsArea;
    private JButton classifyButton;
    private JButton modelDetailsButton;

    // for the ML file organization
    private ArrayList<String> allFiles = new ArrayList<>();
    private ArrayList<String> trainingFiles = new ArrayList<>();
    private ArrayList<String> testingFiles = new ArrayList<>();

    public AudioPlayerGUI() {
    // Set up JFrame
    setTitle("Audio Player and Classifier");
    setSize(900, 400);
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // Create a list model and JList to display audio files
    listModel = new DefaultListModel<>();
    fileList = new JList<>(listModel);
    JScrollPane scrollPane = new JScrollPane(fileList);

    // Create play and stop buttons
    playButton = new JButton("Play Selected File");
    stopButton = new JButton("Stop Audio");
    stopButton.setEnabled(false);

    // Model selector dropdown
    String[] models = {"Decision Tree (J48)", "Logistic Regression"};
    modelSelector = new JComboBox<>(models);

    // Buttons
    classifyButton = new JButton("Classify Selected File");
    modelDetailsButton = new JButton("Get Model Details");

    // Action listeners for buttons
    classifyButton.addActionListener(e -> classifySelectedAudio());
    modelDetailsButton.addActionListener(e -> showModelDetails());

    // Results area
    resultsArea = new JTextArea(10, 40);
    resultsArea.setEditable(false);
    JScrollPane resultsScrollPane = new JScrollPane(resultsArea);

    // Layout setup
    JPanel controlPanel = new JPanel();
    controlPanel.setLayout(new GridLayout(3, 1)); // Vertical alignment
    controlPanel.add(new JLabel("Select Model:"));
    controlPanel.add(modelSelector);
    controlPanel.add(classifyButton);
    controlPanel.add(modelDetailsButton); // Add the Model Details button

    // Layout setup for main GUI
    setLayout(new BorderLayout());
    add(scrollPane, BorderLayout.WEST);
    add(controlPanel, BorderLayout.NORTH);
    add(resultsScrollPane, BorderLayout.CENTER);

    // Load audio files into the list model
    loadAudioFiles();
    splitdata();
    clearFiles();
    generateTrainingFeaturesCSV();
}
    private weka.core.Instance createInstance(double[] features, Instances datasetStructure) {
    // Create a new instance with the same structure as the dataset
    weka.core.Instance instance = new weka.core.DenseInstance(features.length + 1); // +1 for the class attribute
    instance.setDataset(datasetStructure);

    // Set the feature values for the instance
    for (int i = 0; i < features.length; i++) {
        instance.setValue(i, features[i]);
    }

    return instance;
}

private void showModelDetails() {
        String selectedModel = (String) modelSelector.getSelectedItem();
        if (selectedModel == null) {
            JOptionPane.showMessageDialog(this, "Please select a classification model.", "Error", JOptionPane.WARNING_MESSAGE);
            return;
        }

        try {
            // Load the dataset
            Instances data = loadDataset("features.arff");

            // Choose classifier
            Classifier classifier = null;
            if (selectedModel.equals("Decision Tree (J48)")) {
                classifier = new J48();
            } else if (selectedModel.equals("Logistic Regression")) {
                classifier = new Logistic();
            }

            if (classifier != null) {
                // Perform cross-validation
                String results = evaluateModelWithDetails(data, classifier, selectedModel);

                // Show results in a popup
                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this, results, "Model Details", JOptionPane.INFORMATION_MESSAGE);
                });
            }

        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(this, "An error occurred while fetching model details.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }
private String evaluateModelWithDetails(Instances data, Classifier classifier, String classifierName) throws Exception {
        weka.classifiers.Evaluation evaluation = new weka.classifiers.Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new java.util.Random(1));

        StringBuilder results = new StringBuilder();
        results.append("Results for ").append(classifierName).append(":\n");
        results.append("Precision: ").append(evaluation.precision(1)).append("\n");
        results.append("Recall: ").append(evaluation.recall(1)).append("\n");
        results.append("Accuracy: ").append(evaluation.pctCorrect()).append("%\n");
        results.append("F1-Score: ").append(evaluation.fMeasure(1)).append("\n");
        results.append("\nSummary:\n").append(evaluation.toSummaryString()).append("\n");
        results.append("Class Details:\n").append(evaluation.toClassDetailsString()).append("\n");
        results.append("Confusion Matrix:\n").append(evaluation.toMatrixString()).append("\n");

        return results.toString();
    }

    private void loadAudioFiles() {
        // Load music files
        File musicDir = new File("audio/music");
        File[] musicFiles = musicDir.listFiles((dir, name) -> name.endsWith(".wav"));
        if (musicFiles != null) {
        for (File file : musicFiles) {
            String filePath = "music/" + file.getName();
            listModel.addElement(filePath); // Add to JList model
            allFiles.add(filePath);        // Add to allFiles array
        }

        // Load speech files
        File speechDir = new File("audio/speech");
        File[] speechFiles = speechDir.listFiles((dir, name) -> name.endsWith(".wav"));
        if (speechFiles != null) {
        for (File file : speechFiles) {
            String filePath = "speech/" + file.getName();
            listModel.addElement(filePath); // Add to JList model
            allFiles.add(filePath);        // Add to allFiles array
        }
        }
        }
            System.out.println("Total Files: " + allFiles.size());
    }

    private void splitdata() {
        // Shuffles the list of files to ensure randomness
        Collections.shuffle(allFiles);

        // First 2/3 used for training
        int trainSize = (int) (allFiles.size() * 0.67);
        trainingFiles = new ArrayList<>(allFiles.subList(0, trainSize));

// Remaining 1/3 for testing
        testingFiles = new ArrayList<>(allFiles.subList(trainSize, allFiles.size()));
           System.out.println("Training Files: " + trainingFiles.size());
        System.out.println("Testing Files: " + testingFiles.size());
    }

    private void playAudio(String filePath) {
        try {
            File audioFile = new File(filePath);
            AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
            currentClip = AudioSystem.getClip();  // Assign the Clip to the currentClip variable
            currentClip.open(audioStream);
            currentClip.start();
            stopButton.setEnabled(true);  // Enable the Stop button when the audio starts
        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error playing the audio file.");
        }
    }

    private void stopAudio() {
        if (currentClip != null && currentClip.isRunning()) {
            currentClip.stop();
            currentClip.close();
            stopButton.setEnabled(false);  // Disable the Stop button after stopping the audio
        }
    }
private void generateTrainingFeaturesCSV() {
    try {
        // Clear the CSV file before writing new data
        clearFiles();

        // Loop through each training file
        for (String filePath : trainingFiles) {
            String label = filePath.startsWith("music/") ? "1" : "0"; // Label for the file (1 for music, 0 for speech)

            // Extract features for the file
            extractFeatures("audio/" + filePath, label);
        }

        System.out.println("Features for training files saved to features.csv successfully!");
    } catch (Exception e) {
        e.printStackTrace();
        JOptionPane.showMessageDialog(this, "Error generating features for training files.");
    }
}

public void buildModel(String arffFilePath) {
    try {
        // Load the dataset
        DataSource source = new DataSource(arffFilePath);
        Instances data = source.getDataSet();

        // Set the class index (last attribute is typically the label)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Perform 10-fold cross-validation with J48 and Logistic classifiers
        evaluateWithCrossValidation(data, new J48(), "Decision Tree (J48)");
        evaluateWithCrossValidation(data, new Logistic(), "Logistic Regression");

    } catch (Exception e) {
        e.printStackTrace();
    }
}
    

private String evaluateWithCrossValidation(Instances data, Classifier classifier, String classifierName) {
    StringBuilder results = new StringBuilder();
    try {
        // Perform 10-fold cross-validation
        weka.classifiers.Evaluation evaluation = new weka.classifiers.Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new java.util.Random(1));

        // Prepare results as a string
        results.append("Results for ").append(classifierName).append(":\n");
        results.append("Precision: ").append(evaluation.precision(1)).append("\n");
        results.append("Recall: ").append(evaluation.recall(1)).append("\n");
        results.append("Accuracy: ").append(evaluation.pctCorrect()).append("%\n");
        results.append("F1-Score: ").append(evaluation.fMeasure(1)).append("\n");
        results.append("\nSummary:\n").append(evaluation.toSummaryString()).append("\n");
        results.append("Class Details:\n").append(evaluation.toClassDetailsString()).append("\n");
        results.append("Confusion Matrix:\n").append(evaluation.toMatrixString()).append("\n");
    } catch (Exception e) {
        e.printStackTrace();
        results.append("An error occurred during evaluation.\n");
    }
    return results.toString();
}
   
    public void featureSelection(String arffFilePath) {
    try {
        // Load the dataset
        DataSource source = new DataSource(arffFilePath);
        Instances data = source.getDataSet();

        // Set the class index (last attribute is typically the label)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Perform feature selection
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();  // Evaluator
        BestFirst search = new BestFirst();         // Search algorithm
        selector.setEvaluator(eval);
        selector.setSearch(search);
        selector.SelectAttributes(data);

        // Output selected attributes
        int[] selectedAttributes = selector.selectedAttributes();
        System.out.println("Selected attributes:");
        for (int attrIndex : selectedAttributes) {
            System.out.println(data.attribute(attrIndex).name());
        }
     } catch (Exception e) {
        e.printStackTrace();
     }
}

private void classifySelectedAudio() {
    String selectedFile = fileList.getSelectedValue();
    if (selectedFile == null) {
        JOptionPane.showMessageDialog(this, "Please select an audio file to classify.", "Error", JOptionPane.WARNING_MESSAGE);
        return;
    }

    String selectedModel = (String) modelSelector.getSelectedItem();
    if (selectedModel == null) {
        JOptionPane.showMessageDialog(this, "Please select a classification model.", "Error", JOptionPane.WARNING_MESSAGE);
        return;
    }

    try {
        // Extract features for the selected file
        double[] features = extractFeaturesForFile("audio/" + selectedFile); // Ensure the path matches your file structure

        // Load dataset structure for reference
        Instances datasetStructure = loadDataset("features.arff");

        // Create an instance for the selected file
        weka.core.Instance instance = createInstance(features, datasetStructure);

        // Choose classifier
        Classifier classifier = null;
        if (selectedModel.equals("Decision Tree (J48)")) {
            classifier = new J48();
        } else if (selectedModel.equals("Logistic Regression")) {
            classifier = new Logistic();
        }

        if (classifier != null) {
            // Train the classifier on the full dataset
            classifier.buildClassifier(datasetStructure);

            // Classify the selected instance
            double prediction = classifier.classifyInstance(instance);
            String classLabel = datasetStructure.classAttribute().value((int) prediction);

            // Show results in a popup
            SwingUtilities.invokeLater(() -> {
                JOptionPane.showMessageDialog(this, 
                    "The file \"" + selectedFile + "\" is classified as: " + classLabel, 
                    "Classification Result", JOptionPane.INFORMATION_MESSAGE);
            });
        }

    } catch (Exception ex) {
        ex.printStackTrace();
        JOptionPane.showMessageDialog(this, "An error occurred during classification.", "Error", JOptionPane.ERROR_MESSAGE);
    }
}



    public Instances loadDataset(String arffFilePath) throws Exception {
        DataSource source = new DataSource(arffFilePath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }


    // The three fast fourier transformation features are:
    /*
   1.  Spectral Centroid: Measures the "center of mass" of the spectrum.
   2.  Spectral Bandwidth: Measures the width of the spectrum.
   3.  Spectral Roll Off: Measures the point where a certain percentage of the
       total spectral energy is contained.
     */
    private void extractFeatures(String filePath, String label) {
    try {
        File audioFile = new File(filePath);
        AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
        AudioFormat format = audioStream.getFormat();
        int sampleRate = (int) format.getSampleRate();
        byte[] buffer = new byte[1024];
        int bytesRead;
        double rms = 0.0;
        int zeroCrossings = 0;
        int previousSample = 0;
        double maxAmplitude = 0.0;
        int totalSamples = 0; // To track number of samples
        ArrayList<Double> audioData = new ArrayList<>(); // To store the audio data for FFT

        // Check if audio is 16-bit or 8-bit
        boolean is16Bit = format.getSampleSizeInBits() == 16;

        // Process audio samples and extract features
        while ((bytesRead = audioStream.read(buffer)) != -1) {
            for (int i = 0; i < bytesRead; i++) {
                int sample = buffer[i];

                if (is16Bit) {
                    // For 16-bit audio, combine two bytes to form a short sample
                    sample = (buffer[i] & 0xFF) | ((buffer[i + 1] & 0xFF) << 8);
                    i++;  // Skip the next byte
                } else {
                    // For 8-bit audio, use the byte directly
                    sample = buffer[i];
                }

                // Add the sample to audioData for FFT
                audioData.add((double) sample);

                // Amplitude (Max absolute value)
                maxAmplitude = Math.max(maxAmplitude, Math.abs(sample));

                // RMS Energy
                rms += Math.pow(sample, 2);
                totalSamples++; // counts number of samples

                // Zero-Crossing Rate
                if ((previousSample > 0 && sample <= 0) || (previousSample < 0 && sample >= 0)) {
                    zeroCrossings++;
                }
                previousSample = sample;
            }
        }

        // Calculate RMS value
        if (totalSamples > 0) {
            rms = Math.sqrt(rms / totalSamples);
        } else {
            rms = 0.0;
        }

        // Ensure audioData has data before performing FFT
        if (audioData.isEmpty()) {
            System.out.println("No audio data found.");
            return; // Exit early if audio data is empty
        }

        // Convert the audio data list to an array for FFT
        double[] audioArray = audioData.stream().mapToDouble(Double::doubleValue).toArray();

        // Perform FFT to extract frequency-domain features
        DoubleFFT_1D fft = new DoubleFFT_1D(audioArray.length);
        fft.realForward(audioArray);

        // Extract spectral features after FFT
        double spectralCentroid = calculateSpectralCentroid(audioArray);
        double spectralBandwidth = calculateSpectralBandwidth(audioArray, spectralCentroid);
        double spectralRolloff = calculateSpectralRolloff(audioArray, 0.85);  // 85% rolloff point

        // Print extracted features for debugging
        // System.out.println("Features for " + filePath + ":");
        // System.out.println("Max Amplitude: " + maxAmplitude);
        // System.out.println("RMS Energy: " + rms);
        // System.out.println("Zero-Crossing Rate: " + zeroCrossings);
        // System.out.println("Spectral Centroid: " + spectralCentroid);
        // System.out.println("Spectral Bandwidth: " + spectralBandwidth);
        // System.out.println("Spectral Rolloff: " + spectralRolloff);

        // Save features to CSV
        saveFeaturesToCSV(maxAmplitude, rms, zeroCrossings, label, spectralCentroid, spectralBandwidth, spectralRolloff);

        // Convert CSV to ARFF after saving the features
        convertCsvToArff("features.csv", "features.arff");

    } catch (Exception e) {
        e.printStackTrace();
        JOptionPane.showMessageDialog(this, "Error extracting features.");
    }
}

private double[] extractFeaturesForFile(String filePath) {
    try {
        // Load the audio file
        File audioFile = new File(filePath);
        AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
        AudioFormat format = audioStream.getFormat();

        int sampleRate = (int) format.getSampleRate();
        byte[] buffer = new byte[1024];
        int bytesRead;

        // Variables for feature calculations
        double rms = 0.0;
        int zeroCrossings = 0;
        int previousSample = 0;
        double maxAmplitude = 0.0;
        int totalSamples = 0;
        ArrayList<Double> audioData = new ArrayList<>();

        // Check if audio is 16-bit or 8-bit
        boolean is16Bit = format.getSampleSizeInBits() == 16;

        // Process audio samples and calculate time-domain features
        while ((bytesRead = audioStream.read(buffer)) != -1) {
            for (int i = 0; i < bytesRead; i++) {
                int sample = buffer[i];

                if (is16Bit) {
                    // For 16-bit audio, combine two bytes to form a short sample
                    sample = (buffer[i] & 0xFF) | ((buffer[i + 1] & 0xFF) << 8);
                    i++; // Skip the next byte
                } else {
                    // For 8-bit audio, use the byte directly
                    sample = buffer[i];
                }

                // Add the sample to audioData for FFT
                audioData.add((double) sample);

                // Calculate maximum amplitude
                maxAmplitude = Math.max(maxAmplitude, Math.abs(sample));

                // Calculate RMS energy
                rms += Math.pow(sample, 2);
                totalSamples++;

                // Calculate zero-crossing rate
                if ((previousSample > 0 && sample <= 0) || (previousSample < 0 && sample >= 0)) {
                    zeroCrossings++;
                }
                previousSample = sample;
            }
        }

        // Calculate RMS energy
        rms = totalSamples > 0 ? Math.sqrt(rms / totalSamples) : 0.0;

        // Convert audio data to an array for FFT
        double[] audioArray = audioData.stream().mapToDouble(Double::doubleValue).toArray();

        // Perform FFT to calculate frequency-domain features
        DoubleFFT_1D fft = new DoubleFFT_1D(audioArray.length);
        fft.realForward(audioArray);

        // Calculate spectral features
        double spectralCentroid = calculateSpectralCentroid(audioArray);
        double spectralBandwidth = calculateSpectralBandwidth(audioArray, spectralCentroid);
        double spectralRolloff = calculateSpectralRolloff(audioArray, 0.85); // 85% rolloff point

        // Return extracted features as a double array
        return new double[]{maxAmplitude, rms, zeroCrossings, spectralCentroid, spectralBandwidth, spectralRolloff};

    } catch (Exception e) {
        e.printStackTrace();
        JOptionPane.showMessageDialog(null, "Error extracting features from file: " + filePath);
        return new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Return default features on error
    }
}


    // Measures the spread of the spectrum around the centroid.
    private double calculateSpectralBandwidth(double[] spectrum, double centroid) {
        double sum = 0.0; // accumulate the weighted squared distances from the centroid.
        double totalMagnitude = 0.0; // of the spectrum

        // Iterate through the first half of the spectrum (FFT is symmetrical)
        for (int i = 0; i < spectrum.length / 2; i++) {
            double magnitude = Math.abs(spectrum[i]); // at current freq bin
            sum += magnitude * Math.pow(i - centroid, 2);
            totalMagnitude += magnitude;
        }

        // calc spectral bandwidth
        return totalMagnitude > 0 ? Math.sqrt(sum / totalMagnitude) : 0.0;
    }

    private double calculateSpectralRolloff(double[] spectrum, double rolloffPercentage) {
        double totalEnergy = 0.0;
        //  Calculate the total energy of the first half of the spectrum.
        for (int i = 0; i < spectrum.length / 2; i++) {
            totalEnergy += Math.abs(spectrum[i]);
        }

        double threshold = totalEnergy * rolloffPercentage; // energy threshold for the rolloff point
        double cumulativeEnergy = 0.0; // track the cumulative energy as we iterate through the spectrum.

        // iterate through first half of the spectrum
        for (int i = 0; i < spectrum.length / 2; i++) {
            cumulativeEnergy += Math.abs(spectrum[i]);

            // If cumulative energy meets or exceeds the threshold, return the current frequency bin index.
            if (cumulativeEnergy >= threshold) {
                return i; // Frequency bin
            }
        }

        return spectrum.length / 2 - 1; // Default to highest frequency bin if not reached
    }


    // Measures the "center of mass" of the spectrum.
    private double calculateSpectralCentroid(double[] spectrum) {
        double weightedSum = 0.0; //weighted sum of frequency magnitudes.
        double totalMagnitude = 0.0; // total magnitude of the spectrum.

        // iterates through the first half of the spectrum, since FFT output is symmetrical,
        // only need the FIRST HALF for spectral features.
        for (int i = 0; i < spectrum.length / 2; i++) { // Only half due to symmetry in FFT
            double magnitude = Math.abs(spectrum[i]); // at the current frequency bin.
            weightedSum += i * magnitude;
            totalMagnitude += magnitude;
        }

        // calc spectral centroid
        return totalMagnitude > 0 ? weightedSum / totalMagnitude : 0.0;
    }

    // Method to save the features to CSV
    private void saveFeaturesToCSV(double maxAmplitude, double rms, int zeroCrossings, String label, double spectralCentroid, double spectralBandwidth, double spectralRolloff) {
    try {
        BufferedWriter writer = new BufferedWriter(new FileWriter("features.csv", true));  // Open in append mode
        String nominalLabel = label.equals("1") ? "music" : "speech"; // Map numeric to nominal
        String data = maxAmplitude + "," + rms + "," + zeroCrossings + "," + spectralCentroid + "," +
                      spectralBandwidth + "," + spectralRolloff + "," + nominalLabel;
        writer.write(data);
        writer.newLine();  // Add new line for the next row
        writer.close();
    } catch (IOException e) {
        e.printStackTrace();
        JOptionPane.showMessageDialog(this, "Error writing to CSV.");
    }
}


    // Method to train the model using WEKA
    private void trainModel() {
        try {
            // Convert CSV to ARFF after saving the features
            convertCsvToArff("features.csv", "features.arff");

            // Load the ARFF file into WEKA
            DataSource source = new DataSource("features.arff");
            Instances data = source.getDataSet();
            // Set the class index to the last attribute (label)
            data.setClassIndex(data.numAttributes() - 1); // Set the last attribute as the class


            // Train a classifier
            Classifier classifier = new J48();  // J48 is a decision tree classifier (you can use other classifiers)
            classifier.buildClassifier(data);

            // Save the classifier model
            weka.core.SerializationHelper.write("music_model.model", classifier);
            System.out.println("Model trained and saved.");

            // Test the model with the testing data
            testModel(classifier);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Method to test the model using WEKA
    private void testModel(Classifier classifier) {
        try {
            // Load the ARFF file
            DataSource source = new DataSource("features.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // header for the results
            System.out.println("#, Model Output, Ground Truth Label");

            // For testing, use the testing files
            for (String filePath : testingFiles) {
                // Extract features for the testing file (extract and test each file individually)
                double maxAmplitude = 0.0;
                double rms = 0.0;
                int zeroCrossings = 0;
                String label = filePath.startsWith("music/") ? "1" : "0"; // Label for the file (1 for music, 0 for speech)

                // extract features of the current file
                extractFeatures(filePath, label);
                
                // Create a new instance for the test file using the extracted features
                Instance testInstance = createTestInstance(maxAmplitude, rms, zeroCrossings, label, data);

                // Classify the instance using the trained classifier
                double prediction = classifier.classifyInstance(testInstance);

                // Output the prediction result
                String predictedClass = (prediction == 1.0) ? "Music" : "Speech"; // 1 = music, 0 = speech
                System.out.println("File: " + filePath + " | Predicted: " + predictedClass);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Instance createTestInstance(double maxAmplitude, double rms, int zeroCrossings, String label, Instances data) {
        // Create a new instance with the same number of attributes as the training data
        Instance instance = new DenseInstance(4);  // 4 attributes: maxAmplitude, rms, zeroCrossings, and label

        // Set the attribute values for the instance
        instance.setValue(0, maxAmplitude);        // Set amplitude value
        instance.setValue(1, rms);                 // Set RMS value
        instance.setValue(2, zeroCrossings);       // Set zero-crossing rate value
        instance.setValue(3, label.equals("1") ? 1.0 : 0.0);  // Set class label (1 for music, 0 for speech)

        // Set the instance's dataset (this is needed for classification)
        instance.setDataset(data);

        return instance;
    }

    // Method to convert CSV to ARFF
    private void convertCsvToArff(String csvFile, String arffFile) {
    try (BufferedReader br = new BufferedReader(new FileReader(csvFile));
         BufferedWriter bw = new BufferedWriter(new FileWriter(arffFile))) {

        // Write ARFF header
        
        bw.write("@relation audio_features\n\n");

            bw.write("@attribute amplitude numeric\n");
            bw.write("@attribute rms numeric\n");
            bw.write("@attribute zero_crossing_rate numeric\n");
            bw.write("@attribute spectral_centroid numeric\n");
            bw.write("@attribute spectral_bandwidth numeric\n");
            bw.write("@attribute spectral_rolloff numeric\n");
            bw.write("@attribute label {speech, music}\n\n");

            bw.write("@data\n");

        // Read and convert each line from CSV to ARFF
        String line;
        while ((line = br.readLine()) != null) {
            if (!line.trim().isEmpty() && line.split(",").length == 7) {
                bw.write(line.trim() + "\n");
            }
        }

        //System.out.println("CSV converted to ARFF successfully!");

    } catch (IOException e) {
        e.printStackTrace();
        JOptionPane.showMessageDialog(this, "Error converting CSV to ARFF.");
    }
}



    // Method to clear CSV and ARFF files
    private void clearFiles() {
        File csvFile = new File("features.csv");
        File arffFile = new File("features.arff");

        if (csvFile.exists()) {
            csvFile.delete();  // Delete CSV file to clear data
        }

        if (arffFile.exists()) {
            arffFile.delete();  // Delete ARFF file to clear data
        }
    }

public static void main(String[] args) {
    SwingUtilities.invokeLater(new Runnable() {
        public void run() {
            try {
                // Initialize the GUI
                AudioPlayerGUI gui = new AudioPlayerGUI(); // Define and initialize `gui`
                gui.setVisible(true);

                // File paths
                String arffFilePath = "features.arff";

                // Clear old files and prepare the ARFF file
                System.out.println("Clearing old files...");
                gui.clearFiles();
                System.out.println("Generating features for training files...");
                gui.generateTrainingFeaturesCSV();
                System.out.println("Converting CSV to ARFF...");
                gui.convertCsvToArff("features.csv", arffFilePath);

                // Load the dataset
                System.out.println("Loading dataset...");
                Instances data = gui.loadDataset(arffFilePath); // Define and initialize `data`

                // Evaluate J48 classifier
                System.out.println("Evaluating J48 classifier with cross-validation...");
                gui.evaluateWithCrossValidation(data, new weka.classifiers.trees.J48(), "Decision Tree (J48)");

                // Evaluate Logistic Regression classifier
                System.out.println("Evaluating Logistic Regression classifier with cross-validation...");
                gui.evaluateWithCrossValidation(data, new weka.classifiers.functions.Logistic(), "Logistic Regression");

                System.out.println("Workflow completed successfully!");

            } catch (Exception e) {
                e.printStackTrace();
                System.err.println("An error occurred during the workflow.");
            }
        }
    });
}

}
