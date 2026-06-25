import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_v2/tflite_v2.dart';

class EmotionDetection extends StatefulWidget {
  const EmotionDetection({super.key});

  @override
  State<EmotionDetection> createState() => _EmotionDetectionState();
}

class _EmotionDetectionState extends State<EmotionDetection> {
  CameraImage? cameraImage;
  CameraController? cameraController;

  String output = "Waiting...";
  String rawOutput = "No data";

  bool isProcessing = false;

  @override
  void initState() {
    super.initState();
    loadModel();
    loadCamera();
  }

  @override
  void dispose() {
    cameraController?.dispose();
    Tflite.close();
    super.dispose();
  }

  Future<void> loadCamera() async {
    final cameras = await availableCameras();

    cameraController = CameraController(
      cameras[1],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await cameraController!.initialize();

    if (!mounted) return;

    setState(() {});

    cameraController!.startImageStream((image) {
      cameraImage = image;
      runModel();
    });
  }

  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: "assets/model.tflite",
      labels: "assets/labels.txt",
      isAsset: true,
      numThreads: 1,
      useGpuDelegate: false,
    );
  }

  Future<void> runModel() async {
    if (cameraImage == null || isProcessing) return;

    isProcessing = true;

    try {
      var recognitions = await Tflite.runModelOnFrame(
        bytesList: cameraImage!.planes.map((plane) {
          return plane.bytes;
        }).toList(),
        imageHeight: cameraImage!.height,
        imageWidth: cameraImage!.width,
        imageMean: 127.5,
        imageStd: 127.5,
        rotation: 90,
        numResults: 7,
        threshold: 0.05,
        asynch: true,
      );

      if (recognitions != null && recognitions.isNotEmpty) {
        var best = recognitions.first;

        double confidence =
            ((best["confidence"] ?? 0.0) * 100);

        setState(() {
          output =
              "${best["label"]} (${confidence.toStringAsFixed(1)}%)";

          rawOutput = recognitions.toString();
        });
      }
    } catch (e) {
      setState(() {
        rawOutput = "Error: $e";
      });
    }

    isProcessing = false;
  }

  @override
  Widget build(BuildContext context) {
    bool cameraReady =
        cameraController != null &&
        cameraController!.value.isInitialized;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Emotion Detection"),
      ),
      body: Column(
        children: [
          Expanded(
            flex: 5,
            child: cameraReady
                ? CameraPreview(cameraController!)
                : const Center(
                    child: CircularProgressIndicator(),
                  ),
          ),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            child: Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  output,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),

          const Padding(
            padding: EdgeInsets.all(8.0),
            child: Text(
              "Raw Model Output",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ),

          Expanded(
            flex: 3,
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.all(8),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                border: Border.all(),
              ),
              child: SingleChildScrollView(
                child: Text(
                  rawOutput,
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
