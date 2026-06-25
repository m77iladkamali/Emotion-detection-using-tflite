import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class EmotionDetection extends StatefulWidget {
  const EmotionDetection({super.key});

  @override
  State<EmotionDetection> createState() => _EmotionDetectionState();
}

class _EmotionDetectionState extends State<EmotionDetection> {
  CameraController? controller;

  final FaceDetector faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      enableContours: false,
      enableLandmarks: false,
    ),
  );

  bool faceDetected = false;
  bool isBusy = false;

  @override
  void initState() {
    super.initState();
    startCamera();
  }

  Future<void> startCamera() async {
    final cameras = await availableCameras();

    controller = CameraController(
      cameras[1],
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await controller!.initialize();

    if (!mounted) return;

    setState(() {});

    controller!.startImageStream(processCameraImage);
  }

  Future<void> processCameraImage(CameraImage image) async {
    if (isBusy) return;

    isBusy = true;

    try {
      final WriteBuffer allBytes = WriteBuffer();

      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }

      final bytes = allBytes.done().buffer.asUint8List();

      final inputImage = InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(
            image.width.toDouble(),
            image.height.toDouble(),
          ),
          rotation: InputImageRotation.rotation90deg,
          format: InputImageFormat.nv21,
          bytesPerRow: image.planes.first.bytesPerRow,
        ),
      );

      final faces = await faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() {
          faceDetected = faces.isNotEmpty;
        });
      }
    } catch (e) {
      debugPrint(e.toString());
    }

    isBusy = false;
  }

  @override
  void dispose() {
    controller?.dispose();
    faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    bool ready =
        controller != null &&
        controller!.value.isInitialized;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Face Reader"),
      ),
      body: Column(
        children: [
          Expanded(
            child: ready
                ? CameraPreview(controller!)
                : const Center(
                    child: CircularProgressIndicator(),
                  ),
          ),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            color: faceDetected
                ? Colors.green
                : Colors.red,
            child: Text(
              faceDetected
                  ? "Face Detected ✅"
                  : "No Face ❌",
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 24,
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
