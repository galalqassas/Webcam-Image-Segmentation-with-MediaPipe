import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

// Get DOM elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const demosSection = document.getElementById("demos");
let enableWebcamButton;
let webcamRunning = false;
let runningMode = "IMAGE";
let imageSegmenter;
let labels;

let blurAmount = 5; // Default blur level

// Get blur slider input
const blurControl = document.getElementById("blurControl");
blurControl.addEventListener("input", (event) => {
  blurAmount = event.target.value;  // Update blur amount
});

// Initialize MediaPipe Image Segmenter
const createImageSegmenter = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );

  imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
  labels = imageSegmenter.getLabels();
  demosSection.classList.remove("invisible");
};
createImageSegmenter();

async function enableCam() {
  if (!imageSegmenter) return;

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning
    ? "DISABLE SEGMENTATION"
    : "ENABLE WEBCAM";

  if (webcamRunning) {
    const constraints = { video: true };
    video.style.display = 'none'; // Hide the video element
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
    video.addEventListener("loadeddata", predictWebcam);
  } else {
    // Stop the webcam stream when disabling
    const stream = video.srcObject;
    const tracks = stream.getTracks();

    tracks.forEach(function (track) {
      track.stop();
    });

    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

let lastWebcamTime = -1;
async function predictWebcam() {
  if (video.currentTime === lastWebcamTime) {
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  lastWebcamTime = video.currentTime;

  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  if (!imageSegmenter) return;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await imageSegmenter.setOptions({ runningMode });
  }

  const startTimeMs = performance.now();
  imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}

function callbackForVideo(result) {
  let imageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const mask = result.categoryMask.getAsUint8Array();

  // Clear canvas and draw the original video frame
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  // Apply Gaussian blur filter to the entire canvas
  let blurredImage = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight);
  canvasCtx.filter = `blur(${blurAmount}px)`;
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  // Get the blurred image data
  let blurredData = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight);

  // Reset the filter for the foreground
  canvasCtx.filter = 'none';

  // Iterate through the mask, apply blur to background, and restore foreground
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 0) { // Background: keep blurred data
      imageData.data[i * 4] = blurredData.data[i * 4];     // Red
      imageData.data[i * 4 + 1] = blurredData.data[i * 4 + 1]; // Green
      imageData.data[i * 4 + 2] = blurredData.data[i * 4 + 2]; // Blue
      imageData.data[i * 4 + 3] = 255; // Ensure full opacity for background
    } else { // Foreground: ensure the foreground is fully opaque
      imageData.data[i * 4 + 3] = 255; // Set alpha to 255 (fully opaque)
    }
  }

  // Put the processed image back on the canvas
  canvasCtx.putImageData(imageData, 0, 0);

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
