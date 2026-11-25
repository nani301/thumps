Upgraded files are `face-liveness.v2.js` and `index.v2.html`.

**Blink detection upgrades**
- Calibrates your open-eye baseline first, then uses hysteresis (close below 75%, reopen above 85%) so it doesn't flicker.
- Median smoothing over 3 frames removes landmark jitter.
- A blink counts only if all of these hold:
  - the eyes closed deeply (below 62% of baseline)
  - the closure lasted 40–500 ms
  - both eyes closed together
  - the mouth and nose stayed still. A waved photo moves the whole face, a real blink doesn't.
- Blink challenges only count when you face the camera.

**Anti-spoof integration**
- Runs your model on every 2nd frame during the whole session. It aborts instantly if the last 3 scores are all below 0.3.
- Final verdict uses the mean score over the session, plus a limit on how many frames scored below 0.5.
- It fails closed: if the model isn't loaded, nothing passes.
- Optional `seed` lets your server pick the challenge sequence.
- Also uses `tiny_face_detector` automatically if you add those model files (faster).

**Important:** I haven't seen your anti-spoof model, so I defaulted to the common MiniFASNet ONNX format. In `index.v2.html`, edit `loadAntiSpoof({...})` to match yours:
- `url` and `size` (input resolution)
- `realIndex` (which output index means "real")
- `bgr` and `norm` (channel order and pixel scaling)
- `softmax` (set false if the output is already probabilities)

If it's a TFJS model instead of ONNX, send me the file names and I'll swap the inference part. Wrong preprocessing is the most common reason an anti-spoof model rejects everyone, so test on a real face first.

To build an Aadhaar-level (ISO/IEC 30107-3 compliant) facial attendance system on top of your existing face-api.js models, you need to combine four distinct defense layers: [1] (https://www.iso.org/standard/53227.html), [2] (https://www.neurotechnology.com/face-verification-technical.html)
1. Hardware & Camera Integrity Guard (Blocks OBS, virtual cameras, and static image injection).
2. ISO/Aadhaar Biometric Quality Gates (Enforces frontal pose, inter-ocular distance, lighting, and sharpness before capturing a single enrollment profile).
3. Hybrid Active + Passive Liveness (Uses your face_landmark_68_model for randomized blink/head-turn challenges, paired with an optional lightweight ONNX passive spoof detector like MiniFASNetV2 to catch phone screen replays). [1] (https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx)
4. Zero-Raw-Image Vector Storage & Server-Side Matching (Stores only the encrypted 128D float descriptor from your face_recognition_model and verifies Euclidean distance on the backend).
Below is the complete architecture, mathematical logic, and copy-pasteable code context lines for your Next.js / React (v0) app.

1. System Architecture & Data Flow

Show Code
Download diagram SVG

2. Core Logic Module (lib/aadhaarFaceEngine.ts)

Create this file in your v0 project. It connects directly to the ssd_mobilenetv1, face_landmark_68, and face_recognition models you uploaded.
typescript
import * as faceapi from 'face-api.js';

export type ChallengeType = 'BLINK' | 'TURN_LEFT' | 'TURN_RIGHT';

export interface LivenessSession {
  challenges: ChallengeType[];
  currentStep: number;
  eyesClosedFrames: number;
  blinkDetected: boolean;
  previousFrameHash: string | null;
  identicalFrameCount: number;
  startTime: number;
}

export interface FrameEvaluation {
  status: 'SCANNING' | 'QUALITY_FAIL' | 'SPOOF_DETECTED' | 'VERIFIED';
  instruction: string;
  progress: number; // 0 to 100
  descriptor?: number[]; // 128-D float array from your face_recognition_model
}

// Helper: Euclidean distance between two 2D landmarks
const dist = (p1: faceapi.Point, p2: faceapi.Point): number =>
  Math.hypot(p1.x - p2.x, p1.y - p2.y);

// 1. Hardware Anti-Spoof: Block Virtual Cameras (OBS, ManyCam, etc.)
export async function verifyHardwareCamera(): Promise<{ valid: boolean; reason?: string }> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter((d) => d.kind === 'videoinput');
  
  const blockedKeywords = ['obs', 'virtual', 'manycam', 'snap camera', 'xsplit', 'droidcam'];
  for (const device of videoDevices) {
    const label = device.label.toLowerCase();
    if (blockedKeywords.some((kw) => label.includes(kw))) {
      return { valid: false, reason: `Virtual camera blocked: ${device.label}` };
    }
  }
  return { valid: true };
}

// 2. Initialize a Randomized Liveness Session (Prevents Video Replay Attacks)
export function createLivenessSession(mode: 'ENROLL' | 'ATTENDANCE'): LivenessSession {
  if (mode === 'ENROLL') {
    // Aadhaar Single Live Capture: Requires strict frontal alignment + 1 natural blink
    return {
      challenges: ['BLINK'],
      currentStep: 0,
      eyesClosedFrames: 0,
      blinkDetected: false,
      previousFrameHash: null,
      identicalFrameCount: 0,
      startTime: Date.now(),
    };
  }

  // Attendance Mode: Randomize order of Blink + Head Turn
  const turnAction: ChallengeType = Math.random() > 0.5 ? 'TURN_LEFT' : 'TURN_RIGHT';
  const challenges: ChallengeType[] =
    Math.random() > 0.5 ? ['BLINK', turnAction] : [turnAction, 'BLINK'];

  return {
    challenges,
    currentStep: 0,
    eyesClosedFrames: 0,
    blinkDetected: false,
    previousFrameHash: null,
    identicalFrameCount: 0,
    startTime: Date.now(),
  };
}

// 3. Eye Aspect Ratio (EAR) using your face_landmark_68_model (Indices 36-41 & 42-47)
function calculateEAR(eyePts: faceapi.Point[]): number {
  const verticalA = dist(eyePts[1], eyePts[5]);
  const verticalB = dist(eyePts[2], eyePts[4]);
  const horizontal = dist(eyePts[0], eyePts[3]);
  return (verticalA + verticalB) / (2.0 * horizontal);
}

// 4. Main Frame Evaluator (Call inside your video loop)
export function processAadhaarFrame(
  detections: faceapi.WithFaceDescriptor<
    faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>
  >[],
  videoWidth: number,
  session: LivenessSession,
  mode: 'ENROLL' | 'ATTENDANCE'
): FrameEvaluation {
  // Timeout check (15 seconds max per session to prevent replay prep)
  if (Date.now() - session.startTime > 15000) {
    return {
      status: 'SPOOF_DETECTED',
      instruction: 'Session timed out. Please restart verification.',
      progress: 0,
    };
  }

  // Rule 1: Strictly 1 face in frame
  if (detections.length === 0) {
    return { status: 'QUALITY_FAIL', instruction: 'Position your face in the oval', progress: 0 };
  }
  if (detections.length > 1) {
    return { status: 'QUALITY_FAIL', instruction: 'Multiple faces detected. Only one person allowed.', progress: 0 };
  }

  const det = detections[0];
  const pts = det.landmarks.positions;
  const box = det.detection.box;

  // Rule 2: Anti-Static Injection Check (Landmarks must have micro-tremors across frames)
  const noseTip = pts[30];
  const frameHash = `${noseTip.x.toFixed(2)}_${noseTip.y.toFixed(2)}`;
  if (session.previousFrameHash === frameHash) {
    session.identicalFrameCount++;
    if (session.identicalFrameCount > 12) {
      return { status: 'SPOOF_DETECTED', instruction: 'Static image injection detected.', progress: 0 };
    }
  } else {
    session.identicalFrameCount = 0;
    session.previousFrameHash = frameHash;
  }

  // Rule 3: Aadhaar Biometric Quality Gates (Distance & Inter-Ocular Width)
  const leftEyeCenter = pts[36];
  const rightEyeCenter = pts[45];
  const eyeDistancePx = dist(leftEyeCenter, rightEyeCenter);
  const faceWidthRatio = box.width / videoWidth;

  if (det.detection.score < 0.85) {
    return { status: 'QUALITY_FAIL', instruction: 'Improve lighting on your face', progress: 10 };
  }
  if (faceWidthRatio < 0.25 || eyeDistancePx < 55) {
    return { status: 'QUALITY_FAIL', instruction: 'Move closer to the camera', progress: 15 };
  }
  if (faceWidthRatio > 0.72) {
    return { status: 'QUALITY_FAIL', instruction: 'Move slightly back from the camera', progress: 15 };
  }

  // Rule 4: Head Pose Estimation (Yaw & Roll)
  const leftCheekDist = dist(pts[30], pts[0]);   // Nose tip (30) to Left Jaw (0)
  const rightCheekDist = dist(pts[30], pts[16]); // Nose tip (30) to Right Jaw (16)
  const yawRatio = leftCheekDist / (rightCheekDist || 1);

  const eyeDeltaY = Math.abs(pts[36].y - pts[45].y);
  const rollRatio = eyeDeltaY / (eyeDistancePx || 1);

  if (rollRatio > 0.15) {
    return { status: 'QUALITY_FAIL', instruction: 'Keep your head level (do not tilt)', progress: 20 };
  }

  // Rule 5: Active Liveness State Machine
  const activeChallenge = session.challenges[session.currentStep];
  const progress = Math.round(((session.currentStep + 0.5) / session.challenges.length) * 80);

  if (activeChallenge === 'BLINK') {
    // Ensure face is looking straight before checking blink
    if (yawRatio < 0.72 || yawRatio > 1.38) {
      return { status: 'QUALITY_FAIL', instruction: 'Look straight at the camera', progress };
    }

    const leftEAR = calculateEAR(pts.slice(36, 42));
    const rightEAR = calculateEAR(pts.slice(42, 48));
    const avgEAR = (leftEAR + rightEAR) / 2.0;

    if (avgEAR < 0.21) {
      session.eyesClosedFrames++;
    } else if (session.eyesClosedFrames >= 1 && avgEAR > 0.25) {
      // Natural blink completed
      session.eyesClosedFrames = 0;
      session.currentStep++;
    } else {
      return { status: 'SCANNING', instruction: 'Blink your eyes naturally', progress };
    }
  } else if (activeChallenge === 'TURN_LEFT') {
    // Note: Webcam is usually mirrored, so turning left shifts yawRatio > 1.55 or < 0.62
    if (yawRatio > 1.55 || yawRatio < 0.62) {
      session.currentStep++;
    } else {
      return { status: 'SCANNING', instruction: 'Turn your head slightly to the side', progress };
    }
  } else if (activeChallenge === 'TURN_RIGHT') {
    if (yawRatio < 0.62 || yawRatio > 1.55) {
      session.currentStep++;
    } else {
      return { status: 'SCANNING', instruction: 'Turn your head slightly to the other side', progress };
    }
  }

  // Rule 6: All Challenges Passed -> Enforce Final Frontal Pose for Descriptor Capture
  if (session.currentStep >= session.challenges.length) {
    if (yawRatio < 0.75 || yawRatio > 1.33) {
      return {
        status: 'SCANNING',
        instruction: 'Liveness passed! Now look straight ahead to capture...',
        progress: 90,
      };
    }

    // Extract the 128-D Float32 descriptor from your face_recognition_model
    return {
      status: 'VERIFIED',
      instruction: mode === 'ENROLL' ? 'Profile Created!' : 'Attendance Marked!',
      progress: 100,
      descriptor: Array.from(det.descriptor),
    };
  }

  return { status: 'SCANNING', instruction: 'Hold steady...', progress };
}
Use code with caution.

3. React / v0 Component Integration (components/AadhaarAttendanceScanner.tsx)

Here is how your webcam component wires the models from /public/models (where your 8 uploaded files live) into the liveness engine:
tsx
'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import {
  createLivenessSession,
  processAadhaarFrame,
  verifyHardwareCamera,
  LivenessSession,
} from '@/lib/aadhaarFaceEngine';

interface Props {
  mode: 'ENROLL' | 'ATTENDANCE';
  userId?: string;
  onSuccess: (descriptor: number[]) => Promise<void>;
}

export default function AadhaarAttendanceScanner({ mode, userId, onSuccess }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sessionRef = useRef<LivenessSession>(createLivenessSession(mode));
  const isLockedRef = useRef(false);

  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [instruction, setInstruction] = useState('Initializing biometric engine...');
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'IDLE' | 'SCANNING' | 'VERIFIED' | 'ERROR'>('IDLE');

  // 1. Load your exact 3 models from /public/models
  useEffect(() => {
    async function loadModels() {
      const MODEL_URL = '/models';
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);
      setModelsLoaded(true);
      startCamera();
    }
    loadModels();
  }, []);

  // 2. Start Camera with Anti-Virtual-Cam Check
  async function startCamera() {
    const camCheck = await verifyHardwareCamera();
    if (!camCheck.valid) {
      setStatus('ERROR');
      setInstruction(camCheck.reason || 'Invalid camera detected');
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      sessionRef.current = createLivenessSession(mode);
      setStatus('SCANNING');
    }
  }

  // 3. Real-Time Detection & Liveness Loop
  useEffect(() => {
    if (!modelsLoaded) return;

    const interval = setInterval(async () => {
      if (!videoRef.current || isLockedRef.current || videoRef.current.paused) return;

      // Uses your exact uploaded SSD MobileNet + 68 Landmarks + ResNet-34 Descriptor
      const detections = await faceapi
        .detectAllFaces(
          videoRef.current,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.75 })
        )
        .withFaceLandmarks()
        .withFaceDescriptors();

      const result = processAadhaarFrame(
        detections,
        videoRef.current.videoWidth || 640,
        sessionRef.current,
        mode
      );

      setInstruction(result.instruction);
      setProgress(result.progress);

      if (result.status === 'SPOOF_DETECTED') {
        setStatus('ERROR');
        isLockedRef.current = true;
      } else if (result.status === 'VERIFIED' && result.descriptor) {
        isLockedRef.current = true;
        setStatus('VERIFIED');
        await onSuccess(result.descriptor); // Send 128D vector to backend
      }
    }, 150); // Runs ~6.6 FPS for smooth CPU/mobile performance

    return () => clearInterval(interval);
  }, [modelsLoaded, mode, onSuccess]);

  return (
    <div className="flex flex-col items-center gap-4 p-6 border rounded-xl max-w-md mx-auto">
      <div className="relative w-72 h-72 rounded-full overflow-hidden border-4 border-blue-600">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover scale-x-[-1]"
        />
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-200"
          style={{ width: `${progress}%` }}
        />
      </div>

      <p className="text-center font-medium text-sm">{instruction}</p>

      {status === 'ERROR' && (
        <button
          onClick={() => {
            isLockedRef.current = false;
            sessionRef.current = createLivenessSession(mode);
            setStatus('SCANNING');
          }}
          className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm"
        >
          Retry Verification
        </button>
      )}
    </div>
  );
}
Use code with caution.

4. Backend Verification API (app/api/attendance/route.ts)

Never match faces solely on the frontend. Send the 128D descriptor array to your Next.js API route and calculate the Euclidean distance on the server:
typescript
import { NextResponse } from 'next/server';

// Server-side Euclidean distance (matches faceapi.euclideanDistance)
function euclideanDistance(desc1: number[], desc2: number[]): number {
  if (desc1.length !== desc2.length) return 1.0;
  let sum = 0;
  for (let i = 0; i < desc1.length; i++) {
    const diff = desc1[i] - desc2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export async function POST(req: Request) {
  const { descriptor, mode, userId, name } = await req.json();

  // Validate 128-D float array from your face_recognition_model
  if (!Array.isArray(descriptor) || descriptor.length !== 128) {
    return NextResponse.json({ error: 'Invalid biometric vector' }, { status: 400 });
  }

  if (mode === 'ENROLL') {
    // 1. Save `descriptor` (128 floats) into your DB (e.g., Supabase / PostgreSQL / MongoDB)
    // await db.users.create({ data: { userId, name, faceDescriptor: descriptor } });
    return NextResponse.json({ success: true, message: 'Single-capture profile created' });
  }

  if (mode === 'ATTENDANCE') {
    // 2. Fetch stored user descriptors from DB
    const users = await getAllEnrolledUsersFromDB(); // Replace with your DB query

    let bestMatch = { userId: '', name: '', distance: 1.0 };

    for (const user of users) {
      const distance = euclideanDistance(descriptor, user.faceDescriptor);
      if (distance < bestMatch.distance) {
        bestMatch = { userId: user.userId, name: user.name, distance };
      }
    }

    // Aadhaar-grade strict threshold for ResNet-34 128D vectors is 0.42 - 0.45
    if (bestMatch.distance < 0.45) {
      // await db.attendance.create({ data: { userId: bestMatch.userId, timestamp: new Date() } });
      return NextResponse.json({
        verified: true,
        user: bestMatch.name,
        confidence: Math.round((1 - bestMatch.distance) * 100),
      });
    }

    return NextResponse.json({ verified: false, error: 'Face did not match any profile' }, { status: 401 });
  }
}
Use code with caution.

5. Optional Upgrade: Passive Screen-Replay Detection (MiniFASNetV2)

If an attacker holds up a high-resolution video of a person blinking and turning their head on an iPad/phone screen, active challenges alone can sometimes be tricked. To block screen-replay attacks like Aadhaar FaceRD:
• Add MiniFASNetV2 ONNX (~1.8 MB) using onnxruntime-web.
• Crop the detected face bounding box (det.detection.box) to 80x80 pixels right when session.currentStep >= session.challenges.length, pass it to MiniFASNetV2, and verify that class[1] (live probability) is > 0.85 before calling onSuccess(descriptor). [1] (https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT), [2] (https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx)
Would you like me to:
• Show how to add the MiniFASNetV2 ONNX passive screen-replay detector alongside face-api.js, or
• Wire the Supabase / PostgreSQL database schema (pgvector) for fast 1:N attendance matching?


I have analyzed the 8 uploaded files. These are the pre-trained model weights and JSON manifests for face-api.js (TensorFlow.js), which confirms the exact machine learning architecture your v0 app is currently running:
• Files 1 & 8 (ssd_mobilenetv1_model): The SSD MobileNet V1 face detector (shard1 binary weights + JSON manifest). It locates the face bounding box in the camera feed ([1, 5118, 4] output).
• Files 2 & 4 (face_landmark_68_model): The 68-point Facial Landmark detector (shard1 binary weights + JSON manifest). It outputs a 136-value tensor (68 x 2 X/Y coordinates) mapping the eyes, eyebrows, nose, mouth, and jawline.
• Files 3, 5, 6 & 7 (face_recognition_model): The ResNet-34 Face Recognition model (shard1 & shard2 binary weights + JSON manifest). It outputs a 128-dimensional float32 facial descriptor ("shape":[256,128]) used to match a user's face.

Why This Is Great News

You do not need to replace your existing models or add heavy external libraries. Your face_landmark_68_model already tracks the exact 68 points needed to build Aadhaar-level active liveness, anti-spoofing, and single-capture quality gates directly on top of your existing face-api.js setup.

Upgrade Blueprint: Turning Your face-api.js Models into Aadhaar-Level Attendance


1. The 68-Landmark Map (Already in Your Models)

In your face_landmark_68_model (fc/weights shape [256, 136]), the 68 points are indexed 0 to 67:
• Jawline: 0–16 (Point 8 is the chin)
• Nose: 27–35 (Point 30 is the nose tip)
• Left Eye: 36–41
• Right Eye: 42–47
• Mouth: 48–67

2. Single Live Capture Profile Creation (Enrollment Quality Gate)

To match Aadhaar FaceRD standards, reject profile creation unless a single live frame passes all 4 checks simultaneously:
1. Single Face Only: detections.length === 1 (blocks background faces).
2. Confidence & Size Gate: Detection score > 0.85 and face box width occupies 30%–65% of the video frame (prevents tiny/distant faces).
3. Frontal Head Pose Check (Yaw & Pitch): Measure the horizontal distance from the nose tip (landmarks[30]) to the left cheek (landmarks[0]) and right cheek (landmarks[16]). The ratio leftDist / rightDist must be between 0.75 and 1.33 (ensures the user is looking straight ahead, not sideways).
4. Live Blink Confirmation: Require the user to blink once before locking in the 128D descriptor (detection.descriptor) and saving that 128-number array to your database.

3. Attendance Marking with Anti-Spoof Liveness

When an employee/student returns to mark attendance, run a 2-step randomized liveness challenge using your existing withFaceLandmarks().withFaceDescriptor() pipeline:
• Challenge A (Blink Detection via Eye Aspect Ratio): Uses landmarks 36–41 (left eye) and 42–47 (right eye) to detect a natural eye closure and reopening (EAR < 0.21 then > 0.26).
• Challenge B (Micro Head-Turn / Smile): Randomly asks the user to "Turn head slightly left/right" (tracking landmark 30 relative to 0 and 16) or "Smile" (tracking mouth width landmarks[48] to landmarks[54]). This defeats static photos and single-frame spoofs.
• Instant 1:N Descriptor Match: The exact millisecond the liveness state flips to true, compare the live 128D vector against stored vectors using Euclidean distance (faceapi.euclideanDistance(live, stored) < 0.45).

Drop-In Liveness & Quality Gate Helper (face-api.js)

You can paste this utility into your v0 project and call evaluateFaceFrame(detection, videoEl) inside your requestAnimationFrame / setInterval detection loop:
typescript
import * as faceapi from 'face-api.js';

export interface LivenessState {
  blinked: boolean;
  eyesWereClosed: boolean;
  headTurned: boolean;
}

const dist = (p1: faceapi.Point, p2: faceapi.Point) =>
  Math.hypot(p1.x - p2.x, p1.y - p2.y);

// Calculate Eye Aspect Ratio (EAR) from 6 eye landmarks
function getEAR(eye: faceapi.Point[]): number {
  const v1 = dist(eye[1], eye[5]);
  const v2 = dist(eye[2], eye[4]);
  const h = dist(eye[0], eye[3]);
  return (v1 + v2) / (2.0 * h);
}

export function evaluateFaceFrame(
  detection: faceapi.WithFaceDescriptor<
    faceapi.WithFaceLandmarks<{ detection: faceapi.FaceDetection }>
  >,
  videoWidth: number,
  state: LivenessState
) {
  const pts = detection.landmarks.positions;
  const box = detection.detection.box;

  // 1. Quality Gate: Face size & confidence
  const faceRatio = box.width / videoWidth;
  const isGoodDistance = faceRatio >= 0.25 && faceRatio <= 0.70;
  const isHighConfidence = detection.detection.score > 0.85;

  // 2. Pose Gate: Nose tip (30) vs Left jaw (0) and Right jaw (16)
  const leftCheekDist = dist(pts[30], pts[0]);
  const rightCheekDist = dist(pts[30], pts[16]);
  const yawRatio = leftCheekDist / (rightCheekDist || 1);
  const isFacingFront = yawRatio >= 0.70 && yawRatio <= 1.40;

  // 3. Liveness Check A: Blink Detection (Left eye: 36-41, Right eye: 42-47)
  const leftEAR = getEAR(pts.slice(36, 42));
  const rightEAR = getEAR(pts.slice(42, 48));
  const avgEAR = (leftEAR + rightEAR) / 2;

  if (avgEAR < 0.21) {
    state.eyesWereClosed = true;
  } else if (state.eyesWereClosed && avgEAR > 0.25) {
    state.blinked = true;
  }

  // 4. Liveness Check B: Slight Head Movement (anti-photo check)
  if (yawRatio < 0.65 || yawRatio > 1.50) {
    state.headTurned = true;
  }

  return {
    isReadyForEnrollment: isGoodDistance && isHighConfidence && isFacingFront && state.blinked,
    isLivenessPassed: isGoodDistance && isHighConfidence && state.blinked && state.headTurned,
    descriptor: Array.from(detection.descriptor), // 128-float array for DB storage
    feedback: !isGoodDistance
      ? 'Move closer or further from camera'
      : !state.blinked
      ? 'Please blink your eyes'
      : !state.headTurned
      ? 'Turn your head slightly left or right'
      : !isFacingFront
      ? 'Look straight at the camera'
      : 'Verified!',
  };
}
Use code with caution.

Critical Optimization Tip for Your Current Models

In your uploaded files, you are using ssd_mobilenetv1_model (~5.4 MB across two shards). While accurate, it can feel sluggish on mid-range Android phones during continuous liveness tracking. If you notice frame drops while checking for blinks, you can keep your exact same face_landmark_68_model and face_recognition_model, and simply swap the detector to faceapi.nets.tinyFaceDetector (~190 KB) for 3x faster FPS during liveness checks.
If you paste your v0 React component code (the .tsx file handling the webcam and faceapi calls), I can wire this exact enrollment + liveness + matching flow directly into your existing UI. Would you like to share that file?

To upgrade your v0 facial attendance prototype to Aadhaar-level security and accuracy, you need to add three core layers on top of basic face detection: active/passive liveness detection (anti-spoofing), 128D/512D facial embeddings (instead of raw image comparison), and strict quality gates during enrollment.
Since v0 typically builds React/Next.js frontends, you can implement this using modern web AI libraries paired with a secure backend.

1. How Aadhaar-Level Facial Systems Work

Aadhaar FaceRD (Face Registered Device) doesn't just match two photos; it enforces strict rules before accepting a frame:
• Single Live Capture Enrollment: Captures one high-quality, well-lit, front-facing frame, extracts a mathematical vector (embedding), and discards the raw photo (or stores it encrypted) for fast 1:N matching.
• Liveness Check (Anti-Spoofing): Ensures the camera is looking at a real 3D human face, not a photo on a phone screen, a printed picture, or a video replay.
• Quality Gates: Blocks the capture if the face is blurry, too dark, tilted, wearing dark glasses, or if multiple faces are in the frame.

2. Recommended Tech Stack (Next.js / v0 Compatible)


Feature	Recommended Open-Source / Free-Tier Tool	Why It Fits
Face Detection & Landmarks	MediaPipe Face Landmarker (Google)	Runs at 60fps in the browser; gives 478 3D facial landmarks to track blinks, head turns, and face angle.
Liveness Detection	Active Challenge + Passive Texture (Custom MediaPipe logic or Human.js)	Active challenges (e.g., "Blink once" or "Turn head slightly") stop 99% of photo/screen spoofs without paid APIs.
Face Recognition (Matching)	Face-api.js / InsightFace (ArcFace)	Generates a unique numerical array (embedding) for each person to compare against your database.
Vector Database	Supabase (pgvector) or PostgreSQL	Stores the facial embeddings and runs fast cosine-similarity searches to find who is marking attendance.


3. Step-by-Step Upgrade Architecture


Phase A: Single Live Capture (Profile Creation)

When a user registers for the first time, do not let them upload a gallery photo. Force a live camera capture with these checks:
1. Centering & Distance Check: Ensure the face bounding box covers 30%–60% of the camera frame.
2. Pose Check (Yaw/Pitch/Roll): Using MediaPipe landmarks, confirm the user is looking straight at the camera (head tilt < 10 degrees).
3. Liveness Challenge: Ask the user to blink or smile while the camera is open. Only capture the frame immediately after the action completes.
4. Embedding Generation: Pass that clean frame through your recognition model (e.g., ArcFace or face-api.js) to generate a 128- or 512-number array, and save it to your database under user_id.

Phase B: Attendance Marking with Liveness

When the user returns to mark attendance:
1. Trigger Randomized Liveness: Show a quick, randomized prompt on screen (e.g., "Blink your eyes" or "Turn head slightly left"). Randomizing prevents someone from holding up a pre-recorded video.
2. Verify Landmark Movement:
	• Blink Detection: Calculate the Eye Aspect Ratio (EAR) from MediaPipe eye landmarks. When EAR drops below 0.2 and returns above 0.25, a genuine blink occurred.
	• Head Turn Detection: Track the nose tip landmark relative to the left and right cheek landmarks.
3. Extract & Match Embedding: Immediately after liveness passes, extract the live face embedding and compare it to stored embeddings using Cosine Similarity.
4. Threshold Check: If similarity is > 0.65 (for ArcFace) or Euclidean distance < 0.45 (for face-api.js), mark attendance as Verified.

4. Sample Liveness Logic (Blink Detection for React/Next.js)

You can drop this logic into your v0 component alongside MediaPipe Face Landmarker to verify a real person is in front of the webcam before capturing the attendance frame:
typescript
// Eye Aspect Ratio (EAR) helper using MediaPipe 478 landmarks
function calculateEAR(eyeLandmarks: { x: number; y: number }[]) {
  const dist = (p1: any, p2: any) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

  const vertical1 = dist(eyeLandmarks[1], eyeLandmarks[5]);
  const vertical2 = dist(eyeLandmarks[2], eyeLandmarks[4]);
  const horizontal = dist(eyeLandmarks[0], eyeLandmarks[3]);

  return (vertical1 + vertical2) / (2.0 * horizontal);
}

// Inside your video frame loop:
// Left eye indices in MediaPipe: [33, 160, 158, 133, 153, 144]
// Right eye indices in MediaPipe: [362, 385, 387, 263, 373, 380]
const ear = (leftEAR + rightEAR) / 2;

if (ear < 0.20) {
  setEyesClosed(true);
} else if (eyesClosed && ear > 0.25) {
  setLivenessVerified(true); // Blink completed -> Trigger attendance capture!
}
Use code with caution.

5. Security Best Practices (Preventing Hacks)

• Never trust client-only matching: If your v0 app runs face matching purely in browser JavaScript, a tech-savvy user can bypass it via browser DevTools. Send the captured frame (or encrypted embedding) to a Next.js API route / Python FastAPI backend for final verification.
• Block Virtual Cameras: Check navigator.mediaDevices.enumerateDevices() and reject camera labels containing "OBS", "Virtual", or "ManyCam" to stop video-injection attacks.
• DPDP Act Compliance (India): Because you are building in India and handling biometric data, ensure you display a clear consent checkbox during profile creation explaining that facial vectors are stored solely for attendance, and encrypt the embedding columns in your database.
To give you the exact code for your v0 project, could you share:
• Which face detection library your v0 code is currently using (e.g., face-api.js, MediaPipe, or a cloud API)?
• Whether your app has a backend database connected yet (like Supabase, Firebase, or Python/Node)?
SponsoredIf you're exploring alternative ways to handle liveness checks for your attendance system, you can consider this option.
luxand.cloud
Liveness Detection API - Liveness API
Run liveness checks and improve the accuracy of facial recognition technology.
