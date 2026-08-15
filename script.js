// Gesture Meme Tracker - JavaScript Implementation
// Uses MediaPipe Hands and Face Mesh for gesture detection

// Gesture to meme mapping
const GESTURE_MEMES = {
    chill: "images/cat-reactions/chill.png", confident: "images/cat-reactions/confident.jpg", desperate: "images/cat-reactions/desperate.jpeg", happy: "images/cat-reactions/happy.gif", heart: "images/cat-reactions/heart.jpg", mad: "images/cat-reactions/mad.png", pointing_front: "images/cat-reactions/pointing-front.jpeg", pointing_up: "images/cat-reactions/pointing-up.jpg", punch: "images/cat-reactions/punch.png", sad: "images/cat-reactions/sad.jpeg", scared: "images/cat-reactions/scared.jpg", scuba: "images/cat-reactions/scuba.gif", thumbs_up: "images/cat-reactions/thumbs-up.jpg", tongue_out: "images/cat-reactions/tongue-out.png", none: "images/cat-reactions/confident.jpg"
};

// Global variables
let camera = null;
let hands = null;
let faceMesh = null;
let currentGesture = "none";
let lastGesture = "none";
let fpsInterval, startTime, now, then, elapsed;
let fpsCounter = 0;
let scubaHandY = null;
let scubaUntil = 0;

// DOM Elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const gestureLabelElement = document.getElementById('gestureLabel');
const memeVideoElement = document.getElementById('memeVideo');
const memeImageElement = document.getElementById('memeImage');
const loadingElement = document.getElementById('loading');
const startButton = document.getElementById('startButton');

function palmFacesCamera(hand) {
    const a = hand[5], b = hand[17], wrist = hand[0];
    const ux = a.x - wrist.x, uy = a.y - wrist.y, uz = a.z - wrist.z;
    const vx = b.x - wrist.x, vy = b.y - wrist.y, vz = b.z - wrist.z;
    const nz = ux * vy - uy * vx;
    const size = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, nz);
    return size && Math.abs(nz / size) > .7;
}

// Debug elements
const debugInfo = document.getElementById('debugInfo');
const fpsElement = document.getElementById('fps');
const handsCountElement = document.getElementById('handsCount');
const faceDetectedElement = document.getElementById('faceDetected');
const mouthHeightElement = document.getElementById('mouthHeight');
const mouthWidthElement = document.getElementById('mouthWidth');

// Enable debug mode by uncommenting this line
// debugInfo.style.display = 'block';

/**
 * Initialize MediaPipe Hands
 */
function initHands() {
    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);
}

/**
 * Initialize MediaPipe Face Mesh
 */
let storedFaceLandmarks = null;

function initFaceMesh() {
    faceMesh = new FaceMesh({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    // Set up face mesh results handler
    faceMesh.onResults((results) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            storedFaceLandmarks = results.multiFaceLandmarks[0];
            
            if (faceDetectedElement) {
                faceDetectedElement.textContent = "Yes";
            }
        } else {
            storedFaceLandmarks = null;
            if (faceDetectedElement) {
                faceDetectedElement.textContent = "No";
            }
        }
    });
}

/**
 * Check if a finger is extended
 */
function isFingerExtended(landmarks, tipIdx, pipIdx, mcpIdx) {
    // In normalized coordinates, y increases downward
    // Finger is extended if tip.y < pip.y < mcp.y
    return landmarks[tipIdx].y < landmarks[pipIdx].y && 
           landmarks[pipIdx].y < landmarks[mcpIdx].y;
}

/**
 * Detect gesture from hand landmarks
 */
function detectGesture(handLandmarks, allHands, faceLandmarks) {
    if (!handLandmarks) {
        // Check face-only reactions.
        if (faceLandmarks) {
            return detectFaceGesture(faceLandmarks);
        }
        return "none";
    }

    const landmarks = handLandmarks;

    // Count extended fingers
    const extendedFingers = [];

    // Index finger (8: tip, 6: PIP, 5: MCP)
    if (isFingerExtended(landmarks, 8, 6, 5)) {
        extendedFingers.push("index");
    }

    // Middle finger (12: tip, 10: PIP, 9: MCP)
    if (isFingerExtended(landmarks, 12, 10, 9)) {
        extendedFingers.push("middle");
    }

    // Ring finger (16: tip, 14: PIP, 13: MCP)
    if (isFingerExtended(landmarks, 16, 14, 13)) {
        extendedFingers.push("ring");
    }

    // Pinky finger (20: tip, 18: PIP, 17: MCP)
    if (isFingerExtended(landmarks, 20, 18, 17)) {
        extendedFingers.push("pinky");
    }

    const numExtended = extendedFingers.length;

    // Legacy mouth rule
    if (faceLandmarks) {
        const faceResult = detectFaceGesture(faceLandmarks);
        if (faceResult !== "none") {
            return faceResult;
        }
    }

    // Legacy two-fist rule
    if (allHands && allHands.length === 2) {
        let hand1Extended = 0;
        let hand2Extended = 0;

        // Count extended fingers for hand 1
        const fingerChecks = [[8, 6, 5], [12, 10, 9], [16, 14, 13], [20, 18, 17]];
        for (const [tip, pip, mcp] of fingerChecks) {
            if (allHands[0][tip].y < allHands[0][pip].y) {
                hand1Extended++;
            }
        }

        // Count extended fingers for hand 2
        for (const [tip, pip, mcp] of fingerChecks) {
            if (allHands[1][tip].y < allHands[1][pip].y) {
                hand2Extended++;
            }
        }

        // Both hands closed (fists)
        if (hand1Extended === 0 && hand2Extended === 0) {
            return "mad";
        }
    }

    // Legacy index rule
    if (numExtended === 1 && extendedFingers.includes("index")) {
        return "pointing_up";
    }

    // Legacy two-hand rule
    if (allHands && allHands.length === 2) {
        let hand1Extended = 0;
        let hand2Extended = 0;

        // Check if hands have extended fingers
        if (allHands[0][8].y < allHands[0][6].y) hand1Extended++;
        if (allHands[0][12].y < allHands[0][10].y) hand1Extended++;
        if (allHands[0][16].y < allHands[0][14].y) hand1Extended++;

        if (allHands[1][8].y < allHands[1][6].y) hand2Extended++;
        if (allHands[1][12].y < allHands[1][10].y) hand2Extended++;
        if (allHands[1][16].y < allHands[1][14].y) hand2Extended++;

        if (hand1Extended >= 2 && hand2Extended >= 2) {
            // Check if hands are spread apart
            const hand1Wrist = allHands[0][0];
            const hand2Wrist = allHands[1][0];
            const xDistance = Math.abs(hand1Wrist.x - hand2Wrist.x);

            if (xDistance > 0.3) {
                return "scared";
            }
        }
    }

    return "none";
}

/**
 * Detect face-only reactions
 */
function detectFaceGesture(faceLandmarks) {
    if (!faceLandmarks || !faceLandmarks.length) {
        return "none";
    }

    // Key face landmarks for mouth detection
    // 13: upper lip, 14: lower lip, 61: left mouth corner, 291: right mouth corner
    const upperLip = faceLandmarks[13];
    const lowerLip = faceLandmarks[14];
    const leftCorner = faceLandmarks[61];
    const rightCorner = faceLandmarks[291];

    if (!upperLip || !lowerLip || !leftCorner || !rightCorner) {
        return "none";
    }

    // Calculate mouth opening
    const mouthHeight = Math.abs(upperLip.y - lowerLip.y);
    const mouthWidth = Math.abs(rightCorner.x - leftCorner.x);

    // Update debug info
    if (mouthHeightElement) {
        mouthHeightElement.textContent = mouthHeight.toFixed(3);
        mouthWidthElement.textContent = mouthWidth.toFixed(3);
    }

    // Legacy mouth-opening rule
    if (mouthHeight > 0.01 && mouthWidth > 0.005) {
        return "tongue_out";
    }

    return "none";
}

/**
 * Handle results from MediaPipe
 */
function detectFaceGesture(face) {
    if (!face) return "none";
    const [upper, lower, left, right] = [face[13], face[14], face[61], face[291]];
    const height = Math.abs(upper.y - lower.y), width = Math.abs(left.x - right.x), center = (upper.y + lower.y) / 2;
    if (height > .01) return "tongue_out";
    const leftLift = center - left.y, rightLift = center - right.y;
    if (Math.abs(leftLift - rightLift) > .004) return "confident";
    if ((left.y + right.y) / 2 > upper.y + .004 && Math.abs(left.y - right.y) < .02) return "sad";
    if (left.y < center - .004 && right.y < center - .004) return "happy";
    return width > .10 ? "happy" : "none";
}

function detectGesture(hand, allHands, face) {
    if (!hand) return detectFaceGesture(face);
    const fingers = [[8, 6, 5], [12, 10, 9], [16, 14, 13], [20, 18, 17]].map(p => isFingerExtended(hand, ...p));
    const open = fingers.filter(Boolean).length, thumb = Math.abs(hand[4].x - hand[5].x) > .08;
    if (allHands?.length === 2) {
        const counts = allHands.map(h => [[8, 6], [12, 10], [16, 14], [20, 18]].filter(([a, b]) => h[a].y < h[b].y).length), wrists = allHands.map(h => h[0]);
        if (face) {
            const mouth = face[13], covered = allHands.map(h => Math.abs(h[9].x - mouth.x) < .14 && Math.abs(h[9].y - mouth.y) < .14);
            const raisedY = wrists.find((wrist, index) => !covered[index] && wrist.y < .45)?.y;
            const moving = raisedY !== undefined && scubaHandY !== null && Math.abs(raisedY - scubaHandY) > .012;
            scubaHandY = raisedY ?? scubaHandY;
            if (covered.includes(true) && moving) scubaUntil = performance.now() + 750;
            if (covered.includes(true) && raisedY !== undefined) return performance.now() < scubaUntil ? "scuba" : "none";
        }
        if (Math.abs(allHands[0][8].x - allHands[1][8].x) < .08 && Math.abs(allHands[0][4].x - allHands[1][4].x) < .10) return "heart";
        if (counts.every(n => n >= 3)) {
            const faceCenter = face ? (face[234].x + face[454].x) / 2 : .5;
            return wrists.every(w => Math.abs(w.x - faceCenter) < .28 && w.y < .7) ? "desperate" : "scared";
        }
        if (counts.every(n => n === 0)) return "mad";
    }
    if (thumb && fingers[3] && open === 1) return "chill";
    if (thumb && !open) return "thumbs_up";
    if (!open) return "punch";
    const otherFingersHidden = !fingers[1] && !fingers[2] && !fingers[3];
    if (otherFingersHidden && hand[8].z < hand[6].z - .02 && Math.abs(hand[8].x - hand[6].x) >= Math.abs(hand[8].y - hand[6].y) * .7) {
        return "pointing_front";
    }
    if (fingers[0] && open === 1) {
        return palmFacesCamera(hand) ? "pointing_up" : "none";
    }
    if (fingers[0] && fingers[1] && open === 2) return "none";
    return detectFaceGesture(face);
}

async function onResults(results) {
    // Update FPS
    fpsCounter++;
    
    // Clear canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    let detectedGesture = "none";

    // Process with face mesh to detect mouth
    if (results.image && faceMesh) {
        await faceMesh.send({ image: results.image });
    }

    // Use the stored face landmarks from the faceMesh callback
    const currentFaceLandmarks = window.faceLandmarks;

    // Draw hand landmarks
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: '#00FF00',
                lineWidth: 2
            });
            drawLandmarks(canvasCtx, landmarks, {
                color: '#FF0000',
                lineWidth: 1,
                radius: 3
            });
        }

        // Update debug info
        if (handsCountElement) {
            handsCountElement.textContent = results.multiHandLandmarks.length;
        }

        // Detect gesture
        detectedGesture = detectGesture(
            results.multiHandLandmarks[0],
            results.multiHandLandmarks,
            currentFaceLandmarks
        );
    } else {
        // No hands detected
        if (handsCountElement) {
            handsCountElement.textContent = "0";
        }
        
        // Check for face-only gestures
        detectedGesture = detectGesture(null, null, currentFaceLandmarks);
    }

    canvasCtx.restore();

    // Update gesture if changed
    if (detectedGesture !== currentGesture) {
        currentGesture = detectedGesture;
        updateGestureDisplay(currentGesture);
    }
}

// Make face landmarks available to gesture detection
Object.defineProperty(window, 'faceLandmarks', {
    get: function() {
        return storedFaceLandmarks;
    }
});

/**
 * Update gesture display and meme
 */
function updateGestureDisplay(gesture) {
    // Update label
    const gestureText = gesture.charAt(0).toUpperCase() + gesture.slice(1);
    gestureLabelElement.textContent = `Gesture: ${gestureText}`;

    if (gesture === "none") {
        memeVideoElement.style.display = 'none';
        memeImageElement.style.display = 'none';
        return;
    }

    // Update meme
    const memeFile = GESTURE_MEMES[gesture];
    
    if (memeFile.endsWith('.mp4') || memeFile.endsWith('.webm')) {
        // Video meme
        memeImageElement.style.display = 'none';
        memeVideoElement.style.display = 'block';
        
        if (memeVideoElement.src !== memeFile) {
            memeVideoElement.src = memeFile;
            memeVideoElement.load();
            memeVideoElement.play().catch(e => console.log('Video play failed:', e));
        }
    } else {
        // Image meme
        memeVideoElement.style.display = 'none';
        memeImageElement.style.display = 'block';
        memeImageElement.src = memeFile;
    }
}

updateGestureDisplay("none");

/**
 * Initialize camera
 */
async function initCamera() {
    try {
        camera = new Camera(webcamElement, {
            onFrame: async () => {
                await hands.send({ image: webcamElement });
            },
            width: 640,
            height: 480
        });

        await camera.start();
        
        // Hide loading, show video
        loadingElement.style.display = 'none';
        
        // Set canvas size to match video
        canvasElement.width = webcamElement.videoWidth || 640;
        canvasElement.height = webcamElement.videoHeight || 480;

        // Start FPS counter
        startFPSCounter();

        console.log('Camera started successfully!');
    } catch (error) {
        console.error('Error starting camera:', error);
        loadingElement.innerHTML = `
            <p>❌ Camera Error</p>
            <p style="font-size: 0.9rem;">Please allow camera access and refresh the page.</p>
            <button id="retryButton" onclick="location.reload()">Retry</button>
        `;
    }
}

/**
 * FPS Counter
 */
function startFPSCounter() {
    setInterval(() => {
        if (fpsElement) {
            fpsElement.textContent = fpsCounter;
        }
        fpsCounter = 0;
    }, 1000);
}

/**
 * Initialize everything
 */
async function init() {
    console.log('Initializing Gesture Meme Tracker...');
    
    // Initialize MediaPipe
    initHands();
    initFaceMesh();
    
    // Check if user needs to click to start (some browsers require user interaction)
    try {
        await initCamera();
    } catch (error) {
        // Show start button if autoplay is blocked
        startButton.style.display = 'block';
        startButton.onclick = initCamera;
    }
}

// Modify detectGesture to use stored face landmarks
function detectGestureWithFace(handLandmarks, allHands) {
    return detectGesture(handLandmarks, allHands, window.faceLandmarks);
}

// Start when page loads
window.addEventListener('load', init);
