# CrowdManagementAI 
Crowd Management System is a fully autonomous, camera-based crowd intelligence platform designed for large-scale public events, government functions, stadium gatherings, religious events, and any occasion that draws significant crowds.
The system connects directly to IP surveillance cameras, processes every frame in real time using state-of-the-art computer vision AI, and automatically dispatches actionable alerts to the right security personnel — entirely without human intervention in the alert loop.
The platform addresses three critical crowd safety challenges: uncontrolled density buildup, bottleneck formation at corridors and gates, and dignitary arrival coordination — all handled autonomously within seconds of detection

# Real-Time Crowd Density Monitoring
Every camera feed is processed frame-by-frame using a YOLOv8 object detection model trained on the COCO dataset (class 0 = person). The model counts every visible person in the frame and maps the count to the zone the camera covers.

# VIP & Dignitary Face Recognition
The system maintains a database of registered VIPs — ministers, collectors, police superintendents, and other dignitaries. Each VIP is enrolled by registering their reference photograph, from which the AI extracts a 512-dimensional face embedding using InsightFace

# Bottleneck & Flow Anomaly Detection
Beyond static density, the system tracks how the crowd is moving using Lucas-Kanade optical flow — a computer vision algorithm that detects motion vectors between consecutive frames.

# Command Dashboard
The frontend dashboard provides a real-time view of the entire venue for the command centre. It connects to the Python backend via WebSocket and updates every second with no page reload required. All five tabs update simultaneously in the background.

# System Componenets
Camera manager
Crowd detector
VIP recognizer
Flow analyzer
Zone registry
AI pipeline
Dispatch engine
REST API
FastAPI server
Frontend
VIP enrollment
