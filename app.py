import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("fire_detector.pt")

model = load_model()

st.title("Fire Detection App by YOLOv11")
st.write("Upload a video to detect fire using an AI-based detection system.")

# Sidebar settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.35, step=0.05
)
frame_skip = st.sidebar.slider(
    "Frame Skip (Process every nth frame)", min_value=1, max_value=10, value=1, step=1
)

# File uploader
uploaded_file = st.file_uploader("Upload Video (MP4)", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.video(temp_path)

    if st.button("Run Detection"):
        st.write("Running detection... This may take a moment.")

        try:
            # Open video file
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_directory = tempfile.mkdtemp()
            output_video_path = os.path.join(output_directory, "processed_video.mp4")

            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            video_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Process video
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames
                if frame_idx % frame_skip == 0:
                    results = model.predict(source=frame, conf=confidence_threshold, show=False)
                    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame  # Default to original frame if no detection

                    out.write(annotated_frame)
                    video_placeholder.image(
                        annotated_frame, channels="BGR", caption="Detection in Progress", use_container_width=True
                    )

                frame_idx += 1
                progress_bar.progress(min(frame_idx / frame_count, 1.0))

            cap.release()
            out.release()

            st.success("Detection completed!")

            # Display download button for processed video
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
