import os
import time
import pickle
import threading
from collections import OrderedDict, deque
from datetime import datetime, timedelta
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import plotly.express as px
from LineChart import LineChart
from BarChart import BarChart
from PieChart import PieChart

# --------------------------
# Configuration
# --------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_SRC = 0

DETECTION_EVERY_N_FRAMES = 10
MIN_SECONDS_TO_SAVE = 10
SIMILARITY_THRESHOLD = 0.75
TRACKER_MAX_DISAPPEAR = 10
EMBEDDINGS_PATH = 'embeddings.pkl'
ATTENDANCE_CSV = 'attendance.csv'
VISUALS_LOG_PATH = 'visuals_log.csv'  
OUTPUT_WINDOW_NAME = "Face Attendance (OpenCV) - press 'q' or Stop in UI"

# Visual
BOX_COLOR = (0, 255, 0)
UNSAVED_COLOR = (0, 165, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------
# Utilities (persistence + attendance)
# --------------------------
def ensure_files_and_db():
    if not os.path.exists(EMBEDDINGS_PATH):
        db = {'ids': [], 'embeddings': [], 'names': [], 'meta': []}
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(db, f)
    else:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            db = pickle.load(f)
        changed = False
        if 'ids' not in db:
            db['ids'] = []
            changed = True
        if 'embeddings' not in db:
            db['embeddings'] = []
            changed = True
        if 'names' not in db:
            db['names'] = ['Unknown'] * len(db.get('ids', []))
            changed = True
        if 'meta' not in db:
            db['meta'] = []
            changed = True
        if changed:
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(db, f)

    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=['person_id', 'date', 'time']).to_csv(ATTENDANCE_CSV, index=False)

    # Create visuals log file
    if not os.path.exists(VISUALS_LOG_PATH):
        pd.DataFrame(columns=['identifier', 'timestamp', 'datetime']).to_csv(VISUALS_LOG_PATH, index=False)


def load_db():
    with open(EMBEDDINGS_PATH, 'rb') as f:
        db = pickle.load(f)
    if 'names' not in db:
        db['names'] = ['Unknown'] * len(db.get('ids', []))
    return db


def save_db(db):
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(db, f)


def log_attendance(person_id):
    df = pd.read_csv(ATTENDANCE_CSV)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    already = ((df['person_id'] == person_id) & (df['date'] == date_str)).any()
    if not already:
        new_row = {'person_id': person_id, 'date': date_str, 'time': time_str}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)


def log_visual_appearance(identifier):
    """
    Log appearance for visual tracking - only once per identifier per session
    identifier can be person_id (if saved) or track_id (if unsaved)
    """
    df = pd.read_csv(VISUALS_LOG_PATH)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") 
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check if this identifier already logged (no duplicates)
    already = (df['identifier'] == identifier).any()
    if not already:
        new_row = {'identifier': identifier, 'timestamp': timestamp, 'datetime': datetime_str}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(VISUALS_LOG_PATH, index=False)


def delete_person_everywhere(person_id):
    db = load_db()
    if person_id in db['ids']:
        idx = db['ids'].index(person_id)
        for k in ['ids', 'embeddings', 'names', 'meta']:
            if idx < len(db.get(k, [])):
                db[k].pop(idx)
        save_db(db)
    # Remove from attendance.csv as well
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
        df = df[df['person_id'] != person_id]
        df.to_csv(ATTENDANCE_CSV, index=False)


def compute_cosine_sim(a, b):
    return cosine_similarity(a, b)[0]

# --------------------------
# Track with OpenCV tracker for smooth motion
# --------------------------
class Track:
    def __init__(self, track_id, bbox, frame_index, embedding=None):
        self.id = track_id
        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_seen = frame_index
        self.continuous_visible_time = 0.0  # Time continuously visible (resets if person leaves)
        self.last_update_time = time.time()  # Last time we saw this track
        self.saved = False
        self.embedding = embedding
        self.missed_frames = 0
        self.note = ''
        self.tracker = None  # OpenCV tracker for smooth updates
        self.is_visible = True  # Currently visible in frame

    def compute_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox, frame_index, embedding=None, reinit_tracker=False):
        """Update when face is detected/tracked"""
        current_time = time.time()

        # Add time since last update to continuous time (only if was visible)
        if self.is_visible and hasattr(self, 'last_update_time'):
            self.continuous_visible_time += (current_time - self.last_update_time)

        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_seen = frame_index
        self.last_update_time = current_time
        self.is_visible = True

        if embedding is not None:
            self.embedding = embedding
        self.missed_frames = 0

        # Signal that tracker needs reinitialization
        if reinit_tracker:
            self.tracker = None

    def update_bbox_only(self, bbox):
        """Update bbox from tracker without changing other properties"""
        current_time = time.time()

        # Add time since last update to continuous time
        if self.is_visible and hasattr(self, 'last_update_time'):
            self.continuous_visible_time += (current_time - self.last_update_time)

        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_update_time = current_time
        self.is_visible = True

    def mark_missed(self):
        """Mark as not currently visible - RESET continuous time"""
        self.missed_frames += 1
        self.is_visible = False
        self.continuous_visible_time = 0.0  # Reset timer when person leaves

# --------------------------
# Core face attendance system with smooth tracking
# --------------------------
class FaceAttendanceSystem:
    def __init__(self, device=DEVICE):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tracks = OrderedDict()
        self.next_track_id = 0
        self.frame_index = 0
        self.prev_gray = None
        ensure_files_and_db()
        self.db = load_db()
        self.logged_visuals = set()  # Track what we've already logged in this session

    def detect_faces(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        crops, boxes_out = [], []
        if boxes is None:
            return [], []
        h, w = frame_bgr.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
            if x2 - x1 < 20 or y2 - y1 < 20: continue
            crop = rgb[y1:y2, x1:x2]
            boxes_out.append([x1, y1, x2, y2])
            crops.append(crop)
        return boxes_out, crops

    def get_embedding(self, crop_rgb):
        img = Image.fromarray(crop_rgb).resize((160,160))
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
        x = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.resnet(x)
        emb = emb.cpu().numpy().reshape(1, -1)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.reshape(-1)

    def match_to_db(self, embedding):
        if embedding is None:
            return None, 0.0
        if len(self.db['embeddings']) == 0:
            return None, 0.0
        saved = np.vstack(self.db['embeddings'])
        sims = compute_cosine_sim(embedding.reshape(1, -1), saved)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= SIMILARITY_THRESHOLD:
            return self.db['ids'][best_idx], best_sim
        return None, best_sim

    def add_to_db(self, embedding, name=""):
        pid = f"person_{len(self.db['ids']) + 1}"
        self.db['ids'].append(pid)
        self.db['embeddings'].append(embedding)
        self.db['names'].append(name)
        self.db['meta'].append({'added': datetime.now().isoformat()})
        save_db(self.db)
        log_attendance(pid)
        return pid

    def init_tracker_for_track(self, track, frame):
        """Initialize OpenCV CSRT tracker for smooth tracking"""
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        # Create CSRT tracker in a safe, version-compatible way
        try:
            track.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            # For some OpenCV builds use the legacy constructor
            track.tracker = cv2.TrackerCSRT_create()
        track.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def update_tracks(self, boxes, embeddings, matches=None):
        detections_centroids = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for b in boxes]
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0:
            for i, bbox in enumerate(boxes):
                emb = embeddings[i] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[i][0] is not None:
                    pid, sim = matches[i]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1
            return

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        if len(track_centroids) == 0:
            for i, bbox in enumerate(boxes):
                emb = embeddings[i] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[i][0] is not None:
                    pid, sim = matches[i]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1
            return

        D = np.zeros((len(track_centroids), len(detections_centroids)), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(detections_centroids):
                D[i, j] = np.linalg.norm(np.array(tc) - np.array(dc))

        while True:
            if D.size == 0:
                break
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] > 100:
                break
            track_id = track_ids[i]
            if track_id in assigned_tracks or j in assigned_dets:
                D[i, j] = 1e6
                continue
            emb = embeddings[j] if embeddings else None
            self.tracks[track_id].update(boxes[j], self.frame_index, embedding=emb, reinit_tracker=True)
            if matches and matches[j][0] is not None:
                pid, sim = matches[j]
                self.tracks[track_id].saved = True
                self.tracks[track_id].note = f"matched:{pid}"
                log_attendance(pid)
            assigned_tracks.add(track_id)
            assigned_dets.add(j)
            D[i, :] = 1e6
            D[:, j] = 1e6

        for j, bbox in enumerate(boxes):
            if j not in assigned_dets:
                emb = embeddings[j] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[j][0] is not None:
                    pid, sim = matches[j]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid].mark_missed()
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missed_frames > TRACKER_MAX_DISAPPEAR]
        for tid in to_delete:
            del self.tracks[tid]

    def step_on_frame(self, frame):
        events = []
        self.frame_index += 1

        # Only use tracker updates between detection frames (not on detection frames)
        is_detection_frame = (self.frame_index % DETECTION_EVERY_N_FRAMES) == 0

        if not is_detection_frame:
            # Update existing tracks with OpenCV trackers for smooth motion
            for tid, tr in list(self.tracks.items()):
                if tr.tracker is not None:
                    success, box = tr.tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in box]
                        # Validate box dimensions to prevent tracking random objects
                        if w > 20 and h > 20 and w < frame.shape[1] * 0.8 and h < frame.shape[0] * 0.8:
                            tr.update_bbox_only([x, y, x + w, y + h])
                        else:
                            # Invalid box size, mark as missed
                            tr.mark_missed()
                    else:
                        # Tracker lost the object
                        tr.mark_missed()

        # Detect faces periodically
        boxes, crops = [], []
        if is_detection_frame:
            try:
                boxes, crops = self.detect_faces(frame)
            except Exception:
                boxes, crops = [], []

            embeddings = []
            for crop in crops:
                try:
                    embeddings.append(self.get_embedding(crop))
                except Exception:
                    embeddings.append(None)

            if len(embeddings) != len(boxes):
                embeddings = [None] * len(boxes)

            # Update tracks with detections
            self.update_tracks(boxes, embeddings)

            # Reinitialize trackers for all tracks after detection update
            for tid, tr in self.tracks.items():
                try:
                    self.init_tracker_for_track(tr, frame)
                except Exception:
                    # If tracker init fails, skip; it will reattempt next detection frame
                    tr.tracker = None

        # Check each track for matching/saving
        for tid, tr in list(self.tracks.items()):
            # Only process visible tracks
            if not tr.is_visible:
                continue

            # compute embedding if missing by cropping from current frame
            if tr.embedding is None:
                x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                h, w = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    try:
                        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        tr.embedding = self.get_embedding(crop)
                    except Exception:
                        tr.embedding = None

            if tr.embedding is not None and not tr.saved:
                pid, sim = self.match_to_db(tr.embedding)
                if pid is not None:
                    tr.saved = True
                    tr.note = f"matched:{pid}"
                    log_attendance(pid)
                    events.append(('matched', pid, sim))
                else:
                    # Check continuous visible time (resets if person leaves)
                    if tr.continuous_visible_time >= MIN_SECONDS_TO_SAVE:
                        new_pid = self.add_to_db(tr.embedding)
                        tr.saved = True
                        tr.note = f"new_saved:{new_pid}"
                        events.append(('new_saved', new_pid))

        # Annotate frame - only show boxes for visible, saved tracks or tracks close to threshold
        annotated = frame.copy()
        for tid, tr in self.tracks.items():
            # Only show annotation if track is saved OR has been continuously visible for significant time
            # Don't show unsaved tracks that just appeared
            show_box = False

            if tr.saved:
                # Always show saved/matched tracks
                show_box = True
            elif tr.is_visible and tr.continuous_visible_time >= 3.0:
                # Show unsaved tracks only after 3 seconds of continuous visibility
                show_box = True

            # Log to visuals if visible for 3+ seconds (once per identifier)
            if show_box:
                # Determine identifier (person_id if saved, else track_id)
                if tr.note.startswith("matched:") or tr.note.startswith("new_saved:"):
                    identifier = tr.note.split(":")[1]  # person_id

                    # Log only saved persons (not unsaved tracks)
                    if identifier not in self.logged_visuals:
                        log_visual_appearance(identifier)
                        self.logged_visuals.add(identifier)

            if not show_box:
                continue

            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            color = BOX_COLOR if tr.saved else UNSAVED_COLOR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Track:{tid}"
            display_name, display_age, display_gender = "", "", ""
            if tr.note.startswith("matched:") or tr.note.startswith("new_saved:"):
                pid = tr.note.split(":")[1]
                if pid in self.db['ids']:
                    idx = self.db['ids'].index(pid)
                    display_name = self.db['names'][idx] if idx < len(self.db['names']) else ""
                    meta = self.db['meta'][idx] if idx < len(self.db['meta']) else {}
                    display_age = meta.get('age', '')
                    display_gender = meta.get('gender', '')

            if display_name:
                info_line1 = f"{display_name}"
                info_line2 = f"Age: {display_age}  |  Gender: {display_gender}" if display_age or display_gender else ""
            else:
                info_line1 = label
                info_line2 = f"{int(tr.continuous_visible_time)}s"  # Show continuous time

            cv2.putText(annotated, info_line1, (x1, max(y1 - 10, 0)), FONT, 0.6, TEXT_COLOR, 2)
            if info_line2:
                cv2.putText(annotated, info_line2, (x1, y2 + 20), FONT, 0.5, (255, 255, 255), 1)

        db_count = len(self.db['ids'])
        cv2.putText(annotated, f"Total Saved Persons: {db_count}", (10, 25), FONT, 0.7, (0, 255, 255), 2)

        return annotated, events


# --------------------------
# Camera thread
# --------------------------
def camera_thread_fn(engine: FaceAttendanceSystem, stop_event: threading.Event):
    cap = cv2.VideoCapture(VIDEO_SRC)
    try:
        cap.set(cv2.CAP_PROP_FPS, 60)
    except Exception:
        pass 

    if not cap.isOpened():
        print("ERROR: could not open camera")
        stop_event.set()
        return
    cv2.namedWindow(OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break
            annotated, events = engine.step_on_frame(frame)
            cv2.imshow(OUTPUT_WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        stop_event.set()


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Face Attendance ", layout="wide")
st.title("Face Attendance — MTCNN + FaceNet (OpenCV window)")

ensure_files_and_db()

if "engine" not in st.session_state:
    st.session_state.engine = None
if "cam_thread" not in st.session_state:
    st.session_state.cam_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "running" not in st.session_state:
    st.session_state.running = False

tabs = st.tabs(["Home", "Attendance", "Records", "Visuals"])

with tabs[0]:
    st.header("Home — Camera control (OpenCV window)")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Start (OpenCV window)"):
            # Prevent multiple starts
            already_running = (
                st.session_state.running
                and st.session_state.cam_thread is not None
                and getattr(st.session_state.cam_thread, "is_alive", lambda: False)()
            )
            if already_running:
                st.info("Camera already running.")
            else:
                st.session_state.engine = FaceAttendanceSystem(device=DEVICE)
                # ensure stop_event is fresh
                st.session_state.stop_event = threading.Event()
                th = threading.Thread(target=camera_thread_fn, args=(st.session_state.engine, st.session_state.stop_event), daemon=True)
                st.session_state.cam_thread = th
                st.session_state.running = True
                th.start()
                st.success("Camera started (OpenCV window). Close the window or press Stop to end.")
    with col2:
        if st.button("⏹️ Stop"):
            if not st.session_state.running:
                st.info("Camera is not running.")
            else:
                # signal stop and attempt a graceful join
                if st.session_state.stop_event is not None:
                    st.session_state.stop_event.set()
                th = st.session_state.cam_thread
                if th is not None:
                    th.join(timeout=3)
                st.session_state.running = False
                st.success("Stopping camera... (close OpenCV window if still open)")

    st.markdown("""
    **Notes**
    - The live video appears in a separate OpenCV window named: **Face Attendance (OpenCV)**.
    - Press **q** in that window to stop it, or use the **Stop** button here.
    - Streamlit UI will remain responsive while the OpenCV window is open.
    """)

with tabs[1]:
    st.header("Attendance (one entry per person per day)")
    if os.path.exists(ATTENDANCE_CSV):
        df_att = pd.read_csv(ATTENDANCE_CSV).sort_values(['date', 'time'], ascending=[False, False])
        st.dataframe(df_att)
        csv = df_att.to_csv(index=False).encode('utf-8')
        st.download_button("Download attendance CSV", csv, file_name="attendance.csv")
    else:
        st.write("No attendance yet.")

with tabs[2]:
    st.header("Records (Saved persons)")
    db = load_db()

    if len(db['ids']) == 0:
        st.info("No saved persons yet.")
    else:
        names = db.get('names', [])
        meta_list = db.get('meta', [])
        ages, genders = [], []
        for m in meta_list:
            ages.append(m.get('age', ''))
            genders.append(m.get('gender', ''))

        df_db = pd.DataFrame({
            'person_id': db['ids'],
            'name': names,
            'age': ages,
            'gender': genders,
            'added': [m.get('added', '') for m in meta_list]
        })

        st.dataframe(df_db)

        st.markdown("---")
        st.subheader("📝 Edit Person Details")

        edit_pid = st.selectbox("Select person to edit", options=db['ids'], key="edit_select")

        idx = db['ids'].index(edit_pid)
        curr_name = db['names'][idx] if idx < len(db['names']) else ""
        curr_meta = db['meta'][idx] if idx < len(db['meta']) else {}

        new_name = st.text_input("Name", value=curr_name)
        new_age = st.text_input("Age", value=str(curr_meta.get('age', '')))
        gender_options = ["", "Male", "Female", "Other"]
        gender_default_index = 0
        if curr_meta.get('gender', '') in gender_options:
            gender_default_index = gender_options.index(curr_meta.get('gender', ''))
        new_gender = st.selectbox("Gender", options=gender_options, index=gender_default_index, key="gender_edit")

        if st.button("💾 Apply Changes"):
            # --- Update in-memory database ---
            db['names'][idx] = new_name
            if idx < len(db['meta']):
                db['meta'][idx]['age'] = new_age
                db['meta'][idx]['gender'] = new_gender
            else:
                db['meta'].append({
                    'age': new_age,
                    'gender': new_gender,
                    'added': datetime.now().isoformat()
                })
            save_db(db)
            st.success(f"✅ Updated {edit_pid} — Name: {new_name}, Age: {new_age}, Gender: {new_gender}")

            # --- Sync with visuals_log.csv ---
            visuals_path = VISUALS_LOG_PATH
            if os.path.exists(visuals_path):
                try:
                    att_df = pd.read_csv(visuals_path)
                    # Create name column if not exists
                    if "name" not in att_df.columns:
                        att_df["name"] = ""
                    if "age" not in att_df.columns:
                        att_df["age"] = ""
                    if "gender" not in att_df.columns:
                        att_df["gender"] = ""
                    # Identify correct ID column
                    id_col = None
                    for col in att_df.columns:
                        if col.lower() in ["person_id", "id", "identifier"]:
                            id_col = col
                            break
                    if id_col:
                        # Update name wherever person_id matches
                        att_df.loc[att_df[id_col] == edit_pid, "name"] = new_name
                        att_df.to_csv(visuals_path, index=False)
                        st.info(f"🗂️ Synced name to {visuals_path} for {edit_pid}")
                    else:
                        st.warning("⚠️ Could not find person_id column in attendance.csv")
                    if id_col:
                        # Update age wherever person_id matches
                        att_df.loc[att_df[id_col] == edit_pid, "age"] = new_age
                        att_df.to_csv(visuals_path, index=False)
                        st.info(f"🗂️ Synced name to {visuals_path} for {edit_pid}")
                    else:
                        st.warning("⚠️ Could not find person_id column in attendance.csv")
                    if id_col:
                        # Update gender wherever person_id matches
                        att_df.loc[att_df[id_col] == edit_pid, "gender"] = new_gender
                        att_df.to_csv(visuals_path, index=False)
                        st.info(f"🗂️ Synced name to {visuals_path} for {edit_pid}")
                    else:
                        st.warning("⚠️ Could not find person_id column in attendance.csv")
                except Exception as e:
                    st.warning(f"⚠️ Error syncing attendance file: {e}")
            else:
                st.warning("⚠️ visiual_logs.csv not found, skipping sync")

        st.markdown("---")
        st.subheader("🗑️ Delete Person")
        del_pid = st.selectbox("Choose person to delete", options=db['ids'], key="del_select")
        if st.button("Delete selected person"):
            delete_person_everywhere(del_pid)
            st.success(f"Deleted {del_pid} (embeddings + attendance rows removed)")
            # Refresh local db variable so display is up-to-date for this run
            db = load_db()

with tabs[3]:
    st.header("📊 Visuals - Visitor Time Pattern")
    LineChart()
    BarChart()
    PieChart()
    
    # visuals_path = VISUALS_LOG_PATH

    # if os.path.exists(visuals_path):
    #     try:
    #         df = pd.read_csv(visuals_path)
    #         if "timestamp" in df.columns:
    #             df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    #         else:
    #             df["timestamp"] = pd.NaT
    #     except Exception:
    #         df = pd.DataFrame(columns=["identifier", "timestamp", "datetime"])

    #     # --- Filter options ---
    #     filter_option = st.selectbox(
    #         "Select time range:",
    #         ["All time", "Today", "Last 30 minutes", "Last 1 hour", "Last 3 hours"],
    #     )

    #     now = datetime.now()
    #     if not df.empty and df["timestamp"].notna().any():
    #         if filter_option == "Today":
    #             df = df[df["timestamp"].dt.date == now.date()]
    #         elif filter_option == "Last 30 minutes":
    #             df = df[df["timestamp"] >= now - pd.Timedelta(minutes=30)]
    #         elif filter_option == "Last 1 hour":
    #             df = df[df["timestamp"] >= now - pd.Timedelta(hours=1)]
    #         elif filter_option == "Last 3 hours":
    #             df = df[df["timestamp"] >= now - pd.Timedelta(hours=3)]

    #     if not df.empty and df["timestamp"].notna().any():
    #         # --- Filter between 9 AM and 11 PM ---
    #         df = df[(df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour <= 23)]

    #         if not df.empty:
    #             # --- Create hourly bins ---
    #             df["hour_bin"] = df["timestamp"].dt.floor("h")

    #             hour_range = pd.date_range(
    #                 df["hour_bin"].min().replace(hour=9, minute=0, second=0),
    #                 df["hour_bin"].max().replace(hour=23, minute=0, second=0),
    #                 freq="h",
    #             )

    #             grouped = (
    #                 df.groupby("hour_bin")
    #                 .agg({
    #                     "identifier": lambda x: ", ".join(map(str, x)),
    #                     "timestamp": lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()),
    #                     "datetime": "count",
    #                 })
    #                 .reset_index()
    #                 .rename(columns={"datetime": "Count"})
    #             )

    #             grouped = pd.merge(
    #                 pd.DataFrame({"hour_bin": hour_range}),
    #                 grouped,
    #                 on="hour_bin",
    #                 how="left"
    #             ).fillna({
    #                 "Count": 0,
    #                 "identifier": "",
    #                 "timestamp": ""
    #             })

    #             grouped["time_range"] = grouped["hour_bin"].dt.strftime("%H:%M")
                
    #             # --- Create interactive line chart (Plotly Express) ---
    #             fig = px.line(
    #                 grouped,
    #                 x="time_range",
    #                 y="Count",
    #                 markers=True,
    #                 title="🕒 Visitors per Hour (9 AM - 11 PM)",
    #                 hover_data={
    #                     "identifier": True,
    #                     "timestamp": True,   
    #                     "Count": True,
    #                     "time_range": False  
    #                 },
    #             )
    #             fig.update_traces(line=dict(width=3,color="#FC3407"),marker=dict(size=8,color="white"))
    #             st.plotly_chart(fig, use_container_width=True)
    #             # --- Chart Styling ---
    #             fig.update_layout(
    #                 xaxis_title="Time Range (Hour)",
    #                 yaxis_title="Number of Visitors",
    #                 plot_bgcolor="black",
    #                 hoverlabel=dict(bgcolor="white", font_color="black"),
    #                 xaxis=dict(showgrid=True, gridcolor="gray"),
    #                 yaxis=dict(showgrid=True, gridcolor="gray", dtick=1),
    #                 margin=dict(l=70, r=40, t=80, b=50),
    #             )
    #             # --- Hourly Gender Distribution Chart ---
    #             if "timestamp" in df.columns and "gender" in df.columns:
    #                 df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    #                 df = df[df["timestamp"].notna()]

    #                 # Create hour bins (hourly grouping)
    #                 df["hour_bin"] = df["timestamp"].dt.floor("h")
    #                 df["hour_label"] = df["hour_bin"].dt.strftime("%H:%M")

    #                 # Count visitors per gender per hour
    #                 hourly_gender_counts = (
    #                     df.groupby(["hour_label", "gender"])
    #                     .size()
    #                     .reset_index(name="Count")
    #                 )

    #                 # Prepare hover info — exact times of appearances
    #                 hover_data = (
    #                     df.groupby(["hour_label", "gender"])["timestamp"]
    #                     .apply(lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()))
    #                     .reset_index()
    #                     .rename(columns={"timestamp": "Exact_Times"})
    #                 )

    #                 # Merge hover info with counts
    #                 hourly_gender_counts = pd.merge(
    #                     hourly_gender_counts, hover_data, on=["hour_label", "gender"], how="left"
    #                 )

    #                 # Create grouped bar chart
    #                 fig_hourly_gender = px.bar(
    #                     hourly_gender_counts,
    #                     x="hour_label",
    #                     y="Count",
    #                     color="gender",
    #                     barmode="group", 
    #                     title="🕒 Hourly Gender Distribution",
    #                     hover_data=["Exact_Times"],
    #                     color_discrete_map={
    #                         "Male": "#71B4F8",
    #                         "Female": "#F897C8",
    #                         "Other": "#AAAAAA"
    #                     }
    #                 )

    #                 # Style chart
    #                 # fig_hourly_gender.update_traces(
    #                 #     text=hourly_gender_counts["Count"],
    #                 #     textposition="outside"
    #                 # )
    #                 fig_hourly_gender.update_layout(
    #                     xaxis_title="Hour of the Day",
    #                     yaxis_title="Number of Visitors",
    #                     plot_bgcolor="black",
    #                     hoverlabel=dict(bgcolor="white", font_color="black"),
    #                     margin=dict(l=60, r=40, t=60, b=50),
    #                 )

    #                 st.plotly_chart(fig_hourly_gender, use_container_width=True)
    #             else:
    #                 st.info("Timestamp or gender column not found to plot hourly gender distribution.")

    #             # --- Age Group Pie Chart ---
    #             if "age" in df.columns:
    #                 # Ensure numeric ages (in case some entries are missing or text)
    #                 df["age"] = pd.to_numeric(df["age"], errors="coerce")
    #                 df = df[df["age"].notna()]

    #                 # Define age bins and labels
    #                 bins = [10, 20, 30, 40, 50, 60, 100]
    #                 labels = ["10–20 y/o", "20–30 y/o", "30–40 y/o", "40–50 y/o", "50–60 y/o", "60+ y/o"]

    #                 # Categorize into age groups
    #                 df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    #                 # Count occurrences
    #                 age_group_counts = (
    #                     df["age_group"].value_counts()
    #                     .reset_index()
    #                     .rename(columns={"index": "Age Group", "age_group": "Count"})
    #                 )
    #                 age_group_counts.columns = ["Age Group", "Count"]

    #                 # Create pie chart
    #                 fig_age_pie = px.pie(
    #                     age_group_counts,
    #                     names="Age Group",  
    #                     values="Count",
    #                     title="🎂 Age Group Distribution of Appeared Persons",
    #                     color_discrete_sequence=px.colors.qualitative.Pastel
    #                 )

    #                 # Style chart
    #                 fig_age_pie.update_traces(
    #                     textinfo="percent+label",
    #                     hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
    #                     pull=[0.02] * len(age_group_counts)
    #                 )
    #                 fig_age_pie.update_layout(
    #                     showlegend=True,
    #                     legend_title="Age Group",
    #                     margin=dict(l=40, r=40, t=60, b=60)
    #                 )

    #                 st.plotly_chart(fig_age_pie, use_container_width=True)
    #             else:
    #                 st.info("Age column not found to plot age group distribution.")

    #         else:
    #             st.info("No visuals during the selected hours.")
    #     else:
    #         st.info("No visual logs for the selected period or timestamps could not be parsed.")
    # else:
    #     st.info("No visual logs available yet.")

    # --- Clear logs ---
    if st.button("🗑️ Clear Visual Logs"):
        try:
            with open(visuals_path, "w") as f:
                f.write("identifier,timestamp,datetime\n")
            st.success("Visual logs cleared successfully.")
        except Exception as e:
            st.error(f"Failed to clear visual logs: {e}")

st.markdown("---")
st.caption("Pipeline: MTCNN (detection) → FaceNet/InceptionResnetV1 (512-d embeddings) → centroid tracking → 1-minute save → daily attendance log.")

