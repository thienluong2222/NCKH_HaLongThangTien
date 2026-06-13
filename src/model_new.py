import pandas as pd
import os
from pathlib import Path
import cv2
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from ultralytics import YOLO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL
import math
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# PHẦN 1: YOLO PIPELINE 
# ==========================================

class YOLOCSVPipeline:
    def __init__(self, model_path, csv_path):
        """
        Pipeline để detect object bằng YOLO và map với CSV

        Args:
            model_path: Đường dẫn đến model YOLO
            csv_path: Đường dẫn đến file CSV mapping
        """
        self.model = YOLO(model_path)
        self.mapping_df = pd.read_csv(csv_path)

        # Chuẩn hóa tên cột
        self.mapping_df.columns = self.mapping_df.columns.str.strip()

        print(f"✅ Đã load model: {model_path}")
        print(f"✅ Đã load CSV: {csv_path}")
        print(f"📊 Số dòng trong CSV: {len(self.mapping_df)}")
        print(f"📋 Các cột: {list(self.mapping_df.columns)}")

    def predict_and_map(self, image_path, confidence_threshold=0.5, show_image=True):
        """
        Detect object và map với CSV
        """
        # 1. YOLO Predict
        results = self.model.predict(image_path, verbose=False)

        detected_items = []  # Lưu cả class_name VÀ confidence
        matched_results = []

        for result in results:
            if show_image:
                result.show()
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf)

                    # Chỉ lấy detection có confidence >= threshold
                    if confidence >= confidence_threshold:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]

                        # ✅ Lưu CẢ class_name VÀ confidence tương ứng
                        detected_items.append({
                            'class_name': class_name,
                            'confidence': confidence
                        })

        # 2. Map với CSV
        for item in detected_items:
            detected_class = item['class_name']
            conf = item['confidence']  # ✅ Lấy confidence tương ứng

            # Tìm trong CSV (case-insensitive)
            matches = self.mapping_df[
                self.mapping_df['SubClass'].str.lower() == detected_class.lower()
            ]

            if not matches.empty:
                for _, row in matches.iterrows():
                    matched_results.append({
                        'detected_subclass': detected_class,
                        'mapped_subclass': row['SubClass'],
                        'class': row['Class'],
                        'text': row['Text'],
                        'confidence': conf  # ✅ Dùng confidence đúng
                    })
            else:
                # Không tìm thấy mapping
                matched_results.append({
                    'detected_subclass': detected_class,
                    'mapped_subclass': None,
                    'class': None,
                    'text': None,
                    'confidence': conf  # ✅ Dùng confidence đúng
                })

        return matched_results

    def process_single_image(self, image_path, show_unmapped=False):
        """
        Xử lý 1 ảnh và hiển thị kết quả
        """
        print(f"\n{'='*60}")
        print(f"🖼️  Đang xử lý: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        results = self.predict_and_map(image_path)

        if not results:
            print("❌ Không phát hiện object nào!")
            return None

        print(f"\n✅ Phát hiện {len(results)} object(s):\n")

        for i, result in enumerate(results, 1):
            if result['mapped_subclass'] is not None:
                print(f"{i}. 🎯 Detected: {result['detected_subclass']}")
                print(f"   ├─ SubClass: {result['mapped_subclass']}")
                print(f"   ├─ Class: {result['class']}")
                print(f"   ├─ Text: {result['text']}")
                print(f"   └─ Confidence: {result['confidence']:.2%}\n")
            elif show_unmapped:
                print(f"{i}. ⚠️  Detected: {result['detected_subclass']}")
                print(f"   └─ Không tìm thấy mapping trong CSV\n")

        return results

    def process_folder(self, image_folder, output_csv='results.csv',
                    confidence_threshold=0.5):
        """
        Xử lý tất cả ảnh trong thư mục
        """
        print(f"\n{'='*60}")
        print(f"📁 Xử lý thư mục: {image_folder}")
        print(f"{'='*60}\n")

        # Lấy tất cả ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))

        if not image_files:
            print("❌ Không tìm thấy ảnh nào!")
            return None

        print(f"Tìm thấy {len(image_files)} ảnh\n")

        all_results = []

        for img_path in image_files:
            results = self.predict_and_map(str(img_path), confidence_threshold)

            for result in results:
                result['image'] = os.path.basename(str(img_path))
                all_results.append(result)

            print(f"✓ {os.path.basename(str(img_path))}: {len(results)} detections")

        # Tạo DataFrame
        df = pd.DataFrame(all_results)

        # Lọc chỉ lấy kết quả có mapping
        df_mapped = df[df['mapped_subclass'].notna()].copy()

        # Sắp xếp
        df_mapped = df_mapped.sort_values(['image', 'confidence'],
                                        ascending=[True, False])

        # Lưu file
        df_mapped.to_csv(output_csv, index=False, encoding='utf-8-sig')

        print(f"\n{'='*60}")
        print(f"✅ Hoàn thành!")
        print(f"📊 Tổng detections: {len(all_results)}")
        print(f"✓ Có mapping: {len(df_mapped)}")
        print(f"✗ Không mapping: {len(all_results) - len(df_mapped)}")
        print(f"💾 Đã lưu: {output_csv}")
        print(f"{'='*60}\n")

        # Thống kê
        if not df_mapped.empty:
            print("📈 Top 10 Class phổ biến:")
            print(df_mapped['class'].value_counts().head(10))
            print("\n📈 Top 10 SubClass phổ biến:")
            print(df_mapped['mapped_subclass'].value_counts().head(10))

        return df_mapped

    def get_info_by_subclass(self, subclass_name):
        """
        Tra cứu thông tin từ SubClass
        """
        matches = self.mapping_df[
            self.mapping_df['SubClass'].str.lower() == subclass_name.lower()
        ]

        if matches.empty:
            return None

        return matches[['Text', 'Class', 'SubClass']].to_dict('records')

    def draw_detections(self, frame, detections_data):
        """
        Vẽ bounding box và thông tin lên frame

        Args:
            frame: Frame cần vẽ
            detections_data: Danh sách detection với boxes và thông tin

        Returns:
            frame đã được vẽ
        """
        annotated_frame = frame.copy()

        for det in detections_data:
            # Lấy thông tin bounding box
            box = det['box']  # [x1, y1, x2, y2]
            confidence = det['confidence']
            label = det['label']

            # Chuyển đổi tọa độ
            x1, y1, x2, y2 = map(int, box)

            # Màu sắc (xanh lá cho mapped, vàng cho unmapped)
            color = (0, 255, 0) if det.get('mapped', False) else (0, 255, 255)

            # Vẽ bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Chuẩn bị text
            text = f"{label} {confidence:.2f}"

            # Tính kích thước text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Vẽ background cho text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Vẽ text
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

            # Vẽ thêm thông tin Class nếu có mapping
            if det.get('mapped', False) and det.get('class_name'):
                class_text = f"{det['class_name']}"
                cv2.putText(
                    annotated_frame,
                    class_text,
                    (x1, y2 + 20),
                    font,
                    0.5,
                    color,
                    1
                )

        return annotated_frame

    def predict_and_map_with_boxes(self, frame, confidence_threshold=0.5):
        """
        Detect object, map với CSV và trả về cả thông tin boxes

        Args:
            frame: Frame hoặc đường dẫn ảnh
            confidence_threshold: Ngưỡng confidence

        Returns:
            List các detection với boxes và mapping info
        """
        # YOLO Predict
        results = self.model.predict(frame, verbose=False)

        detections_data = []

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf)

                    if confidence >= confidence_threshold:
                        class_id = int(box.cls)
                        class_name = result.names[class_id]

                        # Lấy tọa độ bounding box
                        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                        # Map với CSV
                        matches = self.mapping_df[
                            self.mapping_df['SubClass'].str.lower() == class_name.lower()
                        ]

                        if not matches.empty:
                            row = matches.iloc[0]
                            detection = {
                                'box': xyxy,
                                'confidence': confidence,
                                'label': row['SubClass'],
                                'class_name': row['Class'],
                                'text': row['Text'],
                                'detected_subclass': class_name,
                                'mapped': True
                            }
                        else:
                            detection = {
                                'box': xyxy,
                                'confidence': confidence,
                                'label': class_name,
                                'class_name': None,
                                'text': None,
                                'detected_subclass': class_name,
                                'mapped': False
                            }

                        detections_data.append(detection)

        return detections_data

    def process_video_with_output(self, video_path, output_path=None,
                                top_k=5, top_n_classes=3, confidence_threshold=0.5,
                                fps_detect=1, max_duration=15,
                                output_fps=None, save_frames=False,
                                output_folder='video_frames'):
        """
        Xử lý video, lưu video kết quả với bounding box và tổng hợp thống kê

        Args:
            video_path: Đường dẫn video input
            output_path: Đường dẫn video output (None = tự động tạo)
            top_k: Số lượng top objects (SubClass)
            top_n_classes: Số lượng top Classes có nhiều object nhất
            confidence_threshold: Ngưỡng confidence
            fps_detect: Số frame/giây để chạy detection (1 = detect mỗi giây)
            max_duration: Thời lượng tối đa xử lý (giây)
            output_fps: FPS của video output (None = giữ nguyên FPS gốc)
            save_frames: Có lưu frames đã extract không
            output_folder: Thư mục lưu frames

        Returns:
            dict: Kết quả tổng hợp
        """
        print(f"\n{'='*70}")
        print(f"🎥 BẮT ĐẦU XỬ LÝ VIDEO VỚI LƯU KẾT QUẢ")
        print(f"{'='*70}")
        print(f"📹 Video: {os.path.basename(video_path)}")

        # ========== MỞ VIDEO ==========
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Không thể mở video!")
            return None

        # Lấy thông tin video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps

        print(f"\n📊 Thông tin video:")
        print(f"├─ Kích thước: {width}x{height}")
        print(f"├─ FPS gốc: {video_fps:.2f}")
        print(f"├─ Tổng frames: {total_frames}")
        print(f"└─ Thời lượng: {duration:.2f} giây")


        if output_fps is None:
            output_fps = video_fps

        # Tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if output_path is not None:
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        print(f"\n📹 Video output:")
        print(f"├─ Đường dẫn: {output_path}")
        print(f"├─ FPS: {output_fps:.2f}")
        print(f"└─ Detect rate: {fps_detect} frame/giây")

        # Tính toán
        process_duration = min(duration, max_duration)
        frame_interval = int(video_fps / fps_detect)
        total_process_frames = int(process_duration * video_fps)

        print(f"\n⚙️ Cấu hình xử lý:")
        print(f"├─ Xử lý {process_duration:.2f}s / {duration:.2f}s")
        print(f"├─ Tổng frames sẽ xử lý: {total_process_frames}")
        print(f"└─ Detection mỗi {frame_interval} frames")

        # Tạo thư mục lưu frames nếu cần
        if save_frames:
            os.makedirs(output_folder, exist_ok=True)

        # ========== XỬ LÝ VIDEO ==========
        print(f"\n{'─'*60}")
        print("🔄 BẮT ĐẦU XỬ LÝ")
        print(f"{'─'*60}\n")

        all_detections = []
        frame_results = []
        current_detections = []  # Lưu detection hiện tại để áp dụng cho frames giữa

        frame_count = 0
        detect_count = 0

        while frame_count < total_process_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Kiểm tra xem có cần chạy detection không
            should_detect = (frame_count % frame_interval == 0)

            if should_detect:
                # Chạy detection
                current_detections = self.predict_and_map_with_boxes(
                    frame, confidence_threshold
                )
                detect_count += 1

                # Lưu kết quả
                time_stamp = frame_count / video_fps
                frame_results.append({
                    'frame': frame_count,
                    'time': time_stamp,
                    'detections': current_detections,
                    'count': len(current_detections)
                })

                # Thêm vào all_detections
                all_detections.extend(current_detections)

                # In progress
                if detect_count % 5 == 0 or detect_count == 1:
                    print(f"⏳ Đã detect {detect_count} frames - "
                        f"Tìm thấy {len(current_detections)} objects tại {time_stamp:.1f}s")

                # Lưu frame nếu cần
                if save_frames:
                    frame_filename = f"frame_{detect_count:04d}_at_{time_stamp:.2f}s.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)
                    cv2.imwrite(frame_path, frame)

            # Vẽ bounding box (sử dụng detection gần nhất)
            annotated_frame = self.draw_detections(frame, current_detections)

            # Thêm thông tin timestamp
            timestamp_text = f"Time: {frame_count/video_fps:.2f}s | Objects: {len(current_detections)}"
            cv2.putText(
                annotated_frame,
                timestamp_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Ghi frame vào video output
            if output_path is not None:
                out.write(annotated_frame)

            frame_count += 1

        # Đóng video
        cap.release()
        if output_path is not None:
            out.release()

        print(f"\n✅ Đã xử lý {frame_count} frames")
        print(f"✅ Đã chạy detection trên {detect_count} frames")
        print(f"💾 Video đã lưu: {output_path}")

        # ========== TỔNG HỢP THỐNG KÊ ==========
        print(f"\n{'─'*60}")
        print("📊 TỔNG HỢP KẾT QUẢ")
        print(f"{'─'*60}")

        if all_detections:
            # Đếm các detection (chỉ những cái đã map)
            mapped_detections = [d for d in all_detections if d['mapped']]

            if mapped_detections:
                subclass_counter = Counter()
                class_counter = Counter()
                class_object_counter = Counter()  # Đếm tổng số object cho mỗi Class
                confidence_dict = {}
                class_confidence_dict = {}  # Lưu confidence cho mỗi Class

                for det in mapped_detections:
                    subclass = det['label']
                    main_class = det['class_name']

                    subclass_counter[subclass] += 1
                    class_counter[main_class] += 1
                    class_object_counter[main_class] += 1  # Đếm mỗi object thuộc Class

                    # Lưu confidence cho SubClass
                    if subclass not in confidence_dict:
                        confidence_dict[subclass] = []
                    confidence_dict[subclass].append(det['confidence'])

                    # Lưu confidence cho Class
                    if main_class not in class_confidence_dict:
                        class_confidence_dict[main_class] = []
                    class_confidence_dict[main_class].append(det['confidence'])

                # Tính confidence trung bình cho SubClass
                avg_confidence = {k: np.mean(v) for k, v in confidence_dict.items()}

                # Tính confidence trung bình cho Class
                class_avg_confidence = {k: np.mean(v) for k, v in class_confidence_dict.items()}

                print(f"\n📈 Thống kê tổng quan:")
                print(f"├─ Tổng detections: {len(all_detections)}")
                print(f"├─ Đã mapping: {len(mapped_detections)}")
                print(f"├─ SubClass unique: {len(subclass_counter)}")
                print(f"└─ Class unique: {len(class_counter)}")

                # ========== TOP K OBJECTS (SUBCLASS) ==========
                print(f"\n{'─'*60}")
                print(f"🏆 TOP {top_k} OBJECTS (SUBCLASS) XUẤT HIỆN NHIỀU NHẤT")
                print(f"{'─'*60}")

                top_results = []
                for i, (subclass, count) in enumerate(subclass_counter.most_common(top_k), 1):
                    info = self.mapping_df[
                        self.mapping_df['SubClass'] == subclass
                    ].iloc[0]

                    frequency = (count / detect_count) * 100

                    result_item = {
                        'rank': i,
                        'subclass': subclass,
                        'class': info['Class'],
                        'text': info['Text'],
                        'appearances': count,
                        'total_detections': detect_count,
                        'frequency': frequency,
                        'avg_confidence': avg_confidence[subclass]
                    }

                    top_results.append(result_item)

                    print(f"\n🥇 TOP {i}: {subclass}")
                    print(f"   ├─ Class: {info['Class']}")
                    print(f"   ├─ Xuất hiện: {count}/{detect_count} frames ({frequency:.1f}%)")
                    print(f"   └─ Confidence TB: {avg_confidence[subclass]:.2%}")

                # ========== TOP N CLASSES CÓ NHIỀU OBJECT NHẤT ==========
                print(f"\n{'─'*60}")
                print(f"🏆 TOP {top_n_classes} CLASSES CÓ NHIỀU OBJECT NHẤT")
                print(f"{'─'*60}")

                top_classes = []
                for i, (main_class, total_objects) in enumerate(class_object_counter.most_common(top_n_classes), 1):
                    # Tìm các SubClass thuộc Class này
                    subclasses_in_class = []
                    for subclass, count in subclass_counter.items():
                        info = self.mapping_df[
                            self.mapping_df['SubClass'] == subclass
                        ].iloc[0]
                        if info['Class'] == main_class:
                            subclasses_in_class.append({
                                'subclass': subclass,
                                'count': count
                            })

                    # Sắp xếp SubClass theo số lượng
                    subclasses_in_class.sort(key=lambda x: x['count'], reverse=True)

                    class_frequency = (total_objects / len(mapped_detections)) * 100

                    class_item = {
                        'rank': i,
                        'class': main_class,
                        'total_objects': total_objects,
                        'unique_subclasses': len(subclasses_in_class),
                        'subclasses': subclasses_in_class,
                        'frequency': class_frequency,
                        'avg_confidence': class_avg_confidence[main_class]
                    }

                    top_classes.append(class_item)

                    print(f"\n🏅 TOP {i}: {main_class}")
                    print(f"   ├─ Tổng số objects: {total_objects}")
                    print(f"   ├─ Tỷ lệ: {class_frequency:.1f}% trong tổng số detections")
                    print(f"   ├─ Số SubClass khác nhau: {len(subclasses_in_class)}")
                    print(f"   ├─ Confidence TB: {class_avg_confidence[main_class]:.2%}")
                    print(f"   └─ Chi tiết SubClass:")

                    # Hiển thị top 3 SubClass của Class này
                    for j, sub_item in enumerate(subclasses_in_class[:3], 1):
                        print(f"       {j}. {sub_item['subclass']}: {sub_item['count']} objects")

                    if len(subclasses_in_class) > 3:
                        print(f"       ... và {len(subclasses_in_class) - 3} SubClass khác")

                # ========== THỐNG KÊ THEO THỜI GIAN ==========
                print(f"\n{'─'*60}")
                print("📈 PHÂN BỐ THEO THỜI GIAN")
                print(f"{'─'*60}")

                # Chia video thành các khoảng thời gian
                time_segments = 5  # Chia thành 5 phần
                segment_duration = process_duration / time_segments

                time_distribution = []
                for seg in range(time_segments):
                    start_time = seg * segment_duration
                    end_time = (seg + 1) * segment_duration

                    # Đếm detections trong khoảng thời gian này
                    segment_detections = 0
                    for frame_result in frame_results:
                        if start_time <= frame_result['time'] < end_time:
                            segment_detections += frame_result['count']

                    time_distribution.append({
                        'segment': seg + 1,
                        'start': start_time,
                        'end': end_time,
                        'detections': segment_detections
                    })

                    print(f"   Phút {start_time:.1f}-{end_time:.1f}s: {segment_detections} objects")

                # Tạo summary đầy đủ
                summary = {
                    'video_info': {
                        'input_path': video_path,
                        'output_path': output_path,
                        'duration': duration,
                        'processed_duration': process_duration,
                        'total_frames': frame_count,
                        'detected_frames': detect_count,
                        'resolution': f"{width}x{height}",
                        'fps': video_fps,
                        'output_fps': output_fps
                    },
                    'detection_summary': {
                        'total_detections': len(all_detections),
                        'mapped_detections': len(mapped_detections),
                        'unique_subclasses': len(subclass_counter),
                        'unique_classes': len(class_counter)
                    },
                    'top_objects': top_results,  # Top k objects (SubClass)
                    'top_classes': top_classes,  # Top n Classes có nhiều object nhất
                    'time_distribution': time_distribution,
                    'frame_details': frame_results,
                    'yolo_detections': all_detections

                }

                print(f"\n{'='*70}")
                print("✅ HOÀN THÀNH!")
                print(f"{'='*70}")

                return summary

        print("\n❌ Không phát hiện object nào!")
        return None
    
    def process_video(self, video_path, confidence_threshold=0.5, fps_detect=1):
        """Xử lý video và trả về list ObjectDetection (Dùng cho Bayesian Classifier)"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps_detect)
        
        all_objects = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_interval == 0:
                raw_dets = self.predict_and_map_with_boxes(frame, confidence_threshold)
                
                # Group by subclass trong frame hiện tại
                subclass_groups = {}
                for d in raw_dets:
                    # Chỉ lấy detection có mapping hợp lệ
                    if d.get('mapped'):
                        lbl = d['label']
                        if lbl not in subclass_groups:
                            subclass_groups[lbl] = {'confs': [], 'boxes': []}
                        subclass_groups[lbl]['confs'].append(d['confidence'])
                        subclass_groups[lbl]['boxes'].append(d['box'])

                time_stamp = frame_count / video_fps
                
                for sub, data in subclass_groups.items():
                    obj = ObjectDetection(
                        subclass=sub,
                        confidence=np.mean(data['confs']),
                        frame_id=frame_count,
                        time_stamp=time_stamp,
                        count=len(data['boxes']),
                        bboxs=data['boxes']
                    )
                    all_objects.append(obj)
            
            frame_count += 1
        
        cap.release()
        return all_objects


# Định nghĩa cấu trúc dữ liệu
class ObjectDetection:
    def __init__(self, subclass, confidence, frame_id, time_stamp, count, bboxs):
        self.subclass = subclass          # e.g., "binh_bong_dua"
        self.confidence = confidence      # trung bình các confidence trong frame
        self.frame_id = frame_id          # số thứ tự frame
        self.time_stamp = time_stamp      # thời gian (giây)
        self.count = count        # số lần subclass xuất hiện trong frame
        self.bboxs = bboxs                # danh sách bounding box (list of [x1, y1, x2, y2])
    def __repr__(self):
        return (f"ObjectDetection(subclass='{self.subclass}', "
                f"confidence={self.confidence:.2f}, "
                f"frame_id={self.frame_id}, "
                f"time_stamp={self.time_stamp:.2f}, "
                f"count={self.count}, "
                f"bboxs={self.bboxs})")

# Database ràng buộc: dict[lễ_hội] = list[ràng_buộc]
# Mỗi ràng_buộc là tuple (type, params, is_hard, weight, threshold)
# Ví dụ: ("is_presence", ["binh_bong_dua", "hoa_sen"], True, 1.0, None)  # Hard, phải có cả hai
# ("at_least", ["binh_bong_dua"], True, 1.0, 10)  # Hard, ít nhất 10 instances
# ("is_on", ["bong_dua", "trai_dua"], False, 0.5, None)  # Soft, weight 0.5 nếu "bong_dua" on "trai_dua" (có thể dùng spatial check)
# ("confidence_min", ["all"], True, 1.0, 0.7)  # Hard, avg confidence >=0.7

# ==========================================
# CẤU HÌNH TOÀN CỤC (Từ PSEUDO)
# ==========================================
GLOBAL_CONFIG = {
    "T_high": 0.85,    # Ngưỡng tin cậy cao để chọn ứng viên ngay
    "T_low": 0.50,     # Ngưỡng thấp nhất để xem xét
    "delta": 0.25,     # Chênh lệch tối đa cho phép so với conf_max
    "T_out": 0.85      # Ngưỡng quyết định cuối cùng (sau khi hỏi user)
}

UNCERTAINTY_RULES = {
    "chắc có": 0.85,
    "có": 1.0,
    "hình như có": 0.6,
    "có lẽ có": 0.55,
    "chắc không": 0.35,
    "không": 0.0,
    "hình như không": 0.45,
    "có lẽ không": 0.4
}

# ==========================================
# CÁC HÀM TOÁN HỌC BỔ TRỢ
# ==========================================
def clip_confidence(p):
    """Giới hạn p trong khoảng (epsilon, 1-epsilon) để tránh log(0)"""
    eps = 1e-6
    if p < eps: p = eps
    if p > 1 - eps: p = 1 - eps
    return p

def logit(p):
    """Chuyển đổi xác suất p sang không gian Logit (Log-odds)"""
    p = clip_confidence(p)
    return math.log(p / (1 - p))

def sigmoid(x):
    """Chuyển đổi ngược từ Logit sang xác suất [0, 1]"""
    return 1 / (1 + math.exp(-x))

# ==========================================
# CLASS DATA STRUCTURE
# ==========================================
class ObjectDetection:
    def __init__(self, subclass, confidence, frame_id, time_stamp, count, bboxs):
        self.subclass = subclass          # e.g., "binh_bong_dua"
        self.confidence = confidence      # trung bình các confidence trong frame
        self.frame_id = frame_id          # số thứ tự frame
        self.time_stamp = time_stamp      # thời gian (giây)
        self.count = count                # số lần subclass xuất hiện trong frame
        self.bboxs = bboxs                # danh sách bounding box [x1, y1, x2, y2]
    
    def __repr__(self):
        return f"<Obj: {self.subclass}, Conf: {self.confidence:.2f}, Count: {self.count}>"


# ==========================================
# PHẦN 2: BAYESIAN REASONING CORE (LOGIT + ADDITIVE ONLY)
# ==========================================
class BayesianFestivalClassifier:
    def __init__(self, api_key):
        self.api_key = api_key
        # Sử dụng model mạnh hơn một chút để parse JSON tốt hơn nếu cần, hoặc flash vẫn ok
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, max_retries=3, temperature=0)

    def _index_detections(self, detections):
        by_subclass = defaultdict(list)
        by_frame = defaultdict(list)
        for d in detections:
            by_subclass[d.subclass].append(d)
            by_frame[d.frame_id].append(d)
        return by_subclass, by_frame

    def _check_is_on(self, top_sub, bot_sub, by_subclass, by_frame):
        relevant_frames = set(d.frame_id for d in by_subclass[top_sub]) & set(d.frame_id for d in by_subclass[bot_sub])
        for fid in relevant_frames:
            tops = [d for d in by_frame[fid] if d.subclass == top_sub]
            bots = [d for d in by_frame[fid] if d.subclass == bot_sub]
            for t in tops:
                for b in bots:
                    for box_t in t.bboxs:
                        for box_b in b.bboxs:
                            x_overlap = max(0, min(box_t[2], box_b[2]) - max(box_t[0], box_b[0]))
                            width_t = box_t[2] - box_t[0]
                            vertical_gap = box_b[1] - box_t[3]
                            if width_t > 0 and (x_overlap/width_t) > 0.3 and -50 <= vertical_gap <= 50:
                                return True
        return False

    def check_constraints(self, rule, by_subclass, by_frame):
        ctype, params, is_hard, weight, threshold = rule
        satisfied = False
        if ctype == "is_presence":
            satisfied = len([p for p in params if p not in by_subclass]) == 0
        elif ctype == "is_presence_in_frame":
            for fid, dets in by_frame.items():
                subs = {d.subclass for d in dets}
                if all(p in subs for p in params):
                    satisfied = True; break
        elif ctype == "at_least":
            total = sum(sum(d.count for d in by_subclass[p]) for p in params if p in by_subclass)
            satisfied = total >= (threshold or 1)
        elif ctype == "at_least_in_frame":
            for fid, dets in by_frame.items():
                cnt = sum(d.count for d in dets if d.subclass in params)
                if cnt >= (threshold or 1):
                    satisfied = True; break
        elif ctype == "confidence_min":
            target = list(by_subclass.keys()) if "all" in params else [p for p in params if p in by_subclass]
            if target:
                avg = sum(d.confidence * d.count for s in target for d in by_subclass[s]) / sum(d.count for s in target for d in by_subclass[s])
                satisfied = avg >= (threshold or 0)
        elif ctype == "is_on" and len(params) == 2:
            satisfied = self._check_is_on(params[0], params[1], by_subclass, by_frame)
        return satisfied

    def calculate_initial_logits(self, detections):
        by_subclass, by_frame = self._index_detections(detections)
        festival_logits = {}
        festival_unsatisfied = defaultdict(list)

        for festival, rules in CONSTRAINTS_DB.items():
            current_logit = 0.0
            for rule in rules:
                is_satisfied = self.check_constraints(rule, by_subclass, by_frame)
                weight = rule[3]
                if is_satisfied:
                    current_logit += weight
                else:
                    festival_unsatisfied[festival].append(rule)
            festival_logits[festival] = current_logit
        return festival_logits, festival_unsatisfied

    def select_candidates(self, festival_logits):
        festival_probs = {f: sigmoid(l) for f, l in festival_logits.items()}
        if not festival_probs: return []
        max_prob = max(festival_probs.values())
        candidates = []
        
        print(f"\nBẢNG XẾP HẠNG BAN ĐẦU:")
        for f, p in sorted(festival_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {f}: {p:.2%} (Logit: {festival_logits[f]:.2f})")

        for f, p in festival_probs.items():
            if p >= GLOBAL_CONFIG["T_high"]: candidates.append(f)
            elif p >= GLOBAL_CONFIG["T_low"] and (max_prob - p) <= GLOBAL_CONFIG["delta"]: candidates.append(f)
        return candidates

    # ==========================================
    # PHẦN 3: LLM INTERACTION - CONSOLIDATED QUESTION
    # ==========================================
    
    def generate_consolidated_question(self, candidates, festival_unsatisfied):
        """
        Tạo 1 câu hỏi duy nhất tổng hợp tất cả các đặc trưng còn thiếu.
        """
        all_missing_features = set()
        for fest in candidates:
            rules = festival_unsatisfied[fest]
            for rule in rules:
                # Rule[1] là params (list các subclass cần tìm)
                all_missing_features.update(rule[1])
        
        if not all_missing_features:
            return None
        
        feature_list_str = ", ".join(all_missing_features)
        candidate_str = ", ".join(candidates)
        
        question = (
            f"Hệ thống đang phân vân giữa các lễ hội: {candidate_str}. "
            f"Bạn hãy quan sát kỹ video và cho biết bạn có thấy các đặc trưng sau không: "
            f"{feature_list_str}?"
        )
        
        # Trả về cả text câu hỏi và list features để dùng cho bước analyze sau này
        return {
            "question_text": question,
            "target_features": list(all_missing_features)
        }

    def analyze_complex_answer(self, question, user_answer, target_features):
        """
        Phân tích câu trả lời phức tạp bằng LLM và map với UNCERTAINTY_RULES.
        Trả về JSON mapping: {feature: {"status": True/False, "confidence": float}}
        """
        # Chuyển rules thành string để đưa vào prompt
        rules_desc = json.dumps(UNCERTAINTY_RULES, ensure_ascii=False)
        features_desc = ", ".join(target_features)
        
        prompt = f"""
        Nhiệm vụ: Phân tích câu trả lời của người dùng về sự xuất hiện của các vật thể trong video.
        
        Danh sách vật thể cần tìm (Features): {features_desc}
        
        Bảng điểm tin cậy (Uncertainty Rules):
        {rules_desc}
        
        Câu hỏi của hệ thống: "{question}"
        Câu trả lời của người dùng: "{user_answer}"
        
        Yêu cầu Output:
        Trả về một JSON object duy nhất. Key là tên vật thể (trong danh sách Features), Value là object chứa:
        - "status": true (nếu người dùng bảo có), false (nếu người dùng bảo không).
        - "confidence": Điểm số lấy chính xác từ Bảng điểm tin cậy dựa trên từ ngữ người dùng dùng.
        
        Ví dụ: Nếu user nói "Có đèn gió nhưng chắc không có ghe ngo", output:
        {{
            "đèn gió": {{"status": true, "confidence": 1.0}},
            "ghe ngo": {{"status": false, "confidence": 0.35}}
        }}
        
        Chỉ trả về JSON, không thêm markdown.
        """
        
        parser = JsonOutputParser()
        try:
            result = self.llm.invoke(prompt).content
            # Clean markdown if exists
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            parsed_result = json.loads(result.strip())
            return parsed_result
        except Exception as e:
            print(f"Lỗi parse JSON từ LLM: {e}")
            return {}

    def update_logits_from_consolidated_answer(self, festival_logits, candidates, festival_unsatisfied, parsed_answer):
        """
        Cập nhật điểm Logit dựa trên kết quả phân tích JSON.
        (Có thưởng có phạt).
        """
        final_logits = festival_logits.copy()
        
        print("\nCập nhật điểm dựa trên câu trả lời...")
        
        for fest in candidates:
            unsatisfied_rules = festival_unsatisfied[fest]
            
            for rule in unsatisfied_rules:
                params = rule[1]
                weight = rule[3]
                
                # Kiểm tra xem feature trong rule này có được user nhắc tới không
                # Một rule có thể yêu cầu nhiều params (VD: ["A", "B"]). 
                # Đơn giản hóa: Nếu bất kỳ param nào trong rule được nhắc tới
                
                for param in params:
                    if param in parsed_answer:
                        data = parsed_answer[param]
                        status = data.get("status")
                        conf = data.get("confidence", 0.5)
                        
                        if status is True:
                            # User xác nhận CÓ -> Cộng điểm
                            # Delta = Weight * Confidence
                            delta = weight * conf
                            final_logits[fest] += delta
                            print(f"   => [{fest}] '{param}' CÓ (conf={conf}): +{delta:.2f}")
                            
                        elif status is False:
                            # User xác nhận KHÔNG -> Trừ điểm (Phương án A)
                            # Penalty = (Weight * Confidence) / 2
                            penalty = (weight * conf) / 2
                            final_logits[fest] -= penalty
                            print(f"   => [{fest}] '{param}' KHÔNG (conf={conf}): -{penalty:.2f}")
                            
        return final_logits


    def decide_final_result(self, final_logits):
        """Kết luận cuối cùng"""
        final_probs = {f: sigmoid(l) for f, l in final_logits.items()}
        results = []
        
        print(f"\n KẾT QUẢ CUỐI CÙNG:")
        sorted_res = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        for f, p in sorted_res:
            status = "ĐẠT" if p >= GLOBAL_CONFIG["T_out"] else "TRƯỢT"
            print(f"   {f}: {p:.2%} ({status})")
            if p >= GLOBAL_CONFIG["T_out"]:
                results.append(f)
                
        return results, final_probs

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Cấu hình đường dẫn
    MODEL_PATH = '../weight/best_openvino_model/' 
    CSV_PATH = '../artifacts/merged_data.csv'
    VIDEO_PATH = '../assets/input/2.mp4'
    
    if not API_KEY:
        print("❌ Lỗi: Chưa cấu hình GEMINI_API_KEY trong file .env hoặc code.")
        exit()

    # 1. Khởi tạo Pipeline
    yolo_pipe = YOLOCSVPipeline(MODEL_PATH, CSV_PATH)
    classifier = BayesianFestivalClassifier(API_KEY)

    print("\n--- BƯỚC 1: YOLO DETECTION ---")
    detections = yolo_pipe.process_video(VIDEO_PATH, fps_detect=1)
    print(f"Đã phát hiện {len(detections)} đối tượng/sự kiện.")

    print("\n--- BƯỚC 2: TÍNH LOGIT & CHECK RÀNG BUỘC TỰ ĐỘNG ---")
    logits, unsatisfied = classifier.calculate_initial_logits(detections)

    print("\n--- BƯỚC 3: LỌC ỨNG VIÊN ---")
    candidates = classifier.select_candidates(logits)
    
    if not candidates:
        print("❌ Không có lễ hội nào tiềm năng dựa trên hình ảnh.")
    else:
        # --- BƯỚC 4: HỎI ĐÁP TỔNG HỢP ---
        print("\n--- BƯỚC 4: HỎI ĐÁP TỔNG HỢP (CONSOLIDATED) ---")
        
        # 4a. Sinh câu hỏi duy nhất
        qa_package = classifier.generate_consolidated_question(candidates, unsatisfied)
        
        if qa_package:
            print(f"🤖 AI: {qa_package['question_text']}")
            
            # 4b. Nhận câu trả lời (Giả lập nhập liệu từ terminal)
            # Ví dụ user nhập: "Tôi thấy đèn gió và hình như có cả trang phục khmer, nhưng chắc không có mâm cúng đâu"
            user_ans = input("👤 User (Mô tả tự nhiên): ")
            
            if user_ans:
                # 4c. Phân tích câu trả lời phức tạp
                print("⏳ AI đang phân tích câu trả lời...")
                parsed_result = classifier.analyze_complex_answer(
                    qa_package['question_text'], 
                    user_ans, 
                    qa_package['target_features']
                )
                print(f"   -> Kết quả phân tích: {json.dumps(parsed_result, ensure_ascii=False)}")
                
                # 4d. Cập nhật điểm
                final_logits = classifier.update_logits_from_consolidated_answer(
                    logits, candidates, unsatisfied, parsed_result
                )
                
                # --- BƯỚC 5: KẾT LUẬN ---
                winners, final_probs = classifier.decide_final_result(final_logits)
                
                if len(winners) == 1:
                    print(f"\n🎉 VIDEO NÀY THUỘC VỀ: {winners[0]}")
                elif len(winners) > 1:
                    print(f"\n🎉 VIDEO NÀY CÓ THỂ LÀ SỰ KẾT HỢP: {', '.join(winners)}")
                else:
                    print("\n🤔 KHÔNG XÁC ĐỊNH ĐƯỢC LỄ HỘI CỤ THỂ.")
        else:
            print("✅ Máy đã tự tin, không cần hỏi thêm.")
            winners, final_probs = classifier.decide_final_result(logits)
