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
from langchain_core.output_parsers import StrOutputParser
from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL
import math


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



def check_constraints(detections, CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL=None, score_threshold=0.7):
    """
    Hàm kiểm tra ràng buộc (Logic đã tinh chỉnh).

    Args:
        detections: List[ObjectDetection] - Output từ YOLO sau khi qua xử lý
        CONSTRAINTS_DB: Dict - Cơ sở dữ liệu luật
        SUBCLASS_TO_FESTIVAL: Dict (Optional) - Dùng để lọc nhanh ứng viên
        score_threshold: Float - Ngưỡng điểm để chấp nhận kết quả

    Returns:
        Dict: Cấu trúc kết quả giữ nguyên như cũ.
    """

    # --- 1. Tiền xử lý dữ liệu (Indexing) ---
    # Gom nhóm để truy xuất nhanh O(1) thay vì loop nhiều lần
    detections_by_subclass = defaultdict(list)
    detections_by_frame = defaultdict(list)

    for det in detections:
        # Chỉ xét các object có mapping hợp lệ
        detections_by_subclass[det.subclass].append(det)
        detections_by_frame[det.frame_id].append(det)

    # Hàm phụ kiểm tra vị trí (IS_ON)
    def check_is_on_logic(top_subclass, bottom_subclass):
        # Duyệt qua các frame có xuất hiện cả 2 loại object
        relevant_frames = set(d.frame_id for d in detections_by_subclass[top_subclass]) & \
                        set(d.frame_id for d in detections_by_subclass[bottom_subclass])

        for fid in relevant_frames:
            tops = [d for d in detections_by_frame[fid] if d.subclass == top_subclass]
            bottoms = [d for d in detections_by_frame[fid] if d.subclass == bottom_subclass]

            for t in tops:
                for b in bottoms:
                    for box_t in t.bboxs: # box_t: [x1, y1, x2, y2]
                        for box_b in b.bboxs:
                            # Kiểm tra overlap trục X (ngang)
                            x_overlap = max(0, min(box_t[2], box_b[2]) - max(box_t[0], box_b[0]))
                            width_t = box_t[2] - box_t[0]

                            # Kiểm tra trục Y: Đáy của Top phải nằm gần Đỉnh của Bottom
                            # box[1]=y1 (top), box[3]=y2 (bottom) - Giả sử trục y hướng xuống
                            vertical_gap = box_b[1] - box_t[3]

                            # Logic: Overlap ngang > 30% width vật trên VÀ khoảng cách dọc < 20px
                            if width_t > 0 and (x_overlap / width_t) > 0.3 and -50 <= vertical_gap <= 50:
                                return True
        return False

    # --- 2. Lọc ứng viên (Candidate Filtering) ---
    if SUBCLASS_TO_FESTIVAL:
        detected_subclasses = set(detections_by_subclass.keys())
        candidate_festivals = set()
        for sub in detected_subclasses:
            if sub in SUBCLASS_TO_FESTIVAL:
                candidate_festivals.update(SUBCLASS_TO_FESTIVAL[sub])

        if not candidate_festivals:
            candidate_festivals = set(CONSTRAINTS_DB.keys())
    else:
        candidate_festivals = set(CONSTRAINTS_DB.keys())

    # --- 3. Đánh giá từng lễ hội ---
    festival_results = {}

    for festival in candidate_festivals:
        # Nếu lễ hội không có trong DB luật thì bỏ qua
        if festival not in CONSTRAINTS_DB:
            continue

        constraints = CONSTRAINTS_DB[festival]
        total_weight_achieved = 0.0
        total_weight_possible = 0.0
        hard_failed = False

        # Lưu chi tiết từng luật để debug/giải thích
        rule_details = []

        for (ctype, params, is_hard, weight, threshold) in constraints:
            satisfied = False
            current_val = 0 # Giá trị thực tế đo được (để so sánh với threshold)

            # --- LOGIC TỪNG LOẠI RÀNG BUỘC ---

            # 1. IS_PRESENCE: Có xuất hiện trong video không?
            if ctype == "is_presence":
                # Logic: Tất cả params phải có mặt
                missing_params = [p for p in params if p not in detections_by_subclass]
                satisfied = len(missing_params) == 0

            # 2. IS_PRESENCE_IN_FRAME: Cùng xuất hiện trong 1 frame
            elif ctype == "is_presence_in_frame":
                # Logic: Tìm xem có frame nào chứa đủ tất cả params không
                for fid, dets in detections_by_frame.items():
                    subs_in_frame = {d.subclass for d in dets}
                    if all(p in subs_in_frame for p in params):
                        satisfied = True
                        break

            # 3. AT_LEAST: Tổng số lượng (Cộng dồn count) >= Threshold
            elif ctype == "at_least":
                # Logic: Tổng count của tất cả params >= threshold
                total_count = 0
                for p in params:
                    if p in detections_by_subclass:
                        total_count += sum(d.count for d in detections_by_subclass[p])
                current_val = total_count
                satisfied = total_count >= (threshold or 1)

            # 4. AT_LEAST_IN_FRAME: (Giữ nguyên logic cũ hoặc hiểu là xuất hiện cùng nhau >= N lần)
            # Theo code cũ của bạn: Check xem có frame nào chứa đủ params và count >= threshold
            elif ctype == "at_least_in_frame":
                for fid, dets in detections_by_frame.items():
                    subs_in_frame = {d.subclass for d in dets}
                    # Check đủ loại
                    if all(p in subs_in_frame for p in params):
                        # Check đủ lượng (tổng lượng của các params trong frame này)
                        frame_count = sum(d.count for d in dets if d.subclass in params)
                        if frame_count >= (threshold or 1):
                            satisfied = True
                            break

            # 5. CONFIDENCE_MIN: Độ tin cậy trung bình >= Threshold
            elif ctype == "confidence_min":
                target_subs = []
                if "all" in params:
                    target_subs = list(detections_by_subclass.keys())
                else:
                    target_subs = [p for p in params if p in detections_by_subclass]

                if target_subs:
                    # Tính trung bình có trọng số (weighted by count)
                    total_conf = 0
                    total_cnt = 0
                    for sub in target_subs:
                        for d in detections_by_subclass[sub]:
                            total_conf += d.confidence * d.count
                            total_cnt += d.count

                    avg_conf = total_conf / total_cnt if total_cnt > 0 else 0
                    current_val = avg_conf
                    satisfied = avg_conf >= (threshold or 0)
                else:
                    satisfied = False # Không tìm thấy đối tượng để check confidence

            # 6. IS_ON: Vị trí tương đối
            elif ctype == "is_on" and len(params) == 2:
                satisfied = check_is_on_logic(params[0], params[1])

            # --- TÍNH ĐIỂM ---
            total_weight_possible += weight

            if satisfied:
                total_weight_achieved += weight
            elif is_hard:
                hard_failed = True

            # Lưu log (nếu cần mở rộng sau này)
            # rule_details.append({"type": ctype, "satisfied": satisfied, "hard": is_hard})

        # --- TỔNG HỢP KẾT QUẢ CHO LỄ HỘI ---

        # Điểm chuẩn hóa (Normalized Score): Luôn từ 0.0 đến 1.0
        normalized_score = 0.0
        if total_weight_possible > 0:
            normalized_score = total_weight_achieved / total_weight_possible

        festival_results[festival] = {
            "score": total_weight_achieved,      # Điểm thô (User muốn xem cộng dồn)
            "normalized_score": normalized_score, # Điểm dùng để so sánh (đã chia tổng) --------------
            "hard_failed": hard_failed,
            "satisfied": (not hard_failed) and (total_weight_achieved >= score_threshold) #---------------------
        }

    # --- 4. Chọn kết quả tốt nhất ---
    # Lọc ra các lễ hội thỏa mãn điều kiện
    valid_festivals = {
        f: r["normalized_score"]
        for f, r in festival_results.items()
        if r["satisfied"]
    }

    if valid_festivals:
        # Chọn lễ hội có điểm chuẩn hóa cao nhất
        best_festival = max(valid_festivals, key=valid_festivals.get)
        return {
            "festival": best_festival,
            "score": valid_festivals[best_festival],
            "details": festival_results
        }
    else:
        # Fallback: Nếu không ai đạt threshold, trả về None hoặc người có điểm cao nhất (nhưng satisfied=False)
        return {
            "festival": None,
            "score": 0.0,
            "details": festival_results
        }

def get_unsatisfied_constraints(candidate_name, detections):
    """
    Hàm này thay thế/bổ sung cho model.py.
    Nó kiểm tra lại lễ hội 'candidate_name' với 'detections' hiện có 
    và trả về danh sách các luật KHÔNG thỏa mãn.
    
    Returns:
        list of tuples: [(rule, weight_of_rule), ...]
    """
    if candidate_name not in CONSTRAINTS_DB:
        return []

    constraints = CONSTRAINTS_DB[candidate_name]
    
    # Indexing detections
    detections_by_subclass = defaultdict(list)
    detections_by_frame = defaultdict(list)
    for det in detections:
        detections_by_subclass[det.subclass].append(det)
        detections_by_frame[det.frame_id].append(det)

    unsatisfied = []

    for rule in constraints:
        ctype, params, is_hard, weight, threshold = rule
        satisfied = False
        
        # --- Logic kiểm tra (giống model.py) ---
        if ctype == "is_presence":
            missing = [p for p in params if p not in detections_by_subclass]
            satisfied = len(missing) == 0
            
        elif ctype == "is_presence_in_frame":
            # Cần tất cả params xuất hiện trong cùng 1 frame bất kỳ
            for fid, dets in detections_by_frame.items():
                subs = {d.subclass for d in dets}
                if all(p in subs for p in params):
                    satisfied = True; break
                    
        elif ctype == "at_least":
            threshold = threshold or 1
            total_count = 0
            for p in params:
                if p in detections_by_subclass:
                    total_count += sum(d.count for d in detections_by_subclass[p])
            satisfied = total_count >= threshold
            
        elif ctype == "at_least_in_frame":
            threshold = threshold or 1
            for fid, dets in detections_by_frame.items():
                frame_cnt = sum(d.count for d in dets if d.subclass in params)
                if frame_cnt >= threshold:
                    satisfied = True; break
                    
        elif ctype == "confidence_min":
            # Logic này hơi khó hỏi user, nhưng cứ đưa vào kiểm tra
            target_subs = list(detections_by_subclass.keys()) if "all" in params else [p for p in params if p in detections_by_subclass]
            if target_subs:
                total_conf = sum(d.confidence * d.count for sub in target_subs for d in detections_by_subclass[sub])
                total_cnt = sum(d.count for sub in target_subs for d in detections_by_subclass[sub])
                avg = total_conf / total_cnt if total_cnt > 0 else 0
                satisfied = avg >= (threshold or 0)
            else:
                satisfied = False

        elif ctype == "is_on":
            # Logic is_on đơn giản hóa cho việc check
            if len(params) == 2:
                # Nếu cả 2 cùng xuất hiện trong video thì tạm coi là thỏa (để user confirm lại mối quan hệ)
                satisfied = all(p in detections_by_subclass for p in params)
            else:
                satisfied = False

        # Nếu không thỏa, thêm vào list unsatisfied
        if not satisfied:
            unsatisfied.append(rule)

    return unsatisfied

def calculate_initial_score(candidate_name, detections):
    """Tính điểm ban đầu dựa trên AI detection"""
    if candidate_name not in CONSTRAINTS_DB: return 0.0, 1.0
    
    constraints = CONSTRAINTS_DB[candidate_name]
    total_possible = 0.0
    total_achieved = 0.0
    
    # Tái sử dụng logic get_unsatisfied để biết cái nào satisfied (ngược lại)
    # Tuy nhiên để tối ưu tốc độ, ta code nhanh logic tính tổng
    unsatisfied_rules = get_unsatisfied_constraints(candidate_name, detections)
    # Chuyển list rule thành set để tra cứu
    # Lưu ý: list không hashable, nên ta so sánh identity hoặc content
    
    for rule in constraints:
        weight = rule[3]
        total_possible += weight
        if rule not in unsatisfied_rules:
            total_achieved += weight
            
    return total_achieved, total_possible

def generate_question(candidate_name, rule, api_key):
    """Sinh câu hỏi dựa trên luật bị thiếu"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    ctype, params, is_hard, weight, threshold = rule
    
    prompt_template = """
    Bạn là trợ lý AI xác thực lễ hội.
    Lễ hội đang xét: "{candidate}".
    Luật chưa thỏa mãn: Loại "{ctype}", Đối tượng liên quan "{params}".
    
    Hãy đặt một câu hỏi ngắn gọn (dưới 20 từ) cho người dùng để xác nhận xem họ có thấy các yếu tố này trong video không.
    - Nếu là 'is_presence'/'at_least': Hỏi có thấy [đối tượng] không.
    - Nếu là 'is_on': Hỏi có thấy [đối tượng 1] nằm trên/cạnh [đối tượng 2] không.
    - Nếu là 'at_least_in_frame': Hỏi có thấy nhiều [đối tượng] xuất hiện cùng lúc cùng nhau không.
    - Nếu là 'is_presence_in_frame': Hỏi có thấy các đối tượng {params} xuất hiện cùng nhau không.
    
    Câu hỏi:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"candidate": candidate_name, "ctype": ctype, "params": ", ".join(params)})

def analyze_user_response(question, user_answer, api_key):
    """Phân tích câu trả lời Yes/No"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    
    prompt_template = """
    AI hỏi: "{question}"
    User trả lời: "{answer}"
    
    Hãy xác định ý định của User:
    - Nếu User xác nhận có thấy/đúng -> Trả về "YES"
    - Nếu User phủ nhận/không thấy -> Trả về "NO"
    - Nếu không rõ -> Trả về "UNKNOWN"
    
    Chỉ trả về đúng 1 từ kết quả.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "answer": user_answer}).strip()


# Ví dụ sử dụng
if __name__ == "__main__":

    from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL

    model_path = '../weight/best_openvino_model/'
    csv_path = '../artifacts/merged_data.csv'
    video_path = '../assets/input/1.mp4'
    output_video_path = '../assets/output/1_detected.mp4'

    # Khởi tạo pipeline
    pipeline = YOLOCSVPipeline(
        model_path=model_path,
        csv_path=csv_path
    )

    # Xử lý video và lưu kết quả
    result = pipeline.process_video_with_output(
        video_path=video_path,
        output_path=output_video_path,  # None = tự động tạo tên
        confidence_threshold=0.5,
        fps_detect=1,  # Detect 1 frame/giây
        max_duration=30,  # Xử lý tối đa 30 giây
        output_fps=None,  # None = giữ nguyên FPS gốc
        save_frames=False,  # Lưu các frame đã detect
        output_folder='detected_frames',
        top_k=10
    )

    # Chuyển đổi kết quả của hàm dự đoán thành list đối tượng
    summary = result

    object_detections = []  # danh sách ObjectDetection

    for frame_data in summary['frame_details']:
        frame_id = frame_data['frame']
        time_stamp = frame_data['time']
        detections = frame_data['detections']

        # Gom nhóm detection theo subclass (label)
        subclass_groups = {}
        for det in detections:
            if det['mapped']:
                subclass = det['label']
                if subclass not in subclass_groups:
                    subclass_groups[subclass] = {'confidences': [], 'bboxs': []}
                subclass_groups[subclass]['confidences'].append(det['confidence'])
                subclass_groups[subclass]['bboxs'].append(det['box'].tolist())  # numpy → list

        # Tạo ObjectDetection cho mỗi subclass trong frame
        for subclass, data in subclass_groups.items():
            avg_conf = np.mean(data['confidences']) if data['confidences'] else 0.0
            count = len(data['bboxs'])  # số lần subclass xuất hiện trong frame

            obj = ObjectDetection(
                subclass=subclass,
                confidence=avg_conf,
                frame_id=frame_id,
                time_stamp=time_stamp,
                count=count,
                bboxs=data['bboxs']  # danh sách bounding boxes
            )
            object_detections.append(obj)

    print(f"✅ Đã tạo {len(object_detections)} đối tượng ObjectDetection (có bboxs).")

    print(check_constraints(object_detections, CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL, score_threshold=1.0))
