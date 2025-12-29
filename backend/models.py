from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


@dataclass
class ConstraintResult:
    """Kết quả kiểm tra một ràng buộc"""
    type: str                    # Loại ràng buộc: is_presence, at_least, is_on, etc.
    params: List[str]            # Các tham số (subclass names)
    is_hard: bool                # Hard constraint hay soft
    weight: float                # Trọng số
    threshold: Optional[float]   # Ngưỡng (nếu có)
    satisfied: bool              # Đã thỏa mãn chưa
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FestivalConstraints:
    """Ràng buộc thỏa mãn/không thỏa mãn của một lễ hội"""
    festival: str
    confidence: float
    satisfied: List[ConstraintResult] = field(default_factory=list)
    unsatisfied: List[ConstraintResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "festival": self.festival,
            "confidence": self.confidence,
            "satisfied": [c.to_dict() for c in self.satisfied],
            "unsatisfied": [c.to_dict() for c in self.unsatisfied]
        }


@dataclass
class DetectedObject:
    """Đối tượng được phát hiện"""
    subclass: str
    confidence: float
    frame_id: int
    time_stamp: float
    count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class QARecord:
    """Một lượt hỏi đáp"""
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Kết quả phân tích"""
    winner: Optional[str]
    top_3: List[Dict]                           # [{"festival": str, "confidence": float}]
    top_3_constraints: List[FestivalConstraints] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "winner": self.winner,
            "top_3": self.top_3,
            "top_3_constraints": [c.to_dict() for c in self.top_3_constraints]
        }


@dataclass
class AnalysisHistory:
    """Lịch sử một phiên phân tích"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    detected_objects: List[DetectedObject] = field(default_factory=list)
    result: Optional[AnalysisResult] = None
    qa_history: List[QARecord] = field(default_factory=list)
    status: str = "pending"  # pending, needs_clarification, finished
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "timestamp": self.timestamp,
            "detected_objects": [obj.to_dict() for obj in self.detected_objects],
            "result": self.result.to_dict() if self.result else None,
            "qa_history": [qa.to_dict() for qa in self.qa_history],
            "status": self.status
        }
    
    def to_summary(self) -> Dict:
        """Trả về bản tóm tắt (không bao gồm detected_objects chi tiết)"""
        return {
            "id": self.id,
            "filename": self.filename,
            "timestamp": self.timestamp,
            "detected_objects_count": len(self.detected_objects),
            "result": self.result.to_dict() if self.result else None,
            "qa_count": len(self.qa_history),
            "status": self.status
        }
