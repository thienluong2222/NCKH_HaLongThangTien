from typing import Dict, List, Optional
from models import AnalysisHistory
import logging

logger = logging.getLogger(__name__)


class HistoryStore:
    """
    In-memory storage cho lịch sử phân tích.
    Lưu ý: Dữ liệu sẽ mất khi restart server.
    """
    
    def __init__(self, max_records: int = 100):
        """
        Khởi tạo store
        
        Args:
            max_records: Số lượng records tối đa lưu trữ (tránh memory leak)
        """
        self._store: Dict[str, AnalysisHistory] = {}
        self._max_records = max_records
        logger.info(f"HistoryStore khởi tạo với max_records={max_records}")
    
    def save(self, history: AnalysisHistory) -> str:
        """
        Lưu một record vào store
        
        Args:
            history: AnalysisHistory object
            
        Returns:
            ID của record đã lưu
        """
        # Xóa record cũ nhất nếu vượt quá giới hạn
        if len(self._store) >= self._max_records:
            oldest_id = min(self._store.keys(), 
                        key=lambda k: self._store[k].timestamp)
            del self._store[oldest_id]
            logger.info(f"Đã xóa record cũ nhất: {oldest_id}")
        
        self._store[history.id] = history
        logger.info(f"Đã lưu history: {history.id} ({history.filename})")
        return history.id
    
    def get(self, history_id: str) -> Optional[AnalysisHistory]:
        """
        Lấy một record theo ID
        
        Args:
            history_id: ID của record
            
        Returns:
            AnalysisHistory hoặc None nếu không tìm thấy
        """
        return self._store.get(history_id)
    
    def get_all(self, limit: int = 50, offset: int = 0) -> List[AnalysisHistory]:
        """
        Lấy danh sách records (mới nhất trước)
        
        Args:
            limit: Số lượng records tối đa
            offset: Vị trí bắt đầu
            
        Returns:
            List các AnalysisHistory
        """
        sorted_records = sorted(
            self._store.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_records[offset:offset + limit]
    
    def delete(self, history_id: str) -> bool:
        """
        Xóa một record theo ID
        
        Args:
            history_id: ID của record
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if history_id in self._store:
            del self._store[history_id]
            logger.info(f"Đã xóa history: {history_id}")
            return True
        return False
    
    def delete_all(self) -> int:
        """
        Xóa tất cả records
        
        Returns:
            Số lượng records đã xóa
        """
        count = len(self._store)
        self._store.clear()
        logger.info(f"Đã xóa tất cả {count} records")
        return count
    
    def update(self, history_id: str, history: AnalysisHistory) -> bool:
        """
        Cập nhật một record
        
        Args:
            history_id: ID của record
            history: AnalysisHistory mới
            
        Returns:
            True nếu cập nhật thành công
        """
        if history_id in self._store:
            self._store[history_id] = history
            logger.info(f"Đã cập nhật history: {history_id}")
            return True
        return False
    
    def count(self) -> int:
        """Đếm số lượng records"""
        return len(self._store)


# Singleton instance
history_store = HistoryStore()
