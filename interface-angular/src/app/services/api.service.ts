import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { 
  HealthResponse,
  FestivalsResponse,
  VideoAnalysisResponse,
  AnswerRequest,
  AnswerResponse,
  HistoryListResponse,
  HistoryDetailResponse
} from '../models/api.models';
import { firstValueFrom, catchError, of } from 'rxjs';

/**
 * API Service - Centralized service for backend communication
 * Base URL: /api (proxied through nginx in production)
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly baseUrl = environment.apiBaseUrl;
  private http = inject(HttpClient);

  // ==========================================
  // Health Check
  // ==========================================
  
  /**
   * Check backend health status
   * GET /api/health
   */
  async checkHealth(): Promise<HealthResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HealthResponse>(`${this.baseUrl}/health`)
      );
    } catch (error) {
      this.logError('Health check failed', error);
      return null;
    }
  }

  /**
   * Check if backend is available
   */
  async isBackendAvailable(): Promise<boolean> {
    const health = await this.checkHealth();
    return health?.status === 'healthy';
  }

  // ==========================================
  // Festivals
  // ==========================================

  /**
   * Get list of supported festivals
   * GET /api/festivals
   */
  async getFestivals(): Promise<FestivalsResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<FestivalsResponse>(`${this.baseUrl}/festivals`)
      );
    } catch (error) {
      this.logError('Failed to fetch festivals', error);
      return null;
    }
  }

  // ==========================================
  // Media Analysis (Video/Image)
  // ==========================================

  /**
   * Upload and analyze media (video or image) for festival detection
   * POST /api/analyze
   * @param mediaFile - Video or image file to analyze
   */
  async analyzeMedia(mediaFile: File): Promise<VideoAnalysisResponse | null> {
    try {
      const formData = new FormData();
      formData.append('file', mediaFile);

      return await firstValueFrom(
        this.http.post<VideoAnalysisResponse>(`${this.baseUrl}/analyze`, formData)
      );
    } catch (error) {
      this.logError('Media analysis failed', error);
      throw error; // Re-throw to let caller handle
    }
  }

  /**
   * @deprecated Use analyzeMedia instead
   * Legacy method - redirects to analyzeMedia
   */
  async analyzeVideo(videoFile: File): Promise<VideoAnalysisResponse | null> {
    return this.analyzeMedia(videoFile);
  }

  /**
   * Submit answer for Human-in-the-Loop clarification
   * POST /api/answer
   * @param requestId - Request ID from analysis
   * @param answer - User's answer to clarification question
   */
  async submitAnswer(requestId: string, answer: string): Promise<AnswerResponse | null> {
    try {
      return await firstValueFrom(
        this.http.post<AnswerResponse>(`${this.baseUrl}/answer`, {
          request_id: requestId,
          answer: answer
        })
      );
    } catch (error) {
      this.logError('Submit answer failed', error);
      throw error;
    }
  }

  // ==========================================
  // History
  // ==========================================

  /**
   * Get analysis history list
   * GET /api/history
   * @param limit - Number of records (default 50)
   * @param offset - Pagination offset (default 0)
   */
  async getHistory(limit = 50, offset = 0): Promise<HistoryListResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HistoryListResponse>(
          `${this.baseUrl}/history`,
          { params: { limit: limit.toString(), offset: offset.toString() } }
        )
      );
    } catch (error) {
      this.logError('Failed to fetch history', error);
      return null;
    }
  }

  /**
   * Get history detail by ID
   * GET /api/history/:id
   * @param historyId - History record ID
   */
  async getHistoryDetail(historyId: string): Promise<HistoryDetailResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HistoryDetailResponse>(`${this.baseUrl}/history/${historyId}`)
      );
    } catch (error) {
      this.logError('Failed to fetch history detail', error);
      return null;
    }
  }

  /**
   * Delete a history record
   * DELETE /api/history/:id
   * @param historyId - History record ID to delete
   */
  async deleteHistory(historyId: string): Promise<boolean> {
    try {
      await firstValueFrom(
        this.http.delete(`${this.baseUrl}/history/${historyId}`)
      );
      return true;
    } catch (error) {
      this.logError('Failed to delete history', error);
      return false;
    }
  }

  /**
   * Clear all history records
   * DELETE /api/history
   */
  async clearAllHistory(): Promise<{ deletedCount: number } | null> {
    try {
      const response = await firstValueFrom(
        this.http.delete<{ message: string; deleted_count: number }>(
          `${this.baseUrl}/history`
        )
      );
      return { deletedCount: response.deleted_count };
    } catch (error) {
      this.logError('Failed to clear history', error);
      return null;
    }
  }

  // ==========================================
  // Error Handling
  // ==========================================

  /**
   * Parse error response into user-friendly message
   */
  parseError(error: any): string {
    if (error instanceof HttpErrorResponse) {
      if (error.error?.error) {
        return error.error.error;
      }
      
      switch (error.status) {
        case 0:
          return 'Không thể kết nối đến server. Vui lòng kiểm tra kết nối mạng.';
        case 400:
          return 'Yêu cầu không hợp lệ. Vui lòng kiểm tra dữ liệu đầu vào.';
        case 404:
          return 'Không tìm thấy tài nguyên yêu cầu.';
        case 413:
          return 'File quá lớn. Vui lòng chọn file nhỏ hơn 500MB.';
        case 500:
          return 'Lỗi server. Vui lòng thử lại sau.';
        case 503:
          return 'Services backend chưa sẵn sàng. Vui lòng thử lại sau.';
        default:
          return `Lỗi không xác định (${error.status})`;
      }
    }
    
    return 'Đã xảy ra lỗi không xác định';
  }

  private logError(message: string, error: any): void {
    console.error(`[API Service] ${message}:`, error);
  }
}
