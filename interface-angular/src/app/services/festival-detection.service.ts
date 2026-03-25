import { Injectable, signal, computed } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { 
  VideoAnalysisResponse, 
  VideoNeedsClarificationResponse,
  VideoFinishedResponse,
  AnswerResponse,
  Top3Festival,
  HealthResponse,
  FestivalsResponse,
  FestivalConstraints 
} from '../models/api.models';
import { firstValueFrom } from 'rxjs';

export interface FestivalResult {
  name: string;
  confidence: number;
  description: string;
  color?: string;
}

export interface DetectionState {
  isLoading: boolean;
  results: FestivalResult[];
  winner: FestivalResult | null;
  uploadedFiles: File[];
  topThree: FestivalResult[];
  requestId: string | null;
  historyId: string | null;
  question: string | null;
  targetFeatures: string[];
  status: 'idle' | 'analyzing' | 'needs_clarification' | 'finished' | 'error';
  errorMessage: string | null;
  constraintsData: FestivalConstraints[];  // Store constraints for graph visualization
}

@Injectable({
  providedIn: 'root'
})
export class FestivalDetectionService {
  private readonly API_URL = environment.apiBaseUrl;
  
  // Festival color mapping
  private festivalColors: Record<string, string> = {
    'Ok Om Bok': '#D4A853',
    'Chol Chnam Thmay': '#8B2323',
    'Tết Nguyên Đán': '#E53935',
    'Lễ hội Đền Hùng': '#1A5F5F',
    'Lễ hội Chùa Hương': '#7B1FA2',
    'Lễ hội Gióng': '#00695C',
    'Lễ hội Tháp Bà Ponagar': '#E65100'
  };

  // Festival descriptions
  private festivalDescriptions: Record<string, string> = {
    'Ok Om Bok': 'Lễ hội cúng trăng của đồng bào Khmer Nam Bộ, diễn ra vào Rằm tháng 10 âm lịch',
    'Chol Chnam Thmay': 'Tết cổ truyền của người Khmer, mừng năm mới theo Phật lịch',
    'Tết Nguyên Đán': 'Tết cổ truyền của dân tộc Việt Nam, lễ hội lớn nhất trong năm',
    'Lễ hội Đền Hùng': 'Giỗ Tổ Hùng Vương, tưởng nhớ công đức các vua Hùng dựng nước',
    'Lễ hội Chùa Hương': 'Lễ hội hành hương lớn nhất Việt Nam, diễn ra ở Mỹ Đức, Hà Nội',
    'Lễ hội Gióng': 'Tưởng nhớ Thánh Gióng đánh giặc Ân, Di sản văn hóa phi vật thể UNESCO',
    'Lễ hội Tháp Bà Ponagar': 'Lễ hội của người Chăm tại Nha Trang, thờ Nữ thần Thiên Y A Na'
  };
  
  // Signals for reactive state
  private state = signal<DetectionState>({
    isLoading: false,
    results: [],
    winner: null,
    uploadedFiles: [],
    topThree: [],
    requestId: null,
    historyId: null,
    question: null,
    targetFeatures: [],
    status: 'idle',
    errorMessage: null,
    constraintsData: []
  });

  // Computed values
  readonly isLoading = computed(() => this.state().isLoading);
  readonly results = computed(() => this.state().results);
  readonly winner = computed(() => this.state().winner);
  readonly uploadedFiles = computed(() => this.state().uploadedFiles);
  readonly hasResults = computed(() => this.state().results.length > 0);
  readonly topThree = computed(() => this.state().topThree);
  readonly requestId = computed(() => this.state().requestId);
  readonly historyId = computed(() => this.state().historyId);
  readonly question = computed(() => this.state().question);
  readonly targetFeatures = computed(() => this.state().targetFeatures);
  readonly status = computed(() => this.state().status);
  readonly errorMessage = computed(() => this.state().errorMessage);
  readonly needsClarification = computed(() => this.state().status === 'needs_clarification');
  readonly constraintsData = computed(() => this.state().constraintsData);

  constructor(private http: HttpClient) {}

  /**
   * Check backend health status
   */
  async checkHealth(): Promise<HealthResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HealthResponse>(`${this.API_URL}/health`)
      );
    } catch (error) {
      console.error('Health check failed:', error);
      return null;
    }
  }

  /**
   * Get list of supported festivals
   */
  async getFestivals(): Promise<FestivalsResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<FestivalsResponse>(`${this.API_URL}/festivals`)
      );
    } catch (error) {
      console.error('Failed to fetch festivals:', error);
      return null;
    }
  }

  /**
   * Analyze uploaded files (videos/images) for festival detection
   * Uses unified /api/analyze endpoint for both media types
   */
  async analyzeFiles(files: File[]): Promise<FestivalResult[]> {
    this.state.update(s => ({ 
      ...s, 
      isLoading: true, 
      uploadedFiles: files,
      status: 'analyzing',
      errorMessage: null 
    }));

    // Find media file (video or image)
    const mediaFile = files.find(f => 
      f.type.startsWith('video/') || f.type.startsWith('image/')
    );
    
    if (!mediaFile) {
      this.handleError({ message: 'Không tìm thấy file hình ảnh hoặc video hợp lệ' });
      return [];
    }

    try {
      const formData = new FormData();
      formData.append('file', mediaFile);  // Backend expects 'file' field

      console.log('Sending request to:', `${this.API_URL}/analyze`);
      console.log('File:', mediaFile.name, mediaFile.type, mediaFile.size);

      const response = await firstValueFrom(
        this.http.post<VideoAnalysisResponse>(`${this.API_URL}/analyze`, formData)
      );

      console.log('API Response received:', response);
      
      try {
        return this.processVideoResponse(response);
      } catch (processingError) {
        console.error('Error processing response:', processingError);
        throw processingError;
      }
    } catch (error) {
      console.error('Media analysis failed:', error);
      this.handleError(error);
      return [];
    }
  }

  /**
   * Submit answer to clarification question
   */
  async submitAnswer(answer: string): Promise<FestivalResult[]> {
    const currentRequestId = this.state().requestId;
    
    if (!currentRequestId) {
      console.error('No request ID available');
      return this.state().results;
    }

    this.state.update(s => ({ ...s, isLoading: true }));

    try {
      const response = await firstValueFrom(
        this.http.post<AnswerResponse>(`${this.API_URL}/answer`, {
          request_id: currentRequestId,
          answer: answer
        })
      );

      return this.processAnswerResponse(response);
    } catch (error) {
      console.error('Submit answer failed:', error);
      this.handleError(error);
      return this.state().results;
    }
  }

  /**
   * Process video analysis response
   */
  private processVideoResponse(response: VideoAnalysisResponse): FestivalResult[] {
    console.log('Processing API response:', response);
    
    // Extract top_3 from top_3_constraints if top_3 is not present
    const top3Data = response.top_3 || this.extractTop3FromConstraints(response.top_3_constraints || []);
    console.log('Top 3 data:', top3Data);
    
    const results = this.convertTop3ToResults(top3Data);
    console.log('Converted results:', results);
    
    // Store constraints data for graph visualization
    const constraintsData = response.top_3_constraints || [];
    
    if (response.status === 'needs_clarification') {
      const clarificationResponse = response as VideoNeedsClarificationResponse;
      this.state.update(s => ({
        ...s,
        isLoading: false,
        results,
        winner: results[0] || null,
        topThree: results,
        requestId: clarificationResponse.request_id,
        historyId: clarificationResponse.history_id,
        question: clarificationResponse.question,
        targetFeatures: clarificationResponse.target_features,
        status: 'needs_clarification',
        constraintsData
      }));
    } else {
      const finishedResponse = response as VideoFinishedResponse;
      this.state.update(s => ({
        ...s,
        isLoading: false,
        results,
        winner: results[0] || null,
        topThree: results,
        historyId: finishedResponse.history_id,
        requestId: null,
        question: null,
        targetFeatures: [],
        status: 'finished',
        constraintsData
      }));
    }

    return results;
  }

  /**
   * Process answer submission response
   */
  private processAnswerResponse(response: AnswerResponse): FestivalResult[] {
    // Extract top_3 from top_3_constraints if top_3 is not present
    const top3Data = response.top_3 || this.extractTop3FromConstraints(response.top_3_constraints);
    const results = this.convertTop3ToResults(top3Data);
    
    // Store constraints data for graph visualization
    const constraintsData = response.top_3_constraints || [];

    if (response.status === 'needs_clarification') {
      this.state.update(s => ({
        ...s,
        isLoading: false,
        results,
        winner: results[0] || null,
        topThree: results,
        requestId: response.request_id || s.requestId,
        question: response.question || null,
        targetFeatures: response.target_features || [],
        status: 'needs_clarification',
        constraintsData
      }));
    } else {
      this.state.update(s => ({
        ...s,
        isLoading: false,
        results,
        winner: results[0] || null,
        topThree: results,
        requestId: null,
        question: null,
        targetFeatures: [],
        status: 'finished',
        constraintsData
      }));
    }

    return results;
  }

  /**
   * Convert Top3Festival array to FestivalResult array
   */
  private convertTop3ToResults(top3: Top3Festival[]): FestivalResult[] {
    return top3.map(item => ({
      name: item.festival,
      confidence: Math.round(item.confidence * 100), // Convert 0-1 to 0-100
      description: this.festivalDescriptions[item.festival] || 'Lễ hội truyền thống Việt Nam',
      color: this.festivalColors[item.festival] || '#D4A853'
    }));
  }

  /**
   * Extract Top3Festival from FestivalConstraints array
   * Used when API returns top_3_constraints but not top_3
   */
  private extractTop3FromConstraints(constraints: FestivalConstraints[]): Top3Festival[] {
    return constraints.map(c => ({
      festival: c.festival,
      confidence: c.confidence
    }));
  }

  /**
   * Handle API errors
   */
  private handleError(error: any): void {
    let errorMessage = 'Đã xảy ra lỗi khi xử lý yêu cầu';
    
    if (error instanceof HttpErrorResponse) {
      if (error.error?.error) {
        errorMessage = error.error.error;
      } else if (error.status === 0) {
        errorMessage = 'Không thể kết nối đến server. Vui lòng kiểm tra backend đang chạy.';
      } else if (error.status === 413) {
        errorMessage = 'File quá lớn. Vui lòng chọn file nhỏ hơn 500MB.';
      } else if (error.status === 503) {
        errorMessage = 'Services backend chưa sẵn sàng. Vui lòng thử lại sau.';
      }
    }

    this.state.update(s => ({
      ...s,
      isLoading: false,
      status: 'error',
      errorMessage
    }));
  }

  /**
   * Mock analysis for images or when backend is unavailable
   */
  private mockAnalysis(files: File[]): Promise<FestivalResult[]> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockResults: FestivalResult[] = [
          { 
            name: 'Ok Om Bok', 
            confidence: 78, 
            description: this.festivalDescriptions['Ok Om Bok'],
            color: this.festivalColors['Ok Om Bok']
          },
          { 
            name: 'Chol Chnam Thmay', 
            confidence: 52, 
            description: this.festivalDescriptions['Chol Chnam Thmay'],
            color: this.festivalColors['Chol Chnam Thmay']
          },
          { 
            name: 'Tết Nguyên Đán', 
            confidence: 35, 
            description: this.festivalDescriptions['Tết Nguyên Đán'],
            color: this.festivalColors['Tết Nguyên Đán']
          }
        ];

        this.state.update(s => ({
          ...s,
          isLoading: false,
          results: mockResults,
          winner: mockResults[0],
          topThree: mockResults,
          status: 'needs_clarification',
          question: 'Trong video có xuất hiện đèn hoa đăng hoặc hoạt động đua ghe ngo không?',
          targetFeatures: ['đèn hoa đăng', 'ghe ngo']
        }));

        resolve(mockResults);
      }, 2500);
    });
  }

  /**
   * Update results based on chat messages (Human-in-the-Loop)
   * This will submit answer to backend if request_id exists
   */
  async updateFromChat(message: string): Promise<void> {
    if (this.state().requestId) {
      // Submit answer to backend
      await this.submitAnswer(message);
    } else {
      // Fallback: local confidence boost (for mock mode)
      this.localConfidenceBoost(message);
    }
  }

  /**
   * Local confidence boost when backend is not available
   */
  private localConfidenceBoost(message: string): void {
    const lowerMsg = message.toLowerCase();
    let boostAmount = 0;
    
    if (lowerMsg.includes('đèn hoa đăng') || lowerMsg.includes('đèn nước')) {
      boostAmount = 5;
    } else if (lowerMsg.includes('ghe ngo') || lowerMsg.includes('đua thuyền')) {
      boostAmount = 6;
    } else if (lowerMsg.includes('cúng trăng') || lowerMsg.includes('rằm')) {
      boostAmount = 4;
    } else if (lowerMsg.includes('múa') || lowerMsg.includes('lân')) {
      boostAmount = 2;
    } else if (lowerMsg.includes('có') || lowerMsg.includes('yes')) {
      boostAmount = 3;
    } else {
      boostAmount = 1;
    }
    
    if (boostAmount > 0) {
      this.state.update(s => {
        const updatedResults = s.results.map((r, i) => 
          i === 0 ? { ...r, confidence: Math.min(r.confidence + boostAmount, 99) } : r
        );
        
        const newWinner = s.winner ? { ...s.winner, confidence: Math.min(s.winner.confidence + boostAmount, 99) } : s.winner;
        const shouldComplete = newWinner && newWinner.confidence >= 85;
        
        return {
          ...s,
          winner: newWinner,
          results: updatedResults,
          topThree: updatedResults.slice(0, 3),
          status: shouldComplete ? 'finished' : s.status
        };
      });
    }
  }

  /**
   * Delete all active sessions on backend
   */
  async deleteAllSessions(): Promise<{ deleted_count: number } | null> {
    try {
      const response = await firstValueFrom(
        this.http.delete<{ message: string; deleted_count: number; active_sessions: number }>(
          `${this.API_URL}/sessions`
        )
      );
      console.log('Sessions cleared:', response);
      return { deleted_count: response.deleted_count };
    } catch (error) {
      console.error('Failed to delete sessions:', error);
      return null;
    }
  }

  /**
   * Reset all state
   */
  reset(): void {
    this.state.set({
      isLoading: false,
      results: [],
      winner: null,
      uploadedFiles: [],
      topThree: [],
      requestId: null,
      historyId: null,
      question: null,
      targetFeatures: [],
      status: 'idle',
      errorMessage: null,
      constraintsData: []
    });
  }
}
