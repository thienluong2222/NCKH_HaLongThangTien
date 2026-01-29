import { Injectable, signal, computed, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { 
  HistoryListResponse, 
  HistoryDetailResponse 
} from '../models/api.models';
import { firstValueFrom } from 'rxjs';

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'ai' | 'system';
  timestamp: Date;
  isFromBackend?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private readonly API_URL = environment.apiBaseUrl;
  private http = inject(HttpClient);
  
  private messagesSignal = signal<ChatMessage[]>([]);
  private isLoadingSignal = signal(false);

  readonly messages = computed(() => this.messagesSignal());
  readonly isLoading = computed(() => this.isLoadingSignal());

  /**
   * Add a system message (from frontend)
   */
  addSystemMessage(content: string): void {
    const systemMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      role: 'system',
      timestamp: new Date()
    };
    this.messagesSignal.update(msgs => [...msgs, systemMessage]);
  }

  /**
   * Add an AI message (from backend question or response)
   */
  addAIMessage(content: string, isFromBackend = true): void {
    const aiMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      role: 'ai',
      timestamp: new Date(),
      isFromBackend
    };
    this.messagesSignal.update(msgs => [...msgs, aiMessage]);
  }

  /**
   * Add user message to chat
   */
  addUserMessage(content: string): ChatMessage {
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      role: 'user',
      timestamp: new Date()
    };
    this.messagesSignal.update(msgs => [...msgs, userMessage]);
    return userMessage;
  }

  /**
   * Send message and get AI response (local fallback when backend is not involved)
   */
  sendMessage(content: string): Promise<ChatMessage> {
    const userMessage = this.addUserMessage(content);

    // Generate AI response locally for contextual hints
    return new Promise((resolve) => {
      setTimeout(() => {
        const aiResponse = this.generateAIResponse(content);
        const aiMessage: ChatMessage = {
          id: crypto.randomUUID(),
          content: aiResponse,
          role: 'ai',
          timestamp: new Date(),
          isFromBackend: false
        };
        
        this.messagesSignal.update(msgs => [...msgs, aiMessage]);
        resolve(aiMessage);
      }, 800);
    });
  }

  /**
   * Generate contextual AI response based on user input
   */
  private generateAIResponse(userMessage: string): string {
    const lowerMsg = userMessage.toLowerCase();
    
    if (lowerMsg.includes('đèn') || lowerMsg.includes('hoa đăng') || lowerMsg.includes('đèn nước')) {
      return `🏮 Tuyệt vời! Đèn hoa đăng là đặc trưng quan trọng của lễ hội Ok Om Bok. Độ tin cậy đã được cải thiện!\n\nBạn có thể cho tôi biết thêm về hoạt động khác không?`;
    }
    
    if (lowerMsg.includes('ghe ngo') || lowerMsg.includes('đua thuyền') || lowerMsg.includes('đua ghe')) {
      return `🚣 Đua ghe ngo là hoạt động đặc sắc của lễ hội Ok Om Bok! Đây là một điểm nhận diện quan trọng.\n\nCó nghi thức cúng bái nào diễn ra không?`;
    }
    
    if (lowerMsg.includes('trăng') || lowerMsg.includes('cúng') || lowerMsg.includes('rằm')) {
      return `🌙 Các nghi thức cúng trăng rằm phù hợp với lễ hội Ok Om Bok của đồng bào Khmer!\n\nBạn có nhìn thấy trang phục truyền thống nào không?`;
    }
    
    if (lowerMsg.includes('múa') || lowerMsg.includes('lân') || lowerMsg.includes('sư') || lowerMsg.includes('rồng')) {
      return `🎭 Múa lân sư rồng có thể xuất hiện trong nhiều lễ hội. Điều này giúp thu hẹp phạm vi nhận diện!`;
    }
    
    if (lowerMsg.includes('chùa') || lowerMsg.includes('phật') || lowerMsg.includes('sư')) {
      return `🛕 Yếu tố tôn giáo Phật giáo thường xuất hiện trong các lễ hội của đồng bào Khmer như Chol Chnam Thmay.`;
    }

    if (lowerMsg.includes('có') || lowerMsg.includes('yes') || lowerMsg.includes('đúng')) {
      return `✅ Cảm ơn xác nhận! Thông tin này giúp tăng độ chính xác của phân tích.\n\nCòn đặc điểm nào khác bạn nhận thấy không?`;
    }

    if (lowerMsg.includes('không') || lowerMsg.includes('no') || lowerMsg.includes('chưa')) {
      return `📝 Đã ghi nhận. Tôi sẽ điều chỉnh phân tích dựa trên thông tin này.\n\nBạn có thể mô tả thêm những gì bạn thấy trong video không?`;
    }
    
    return `📝 Cảm ơn thông tin! Tôi đã ghi nhận và cập nhật phân tích.\n\nBạn có thể mô tả thêm về: đèn hoa đăng, đua ghe, hoặc các nghi lễ khác không?`;
  }

  /**
   * Get analysis history list from backend
   */
  async getHistory(limit = 50, offset = 0): Promise<HistoryListResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HistoryListResponse>(
          `${this.API_URL}/history?limit=${limit}&offset=${offset}`
        )
      );
    } catch (error) {
      console.error('Failed to fetch history:', error);
      return null;
    }
  }

  /**
   * Get detailed history by ID
   */
  async getHistoryDetail(historyId: string): Promise<HistoryDetailResponse | null> {
    try {
      return await firstValueFrom(
        this.http.get<HistoryDetailResponse>(`${this.API_URL}/history/${historyId}`)
      );
    } catch (error) {
      console.error('Failed to fetch history detail:', error);
      return null;
    }
  }

  /**
   * Delete a history record
   */
  async deleteHistory(historyId: string): Promise<boolean> {
    try {
      await firstValueFrom(
        this.http.delete(`${this.API_URL}/history/${historyId}`)
      );
      return true;
    } catch (error) {
      console.error('Failed to delete history:', error);
      return false;
    }
  }

  /**
   * Clear all history
   */
  async clearAllHistory(): Promise<boolean> {
    try {
      await firstValueFrom(
        this.http.delete(`${this.API_URL}/history`)
      );
      return true;
    } catch (error) {
      console.error('Failed to clear history:', error);
      return false;
    }
  }

  /**
   * Clear local chat messages
   */
  clearHistory(): void {
    this.messagesSignal.set([]);
  }
}
