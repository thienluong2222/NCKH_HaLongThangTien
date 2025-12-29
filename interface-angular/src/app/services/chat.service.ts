import { Injectable, signal, computed } from '@angular/core';

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'ai' | 'system';
  timestamp: Date;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private messagesSignal = signal<ChatMessage[]>([]);

  readonly messages = computed(() => this.messagesSignal());

  addSystemMessage(content: string): void {
    const systemMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      role: 'system',
      timestamp: new Date()
    };
    this.messagesSignal.update(msgs => [...msgs, systemMessage]);
  }

  sendMessage(content: string): Promise<ChatMessage> {
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      role: 'user',
      timestamp: new Date()
    };

    this.messagesSignal.update(msgs => [...msgs, userMessage]);

    // Simulate AI response
    return new Promise((resolve) => {
      setTimeout(() => {
        const aiResponse = this.generateAIResponse(content);
        const aiMessage: ChatMessage = {
          id: crypto.randomUUID(),
          content: aiResponse,
          role: 'ai',
          timestamp: new Date()
        };
        
        this.messagesSignal.update(msgs => [...msgs, aiMessage]);
        resolve(aiMessage);
      }, 800);
    });
  }

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
    
    return `📝 Cảm ơn thông tin! Tôi đã ghi nhận và cập nhật phân tích.\n\nBạn có thể mô tả thêm về: đèn hoa đăng, đua ghe, hoặc các nghi lễ khác không?`;
  }

  clearHistory() {
    this.messagesSignal.set([]);
  }
}
