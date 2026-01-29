import { Component, signal, ViewChild, ElementRef, AfterViewChecked, computed, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { ChatService, ChatMessage } from '../../services/chat.service';
import { FestivalDetectionService } from '../../services/festival-detection.service';

interface PreviewFile {
  file: File;
  url: string;
  type: 'image' | 'video';
}

type WorkflowStep = 'upload' | 'analyzing' | 'questioning' | 'completed';

@Component({
  selector: 'app-chat-box',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatInputModule,
    MatFormFieldModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatTooltipModule,
    MatSnackBarModule
  ],
  templateUrl: './chat-box.component.html',
  styleUrl: './chat-box.component.scss'
})
export class ChatBoxComponent implements AfterViewChecked, OnInit {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  @ViewChild('imageInput') private imageInput!: ElementRef<HTMLInputElement>;
  @ViewChild('videoInput') private videoInput!: ElementRef<HTMLInputElement>;
  
  inputMessage = signal('');
  isTyping = signal(false);
  isDragging = signal(false);
  previews = signal<PreviewFile[]>([]);
  workflowStep = signal<WorkflowStep>('upload');
  backendConnected = signal(false);
  mediaExpanded = signal(false);
  activeMediaIndex = signal(0);
  
  // Computed state
  hasUploads = computed(() => this.previews().length > 0);
  canSendMessage = computed(() => 
    this.inputMessage().trim() && !this.isTyping() && 
    (this.workflowStep() === 'questioning' || this.workflowStep() === 'completed')
  );
  
  suggestions = [
    { icon: '🏮', text: 'Có đèn hoa đăng' },
    { icon: '🚣', text: 'Đua ghe ngo' },
    { icon: '🌙', text: 'Cúng trăng rằm' },
    { icon: '🎎', text: 'Múa lân sư rồng' }
  ];

  constructor(
    public chatService: ChatService,
    public festivalService: FestivalDetectionService,
    private snackBar: MatSnackBar
  ) {}

  async ngOnInit() {
    // Check backend health on init
    const health = await this.festivalService.checkHealth();
    this.backendConnected.set(health?.status === 'healthy');
    
    if (!this.backendConnected()) {
      console.warn('Backend không khả dụng, sử dụng chế độ mock');
    }
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  // Drag & Drop handlers
  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);
    
    const files = event.dataTransfer?.files;
    if (files) {
      this.processFiles(Array.from(files));
    }
  }

  // File selection
  triggerImageUpload() {
    this.imageInput?.nativeElement.click();
  }

  triggerVideoUpload() {
    this.videoInput?.nativeElement.click();
  }

  onFileSelect(event: Event, type: 'image' | 'video') {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.processFiles(Array.from(input.files), type);
    }
    input.value = ''; // Reset for re-selection
  }

  private processFiles(files: File[], filterType?: 'image' | 'video') {
    const newPreviews: PreviewFile[] = files
      .filter(file => {
        if (filterType === 'image') return file.type.startsWith('image/');
        if (filterType === 'video') return file.type.startsWith('video/');
        return file.type.startsWith('image/') || file.type.startsWith('video/');
      })
      .map(file => ({
        file,
        url: URL.createObjectURL(file),
        type: file.type.startsWith('image/') ? 'image' : 'video'
      }));

    this.previews.update(current => [...current, ...newPreviews]);
  }

  removeFile(preview: PreviewFile) {
    URL.revokeObjectURL(preview.url);
    this.previews.update(current => current.filter(p => p !== preview));
    // Reset active index if needed
    if (this.activeMediaIndex() >= this.previews().length) {
      this.activeMediaIndex.set(Math.max(0, this.previews().length - 1));
    }
  }

  // Media preview controls
  toggleMediaExpand() {
    this.mediaExpanded.update(v => !v);
  }

  setActiveMedia(index: number) {
    this.activeMediaIndex.set(index);
    if (!this.mediaExpanded()) {
      this.mediaExpanded.set(true);
    }
  }

  // Start analysis workflow
  async startAnalysis() {
    if (!this.hasUploads()) return;
    
    this.workflowStep.set('analyzing');
    const files = this.previews().map(p => p.file);
    
    // Add system message about analysis
    this.chatService.addSystemMessage('🔍 Đang phân tích tệp tin của bạn... Vui lòng đợi trong giây lát.');
    
    try {
      // Trigger analysis (real API call or mock)
      await this.festivalService.analyzeFiles(files);
      
      // Check for errors
      if (this.festivalService.status() === 'error') {
        this.chatService.addSystemMessage(
          `❌ ${this.festivalService.errorMessage() || 'Đã xảy ra lỗi khi phân tích video.'}`
        );
        this.snackBar.open('Lỗi phân tích video', 'Đóng', { duration: 5000 });
        this.workflowStep.set('upload');
        return;
      }
      
      // Move to questioning phase
      this.workflowStep.set('questioning');
      
      // Use question from backend if available, otherwise use default
      const backendQuestion = this.festivalService.question();
      const targetFeatures = this.festivalService.targetFeatures();
      
      if (backendQuestion) {
        // AI question from backend (Human-in-the-Loop)
        this.chatService.addAIMessage(
          `✨ Phân tích hoàn tất! ${backendQuestion}`
        );
        
        // Update suggestions based on target features
        if (targetFeatures.length > 0) {
          this.updateSuggestionsFromFeatures(targetFeatures);
        }
      } else {
        // Default question when using mock or finished immediately
        this.chatService.addSystemMessage(
          '✨ Phân tích hoàn tất! Tôi đã phát hiện một số đặc điểm văn hóa. Để tăng độ chính xác, bạn có thể cho tôi biết thêm:\n\n' +
          '• Bạn có thấy đèn hoa đăng hoặc đèn nước không?\n' +
          '• Có hoạt động đua ghe ngo không?\n' +
          '• Lễ hội diễn ra vào thời điểm nào (ngày/đêm)?\n' +
          '• Có nghi thức cúng bái nào không?'
        );
      }
      
      // Check if analysis is already finished
      if (this.festivalService.status() === 'finished') {
        this.workflowStep.set('completed');
        const winner = this.festivalService.winner();
        if (winner) {
          this.chatService.addSystemMessage(
            `🎉 **Kết quả:** Đây là lễ hội **${winner.name}** với độ tin cậy ${winner.confidence}%!`
          );
        }
      }
    } catch (error) {
      console.error('Analysis error:', error);
      this.chatService.addSystemMessage('❌ Đã xảy ra lỗi khi phân tích video. Vui lòng thử lại.');
      this.snackBar.open('Lỗi phân tích video', 'Đóng', { duration: 5000 });
      this.workflowStep.set('upload');
    }
  }

  /**
   * Update suggestion buttons based on target features from backend
   */
  private updateSuggestionsFromFeatures(features: string[]): void {
    const featureSuggestions: { icon: string; text: string }[] = [];
    
    for (const feature of features) {
      const lowerFeature = feature.toLowerCase();
      if (lowerFeature.includes('đèn') || lowerFeature.includes('hoa đăng')) {
        featureSuggestions.push({ icon: '🏮', text: `Có ${feature}` });
      } else if (lowerFeature.includes('ghe') || lowerFeature.includes('thuyền')) {
        featureSuggestions.push({ icon: '🚣', text: `Có ${feature}` });
      } else if (lowerFeature.includes('trăng') || lowerFeature.includes('cúng')) {
        featureSuggestions.push({ icon: '🌙', text: `Có ${feature}` });
      } else if (lowerFeature.includes('múa') || lowerFeature.includes('lân')) {
        featureSuggestions.push({ icon: '🎭', text: `Có ${feature}` });
      } else {
        featureSuggestions.push({ icon: '✅', text: `Có ${feature}` });
      }
    }
    
    // Add "Không thấy" options
    featureSuggestions.push({ icon: '❌', text: 'Không thấy những điều này' });
    
    if (featureSuggestions.length > 0) {
      this.suggestions = featureSuggestions;
    }
  }

  async sendMessage() {
    if (!this.canSendMessage()) return;

    const message = this.inputMessage();
    this.inputMessage.set('');
    this.isTyping.set(true);

    // Add user message to chat
    this.chatService.addUserMessage(message);
    
    // Update festival detection (sends to backend if request_id exists)
    await this.festivalService.updateFromChat(message);
    
    // Check if backend returned a new question
    const backendQuestion = this.festivalService.question();
    const status = this.festivalService.status();
    
    if (status === 'needs_clarification' && backendQuestion) {
      // Show backend's follow-up question
      this.chatService.addAIMessage(backendQuestion);
      
      // Update suggestions
      const targetFeatures = this.festivalService.targetFeatures();
      if (targetFeatures.length > 0) {
        this.updateSuggestionsFromFeatures(targetFeatures);
      }
    } else if (status === 'finished') {
      // Analysis complete
      this.workflowStep.set('completed');
      const winner = this.festivalService.winner();
      if (winner) {
        this.chatService.addSystemMessage(
          `🎉 **Xác nhận:** Đây là lễ hội **${winner.name}** với độ tin cậy ${winner.confidence}%!\n\n${winner.description}`
        );
      }
    } else {
      // Local AI response (fallback mode)
      await this.chatService.sendMessage('');  // Empty to trigger contextual response
    }
    
    this.isTyping.set(false);
    
    // Check if we should complete the workflow based on confidence
    const winner = this.festivalService.winner();
    if (winner && winner.confidence >= 85) {
      this.workflowStep.set('completed');
    }
  }

  useSuggestion(suggestion: { icon: string; text: string }) {
    this.inputMessage.set(suggestion.text);
    this.sendMessage();
  }

  formatTime(date: Date): string {
    return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
  }

  formatSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  async resetWorkflow() {
    // Clear backend sessions first
    await this.festivalService.deleteAllSessions();
    
    // Clear local state
    this.previews().forEach(p => URL.revokeObjectURL(p.url));
    this.previews.set([]);
    this.workflowStep.set('upload');
    this.chatService.clearHistory();
    this.festivalService.reset();
    this.mediaExpanded.set(false);
    this.activeMediaIndex.set(0);
  }

  private scrollToBottom() {
    if (this.messagesContainer) {
      const el = this.messagesContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }
}
