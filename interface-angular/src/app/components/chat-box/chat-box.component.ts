import { Component, signal, ViewChild, ElementRef, AfterViewChecked, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
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
    MatTooltipModule
  ],
  templateUrl: './chat-box.component.html',
  styleUrl: './chat-box.component.scss'
})
export class ChatBoxComponent implements AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;
  @ViewChild('imageInput') private imageInput!: ElementRef<HTMLInputElement>;
  @ViewChild('videoInput') private videoInput!: ElementRef<HTMLInputElement>;
  
  inputMessage = signal('');
  isTyping = signal(false);
  isDragging = signal(false);
  previews = signal<PreviewFile[]>([]);
  workflowStep = signal<WorkflowStep>('upload');
  
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
    public festivalService: FestivalDetectionService
  ) {}

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
  }

  // Start analysis workflow
  async startAnalysis() {
    if (!this.hasUploads()) return;
    
    this.workflowStep.set('analyzing');
    const files = this.previews().map(p => p.file);
    
    // Add system message about analysis
    this.chatService.addSystemMessage('🔍 Đang phân tích tệp tin của bạn... Vui lòng đợi trong giây lát.');
    
    // Trigger analysis
    await this.festivalService.analyzeFiles(files);
    
    // Move to questioning phase
    this.workflowStep.set('questioning');
    
    // AI asks clarifying questions
    this.chatService.addSystemMessage(
      '✨ Phân tích hoàn tất! Tôi đã phát hiện một số đặc điểm văn hóa. Để tăng độ chính xác, bạn có thể cho tôi biết thêm:\n\n' +
      '• Bạn có thấy đèn hoa đăng hoặc đèn nước không?\n' +
      '• Có hoạt động đua ghe ngo không?\n' +
      '• Lễ hội diễn ra vào thời điểm nào (ngày/đêm)?\n' +
      '• Có nghi thức cúng bái nào không?'
    );
  }

  async sendMessage() {
    if (!this.canSendMessage()) return;

    const message = this.inputMessage();
    this.inputMessage.set('');
    this.isTyping.set(true);

    await this.chatService.sendMessage(message);
    this.festivalService.updateFromChat(message);
    
    this.isTyping.set(false);
    
    // Check if we should complete the workflow
    if (this.chatService.messages().length > 5) {
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

  resetWorkflow() {
    this.previews().forEach(p => URL.revokeObjectURL(p.url));
    this.previews.set([]);
    this.workflowStep.set('upload');
    this.chatService.clearHistory();
    this.festivalService.reset();
  }

  private scrollToBottom() {
    if (this.messagesContainer) {
      const el = this.messagesContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }
}
