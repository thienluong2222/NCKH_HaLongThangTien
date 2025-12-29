import { Injectable, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';

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
}

@Injectable({
  providedIn: 'root'
})
export class FestivalDetectionService {
  private readonly API_URL = 'http://localhost:8000/api'; // Backend API
  
  // Festival color mapping
  private festivalColors: Record<string, string> = {
    'Ok Om Bok': '#D4A853',
    'Chol Chnam Thmay': '#8B2323',
    'Tết Nguyên Đán': '#E53935',
    'Lễ hội Đền Hùng': '#1A5F5F',
    'Lễ hội Chùa Hương': '#7B1FA2'
  };
  
  // Signals for reactive state
  private state = signal<DetectionState>({
    isLoading: false,
    results: [],
    winner: null,
    uploadedFiles: [],
    topThree: []
  });

  // Computed values
  readonly isLoading = computed(() => this.state().isLoading);
  readonly results = computed(() => this.state().results);
  readonly winner = computed(() => this.state().winner);
  readonly uploadedFiles = computed(() => this.state().uploadedFiles);
  readonly hasResults = computed(() => this.state().results.length > 0);
  readonly topThree = computed(() => this.state().topThree);

  constructor(private http: HttpClient) {}

  analyzeFiles(files: File[]): Promise<FestivalResult[]> {
    this.state.update(s => ({ ...s, isLoading: true, uploadedFiles: files }));

    // Simulate API call (replace with real backend call)
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockResults: FestivalResult[] = [
          { 
            name: 'Ok Om Bok', 
            confidence: 78, 
            description: 'Lễ hội cúng trăng của đồng bào Khmer Nam Bộ',
            color: this.festivalColors['Ok Om Bok']
          },
          { 
            name: 'Chol Chnam Thmay', 
            confidence: 52, 
            description: 'Tết cổ truyền của người Khmer',
            color: this.festivalColors['Chol Chnam Thmay']
          },
          { 
            name: 'Tết Nguyên Đán', 
            confidence: 35, 
            description: 'Tết cổ truyền của dân tộc Việt Nam',
            color: this.festivalColors['Tết Nguyên Đán']
          }
        ];

        const topThree = mockResults.slice(0, 3);

        this.state.update(s => ({
          ...s,
          isLoading: false,
          results: mockResults,
          winner: mockResults[0],
          topThree
        }));

        resolve(mockResults);
      }, 2500);
    });
  }

  updateFromChat(message: string) {
    const lowerMsg = message.toLowerCase();
    let boostAmount = 0;
    
    // Detect keywords and boost confidence
    if (lowerMsg.includes('đèn hoa đăng') || lowerMsg.includes('đèn nước')) {
      boostAmount = 5;
    } else if (lowerMsg.includes('ghe ngo') || lowerMsg.includes('đua thuyền')) {
      boostAmount = 6;
    } else if (lowerMsg.includes('cúng trăng') || lowerMsg.includes('rằm')) {
      boostAmount = 4;
    } else if (lowerMsg.includes('múa') || lowerMsg.includes('lân')) {
      boostAmount = 2;
    } else {
      boostAmount = 1;
    }
    
    if (boostAmount > 0) {
      this.state.update(s => {
        const updatedResults = s.results.map((r, i) => 
          i === 0 ? { ...r, confidence: Math.min(r.confidence + boostAmount, 99) } : r
        );
        
        return {
          ...s,
          winner: s.winner ? { ...s.winner, confidence: Math.min(s.winner.confidence + boostAmount, 99) } : s.winner,
          results: updatedResults,
          topThree: updatedResults.slice(0, 3)
        };
      });
    }
  }

  reset() {
    this.state.set({
      isLoading: false,
      results: [],
      winner: null,
      uploadedFiles: [],
      topThree: []
    });
  }
}
