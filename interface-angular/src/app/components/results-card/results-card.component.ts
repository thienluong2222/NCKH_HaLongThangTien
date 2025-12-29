import { Component, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { FestivalDetectionService } from '../../services/festival-detection.service';
import { trigger, transition, style, animate, state } from '@angular/animations';

@Component({
  selector: 'app-results-card',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatButtonModule
  ],
  animations: [
    trigger('cardState', [
      state('hidden', style({ opacity: 0, transform: 'scale(0.8) translateY(20px)' })),
      state('visible', style({ opacity: 1, transform: 'scale(1) translateY(0)' })),
      transition('hidden => visible', animate('500ms cubic-bezier(0.35, 0, 0.25, 1)')),
    ]),
    trigger('confetti', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-100%)' }),
        animate('1s ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ])
  ],
  templateUrl: './results-card.component.html',
  styleUrl: './results-card.component.scss'
})
export class ResultsCardComponent {
  constructor(public festivalService: FestivalDetectionService) {}

  getCircumference(radius: number = 42): string {
    return `${2 * Math.PI * radius}`;
  }

  getOffset(radius: number = 42): string {
    const confidence = this.festivalService.winner()?.confidence || 0;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (confidence / 100) * circumference;
    return `${offset}`;
  }
}
