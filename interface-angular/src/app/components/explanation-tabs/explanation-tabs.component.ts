import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatChipsModule } from '@angular/material/chips';
import { FestivalDetectionService } from '../../services/festival-detection.service';

declare const Plotly: any;

@Component({
  selector: 'app-explanation-tabs',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatTabsModule,
    MatIconModule,
    MatListModule,
    MatChipsModule
  ],
  templateUrl: './explanation-tabs.component.html',
  styleUrl: './explanation-tabs.component.scss'
})
export class ExplanationTabsComponent implements AfterViewInit {
  @ViewChild('chartContainer') chartContainer!: ElementRef;
  
  visualRules = [
    { name: 'Đèn gió (Sky Lanterns)', description: 'Traditional floating lanterns', satisfied: true },
    { name: 'Ghe ngo (Racing Boats)', description: 'Khmer traditional boats', satisfied: true },
    { name: 'Đèn nước (Water Lanterns)', description: 'Floating water lamps', satisfied: false },
    { name: 'Moon offerings', description: 'Traditional moon worship items', satisfied: true }
  ];
  
  chatRules: { keyword: string; impact: string; boost: number }[] = [];

  constructor(public festivalService: FestivalDetectionService) {
    effect(() => {
      if (this.festivalService.hasResults()) {
        setTimeout(() => this.renderChart(), 100);
      }
    });
  }

  ngAfterViewInit() {
    this.loadPlotly();
  }

  private async loadPlotly() {
    if (typeof Plotly === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
      script.onload = () => this.renderChart();
      document.head.appendChild(script);
    } else {
      this.renderChart();
    }
  }

  private renderChart() {
    if (!this.chartContainer?.nativeElement) return;
    
    const results = this.festivalService.results();
    
    const data = [{
      x: results.map(r => r.name),
      y: results.map(r => r.confidence),
      type: 'bar',
      marker: {
        color: results.map((_, i) => 
          i === 0 ? '#ff6b6b' : i === 1 ? '#4ecdc4' : '#45b7d1'
        ),
        line: {
          color: 'rgba(0,0,0,0.1)',
          width: 2
        }
      },
      text: results.map(r => `${r.confidence}%`),
      textposition: 'outside',
      hovertemplate: '<b>%{x}</b><br>Confidence: %{y}%<extra></extra>'
    }];

    const layout = {
      title: {
        text: 'Festival Probability Distribution',
        font: { family: 'K2D', size: 18, color: '#333' }
      },
      xaxis: {
        tickfont: { family: 'K2D', size: 12 }
      },
      yaxis: {
        title: 'Confidence (%)',
        range: [0, 100],
        tickfont: { family: 'K2D', size: 12 }
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      margin: { t: 60, b: 60, l: 60, r: 30 },
      hoverlabel: {
        bgcolor: 'white',
        font: { family: 'K2D' }
      }
    };

    const config = {
      responsive: true,
      displayModeBar: false
    };

    Plotly.newPlot(this.chartContainer.nativeElement, data, layout, config);
  }
}
