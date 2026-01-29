import { Component, OnInit, AfterViewInit, ElementRef, ViewChild, effect, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { FestivalDetectionService } from '../../services/festival-detection.service';
import { FestivalConstraints } from '../../models/api.models';

declare const Plotly: any;
declare const d3: any;

// Graph node and link interfaces for D3
interface D3Node {
  id: string;
  label: string;
  type: 'festival' | 'feature';
  confidence?: number;
  color: string;
  radius: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface D3Link {
  source: string | D3Node;
  target: string | D3Node;
  satisfied: boolean;
  weight: number;
}

@Component({
  selector: 'app-explanation-tabs',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatTabsModule,
    MatIconModule,
    MatListModule,
    MatChipsModule,
    MatTooltipModule
  ],
  templateUrl: './explanation-tabs.component.html',
  styleUrl: './explanation-tabs.component.scss'
})
export class ExplanationTabsComponent implements AfterViewInit, OnDestroy {
  @ViewChild('chartContainer') chartContainer!: ElementRef;
  @ViewChild('graphContainer') graphContainer!: ElementRef;

  // Graph data
  graphNodes: D3Node[] = [];
  graphLinks: D3Link[] = [];
  
  // D3 simulation
  private simulation: any = null;
  private svg: any = null;
  private d3Loaded = false;
  
  // Festival colors mapping
  private festivalColors: Record<string, string> = {
    'Ok Om Bok': '#D4A853',
    'Chol Chnam Thmay': '#8B2323',
    'Tết Nguyên Đán': '#E53935',
    'Lễ hội Đền Hùng': '#1A5F5F',
    'Lễ hội Chùa Hương': '#7B1FA2',
    'Lễ hội Gióng': '#00695C',
    'Lễ hội Tháp Bà Ponagar': '#E65100'
  };
  
  chatRules: { keyword: string; impact: string; boost: number }[] = [];

  constructor(public festivalService: FestivalDetectionService) {
    // Watch for changes in results and constraints data
    effect(() => {
      const results = this.festivalService.results();
      const constraints = this.festivalService.constraintsData();
      
      if (results && results.length > 0) {
        console.log('Results updated, re-rendering chart');
        setTimeout(() => this.renderChart(), 100);
      }
      
      if (constraints && constraints.length > 0) {
        console.log('Constraints updated, building graph');
        this.buildGraphData(constraints);
        setTimeout(() => this.renderD3Graph(), 150);
      }
    });
  }

  ngAfterViewInit() {
    this.loadLibraries();
  }

  ngOnDestroy() {
    // Cleanup
    if (this.simulation) {
      this.simulation.stop();
    }
    if (typeof Plotly !== 'undefined' && this.chartContainer?.nativeElement) {
      Plotly.purge(this.chartContainer.nativeElement);
    }
  }

  private async loadLibraries() {
    // Load Plotly
    if (typeof Plotly === 'undefined') {
      const plotlyScript = document.createElement('script');
      plotlyScript.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
      plotlyScript.onload = () => this.renderChart();
      document.head.appendChild(plotlyScript);
    } else {
      this.renderChart();
    }
    
    // Load D3.js
    if (typeof d3 === 'undefined') {
      const d3Script = document.createElement('script');
      d3Script.src = 'https://d3js.org/d3.v7.min.js';
      d3Script.onload = () => {
        this.d3Loaded = true;
        this.renderD3Graph();
      };
      document.head.appendChild(d3Script);
    } else {
      this.d3Loaded = true;
      this.renderD3Graph();
    }
  }

  /**
   * Build graph data from constraints
   */
  private buildGraphData(constraints: FestivalConstraints[]): void {
    this.graphNodes = [];
    this.graphLinks = [];
    
    const featureSet = new Set<string>();
    const featureWeights: Record<string, number> = {};
    const featureSatisfied: Record<string, boolean> = {};
    
    // Process each festival
    constraints.forEach((fc, index) => {
      const festivalColor = this.festivalColors[fc.festival] || this.getDefaultColor(index);
      this.graphNodes.push({
        id: `festival_${index}`,
        label: fc.festival,
        type: 'festival',
        confidence: fc.confidence,
        color: festivalColor,
        radius: 25 + (fc.confidence * 15)
      });
      
      // Process satisfied constraints
      fc.satisfied.forEach(constraint => {
        constraint.params.forEach(param => {
          const featureId = this.normalizeFeatureId(param);
          featureSet.add(param);
          featureWeights[param] = Math.max(featureWeights[param] || 0, constraint.weight);
          featureSatisfied[param] = true;
          
          this.graphLinks.push({
            source: `festival_${index}`,
            target: `feature_${featureId}`,
            satisfied: true,
            weight: constraint.weight
          });
        });
      });
      
      // Process unsatisfied constraints
      fc.unsatisfied.forEach(constraint => {
        constraint.params.forEach(param => {
          const featureId = this.normalizeFeatureId(param);
          featureSet.add(param);
          featureWeights[param] = Math.max(featureWeights[param] || 0, constraint.weight);
          if (featureSatisfied[param] === undefined) {
            featureSatisfied[param] = false;
          }
          
          this.graphLinks.push({
            source: `festival_${index}`,
            target: `feature_${featureId}`,
            satisfied: false,
            weight: constraint.weight
          });
        });
      });
    });
    
    // Add feature nodes
    featureSet.forEach(feature => {
      const featureId = this.normalizeFeatureId(feature);
      const isSatisfied = featureSatisfied[feature] || false;
      this.graphNodes.push({
        id: `feature_${featureId}`,
        label: feature,
        type: 'feature',
        color: isSatisfied ? '#28a745' : '#dc3545',
        radius: 12 + (featureWeights[feature] * 2)
      });
    });
  }

  private normalizeFeatureId(feature: string): string {
    return feature.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
  }

  private getDefaultColor(index: number): string {
    const colors = ['#D4A853', '#1A5F5F', '#E53935', '#7B1FA2', '#00695C'];
    return colors[index % colors.length];
  }

  /**
   * Render interactive D3 force-directed graph
   */
  private renderD3Graph(): void {
    if (!this.d3Loaded || !this.graphContainer?.nativeElement || this.graphNodes.length === 0) return;
    
    const container = this.graphContainer.nativeElement;
    const width = container.clientWidth || 600;
    const height = 420;
    
    // Clear previous
    d3.select(container).selectAll('*').remove();
    if (this.simulation) this.simulation.stop();
    
    // Create SVG with zoom
    this.svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('class', 'd3-graph');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event: any) => {
        g.attr('transform', event.transform);
      });
    
    this.svg.call(zoom);
    
    // Main group for zoom/pan
    const g = this.svg.append('g');
    
    // Defs for filters and gradients
    const defs = this.svg.append('defs');
    
    // Glow filter for nodes
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');
    
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');
    
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    
    // Arrow marker for links
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#999');
    
    // Copy data for simulation
    const nodes: D3Node[] = this.graphNodes.map(d => ({...d}));
    const links: D3Link[] = this.graphLinks.map(d => ({...d}));
    
    // Force simulation
    this.simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id((d: D3Node) => d.id)
        .distance(100)
        .strength((d: D3Link) => d.satisfied ? 0.3 : 0.1))
      .force('charge', d3.forceManyBody()
        .strength((d: D3Node) => d.type === 'festival' ? -400 : -150))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius((d: D3Node) => d.radius + 10));
    
    // Links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', (d: D3Link) => d.satisfied ? '#28a745' : '#dc3545')
      .attr('stroke-opacity', 0.4)
      .attr('stroke-width', (d: D3Link) => Math.max(1, d.weight * 1.5))
      .attr('stroke-dasharray', (d: D3Link) => d.satisfied ? 'none' : '5,5');
    
    // Node groups
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', (d: D3Node) => `node node-${d.type}`)
      .call(this.drag(this.simulation));
    
    // Festival nodes (circles)
    node.filter((d: D3Node) => d.type === 'festival')
      .append('circle')
      .attr('r', (d: D3Node) => d.radius)
      .attr('fill', (d: D3Node) => d.color)
      .attr('stroke', 'white')
      .attr('stroke-width', 3)
      .attr('class', 'node-circle festival-circle');
    
    // Feature nodes - satisfied (diamonds)
    node.filter((d: D3Node) => d.type === 'feature' && d.color === '#28a745')
      .append('path')
      .attr('d', (d: D3Node) => {
        const r = d.radius;
        return `M0,${-r} L${r},0 L0,${r} L${-r},0 Z`;
      })
      .attr('fill', '#28a745')
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .attr('class', 'node-shape feature-satisfied');
    
    // Feature nodes - unsatisfied (squares)
    node.filter((d: D3Node) => d.type === 'feature' && d.color === '#dc3545')
      .append('rect')
      .attr('x', (d: D3Node) => -d.radius)
      .attr('y', (d: D3Node) => -d.radius)
      .attr('width', (d: D3Node) => d.radius * 2)
      .attr('height', (d: D3Node) => d.radius * 2)
      .attr('rx', 3)
      .attr('fill', '#dc3545')
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .attr('class', 'node-shape feature-unsatisfied');
    
    // Labels
    node.append('text')
      .attr('class', 'node-label')
      .attr('dy', (d: D3Node) => d.type === 'festival' ? d.radius + 15 : d.radius + 12)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'K2D, sans-serif')
      .attr('font-size', (d: D3Node) => d.type === 'festival' ? '11px' : '9px')
      .attr('font-weight', (d: D3Node) => d.type === 'festival' ? '600' : '400')
      .attr('fill', (d: D3Node) => d.type === 'festival' ? '#333' : d.color)
      .text((d: D3Node) => d.label.length > 15 ? d.label.substring(0, 15) + '...' : d.label);
    
    // Confidence badge for festivals
    node.filter((d: D3Node) => d.type === 'festival')
      .append('text')
      .attr('class', 'confidence-badge')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'K2D, sans-serif')
      .attr('font-size', '10px')
      .attr('font-weight', '700')
      .attr('fill', 'white')
      .text((d: D3Node) => `${Math.round((d.confidence || 0) * 100)}%`);
    
    // Hover interactions
    node.on('mouseenter', (event: any, d: D3Node) => {
      // Highlight connected links
      link.attr('stroke-opacity', (l: D3Link) => {
        const source = typeof l.source === 'object' ? l.source.id : l.source;
        const target = typeof l.target === 'object' ? l.target.id : l.target;
        return source === d.id || target === d.id ? 1 : 0.1;
      }).attr('stroke-width', (l: D3Link) => {
        const source = typeof l.source === 'object' ? l.source.id : l.source;
        const target = typeof l.target === 'object' ? l.target.id : l.target;
        return source === d.id || target === d.id ? l.weight * 2.5 : l.weight * 1.5;
      });
      
      // Highlight connected nodes
      const connectedIds = new Set<string>();
      connectedIds.add(d.id);
      links.forEach(l => {
        const source = typeof l.source === 'object' ? l.source.id : l.source;
        const target = typeof l.target === 'object' ? l.target.id : l.target;
        if (source === d.id) connectedIds.add(target);
        if (target === d.id) connectedIds.add(source);
      });
      
      node.attr('opacity', (n: D3Node) => connectedIds.has(n.id) ? 1 : 0.3);
      
      // Enlarge hovered node
      d3.select(event.currentTarget)
        .select('circle, path, rect')
        .transition()
        .duration(200)
        .attr('filter', 'url(#glow)')
        .attr('transform', 'scale(1.2)');
    })
    .on('mouseleave', (event: any) => {
      // Reset links
      link.attr('stroke-opacity', 0.4)
        .attr('stroke-width', (d: D3Link) => Math.max(1, d.weight * 1.5));
      
      // Reset nodes
      node.attr('opacity', 1);
      
      // Reset hovered node
      d3.select(event.currentTarget)
        .select('circle, path, rect')
        .transition()
        .duration(200)
        .attr('filter', null)
        .attr('transform', 'scale(1)');
    });
    
    // Click to lock/unlock node position
    node.on('click', (event: any, d: D3Node) => {
      event.stopPropagation();
      if (d.fx !== null && d.fx !== undefined) {
        d.fx = null;
        d.fy = null;
        d3.select(event.currentTarget).classed('locked', false);
      } else {
        d.fx = d.x;
        d.fy = d.y;
        d3.select(event.currentTarget).classed('locked', true);
      }
    });
    
    // Click on background to reset zoom
    this.svg.on('dblclick.zoom', null);
    this.svg.on('dblclick', () => {
      this.svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
    });
    
    // Update positions on tick
    this.simulation.on('tick', () => {
      link
        .attr('x1', (d: D3Link) => (d.source as D3Node).x!)
        .attr('y1', (d: D3Link) => (d.source as D3Node).y!)
        .attr('x2', (d: D3Link) => (d.target as D3Node).x!)
        .attr('y2', (d: D3Link) => (d.target as D3Node).y!);
      
      node.attr('transform', (d: D3Node) => `translate(${d.x},${d.y})`);
    });
    
    // Initial animation - nodes fly in
    node.attr('opacity', 0)
      .transition()
      .duration(800)
      .delay((d: D3Node, i: number) => i * 50)
      .attr('opacity', 1);
    
    link.attr('stroke-opacity', 0)
      .transition()
      .duration(500)
      .delay(300)
      .attr('stroke-opacity', 0.4);
  }
  
  /**
   * D3 drag behavior
   */
  private drag(simulation: any) {
    function dragstarted(event: any, d: D3Node) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event: any, d: D3Node) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event: any, d: D3Node) {
      if (!event.active) simulation.alphaTarget(0);
      // Keep position fixed after drag (user can click to release)
    }
    
    return d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended);
  }

  /**
   * Get statistics for display
   */
  get graphStats() {
    const festivalCount = this.graphNodes.filter(n => n.type === 'festival').length;
    const featureCount = this.graphNodes.filter(n => n.type === 'feature').length;
    const satisfiedCount = this.graphLinks.filter(l => l.satisfied).length;
    const unsatisfiedCount = this.graphLinks.filter(l => !l.satisfied).length;
    
    return {
      festivals: festivalCount,
      features: featureCount,
      satisfiedLinks: satisfiedCount,
      unsatisfiedLinks: unsatisfiedCount,
      totalLinks: satisfiedCount + unsatisfiedCount
    };
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
        line: { color: 'rgba(0,0,0,0.1)', width: 2 }
      },
      text: results.map(r => `${r.confidence}%`),
      textposition: 'auto',
      insidetextanchor: 'middle',
      textfont: { family: 'K2D', size: 14, color: '#fff', weight: 'bold' },
      hovertemplate: '<b>%{x}</b><br>Xác suất: %{y}%<extra></extra>'
    }];

    const layout = {
      title: {
        text: 'Phân bố xác suất lễ hội',
        font: { family: 'K2D', size: 18, color: '#333' }
      },
      xaxis: { tickfont: { family: 'K2D', size: 12 } },
      yaxis: {
        title: 'Xác suất (%)',
        range: [0, 100],
        tickfont: { family: 'K2D', size: 12 }
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      margin: { t: 60, b: 60, l: 60, r: 30 },
      hoverlabel: { bgcolor: 'white', font: { family: 'K2D' } }
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(this.chartContainer.nativeElement, data, layout, config);
  }
}
