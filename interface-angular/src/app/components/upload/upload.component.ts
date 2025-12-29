import { Component, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';

interface PreviewFile {
  file: File;
  url: string;
  type: 'image' | 'video';
}

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatProgressBarModule,
    MatChipsModule
  ],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.scss'
})
export class UploadComponent {
  filesUploaded = output<File[]>();
  
  isDraggingImage = signal(false);
  isDraggingVideo = signal(false);
  previews = signal<PreviewFile[]>([]);

  onDragOver(event: DragEvent, type: 'image' | 'video') {
    event.preventDefault();
    event.stopPropagation();
    if (type === 'image') {
      this.isDraggingImage.set(true);
    } else {
      this.isDraggingVideo.set(true);
    }
  }

  onDragLeave(type: 'image' | 'video') {
    if (type === 'image') {
      this.isDraggingImage.set(false);
    } else {
      this.isDraggingVideo.set(false);
    }
  }

  onDrop(event: DragEvent, type: 'image' | 'video') {
    event.preventDefault();
    event.stopPropagation();
    this.isDraggingImage.set(false);
    this.isDraggingVideo.set(false);

    const files = event.dataTransfer?.files;
    if (files) {
      this.processFiles(Array.from(files), type);
    }
  }

  onFileSelect(event: Event, type: 'image' | 'video') {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.processFiles(Array.from(input.files), type);
    }
  }

  private processFiles(files: File[], type: 'image' | 'video') {
    const newPreviews: PreviewFile[] = files
      .filter(file => type === 'image' ? file.type.startsWith('image/') : file.type.startsWith('video/'))
      .map(file => ({
        file,
        url: URL.createObjectURL(file),
        type
      }));

    this.previews.update(current => [...current, ...newPreviews]);
  }

  removeFile(preview: PreviewFile) {
    URL.revokeObjectURL(preview.url);
    this.previews.update(current => current.filter(p => p !== preview));
  }

  clearAll() {
    this.previews().forEach(p => URL.revokeObjectURL(p.url));
    this.previews.set([]);
  }

  startAnalysis() {
    const files = this.previews().map(p => p.file);
    this.filesUploaded.emit(files);
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
}
