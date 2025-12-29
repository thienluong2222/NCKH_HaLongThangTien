import { Component } from '@angular/core';
import { HeaderComponent } from './components/header/header.component';
import { ChatBoxComponent } from './components/chat-box/chat-box.component';
import { ResultsCardComponent } from './components/results-card/results-card.component';
import { ExplanationTabsComponent } from './components/explanation-tabs/explanation-tabs.component';
import { FooterComponent } from './components/footer/footer.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    HeaderComponent,
    ChatBoxComponent,
    ResultsCardComponent,
    ExplanationTabsComponent,
    FooterComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {}
