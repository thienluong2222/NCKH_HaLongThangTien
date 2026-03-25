import { Injectable, signal } from "@angular/core";
import { TranslateService } from "@ngx-translate/core";

export type AppLanguage = "vi" | "en";

@Injectable({
  providedIn: "root",
})
export class TranslationService {
  private readonly STORAGE_KEY = "icheritage_lang";

  currentLang = signal<AppLanguage>("vi");

  constructor(private translate: TranslateService) {
    // Setup available languages
    this.translate.addLangs(["vi", "en"]);
    this.translate.setDefaultLang("vi");

    // Restore saved language or use default
    const saved = localStorage.getItem(this.STORAGE_KEY) as AppLanguage | null;
    const lang = saved && ["vi", "en"].includes(saved) ? saved : "vi";

    this.switchLanguage(lang);
  }

  switchLanguage(lang: AppLanguage): void {
    this.translate.use(lang);
    this.currentLang.set(lang);
    localStorage.setItem(this.STORAGE_KEY, lang);
  }

  toggleLanguage(): void {
    const next: AppLanguage = this.currentLang() === "vi" ? "en" : "vi";
    this.switchLanguage(next);
  }
}
