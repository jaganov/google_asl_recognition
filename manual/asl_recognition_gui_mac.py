"""
GUI версия ASL Recognition для macOS
Использует tkinter для создания удобного пользовательского интерфейса
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
from pathlib import Path
import platform
import sys
import os

# Импортируем основной класс распознавания
from step5_live_recognition_mac import MacLiveASLRecognition

class ASLRecognitionGUI:
    """GUI приложение для распознавания ASL жестов на macOS"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASL Recognition for macOS")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Проверяем платформу
        if platform.system() != "Darwin":
            messagebox.showwarning("Предупреждение", 
                                 "Это приложение оптимизировано для macOS")
        
        # Переменные состояния
        self.recognizer = None
        self.recognition_thread = None
        self.is_running = False
        self.message_queue = queue.Queue()
        
        # Настройки
        self.model_path = tk.StringVar()
        self.camera_id = tk.IntVar(value=0)
        self.confidence_threshold = tk.DoubleVar(value=0.8)
        self.save_screenshots = tk.BooleanVar(value=True)
        self.use_only_face = tk.BooleanVar(value=False)
        
        # Создаем интерфейс
        self.create_widgets()
        self.update_status("Готов к работе")
        
        # Автоматически ищем модель
        self.auto_find_model()
        
        # Запускаем обработку сообщений
        self.process_messages()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="🍎 ASL Recognition для macOS", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Настройки модели
        model_frame = ttk.LabelFrame(main_frame, text="Настройки модели", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Путь к модели:").grid(row=0, column=0, sticky=tk.W, pady=2)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=50)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=2)
        ttk.Button(model_frame, text="Обзор...", 
                  command=self.browse_model).grid(row=0, column=2, pady=2)
        
        # Настройки камеры
        camera_frame = ttk.LabelFrame(main_frame, text="Настройки камеры", padding="10")
        camera_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(camera_frame, text="ID камеры:").grid(row=0, column=0, sticky=tk.W, pady=2)
        camera_spinbox = ttk.Spinbox(camera_frame, from_=0, to=10, width=10, 
                                    textvariable=self.camera_id)
        camera_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        ttk.Label(camera_frame, text="Порог уверенности:").grid(row=1, column=0, sticky=tk.W, pady=2)
        confidence_scale = ttk.Scale(camera_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_threshold, 
                                   orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        confidence_label = ttk.Label(camera_frame, text="80%")
        confidence_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        
        # Обновляем отображение процентов
        def update_confidence_label(*args):
            confidence_label.config(text=f"{int(self.confidence_threshold.get() * 100)}%")
        self.confidence_threshold.trace('w', update_confidence_label)
        
        # Дополнительные настройки
        options_frame = ttk.LabelFrame(main_frame, text="Дополнительные настройки", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="Сохранять скриншоты", 
                       variable=self.save_screenshots).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Использовать только лицо", 
                       variable=self.use_only_face).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Кнопки управления
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(control_frame, text="🎬 Начать распознавание", 
                                      command=self.start_recognition, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="⏹ Остановить", 
                                     command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="📁 Открыть папку скриншотов", 
                  command=self.open_screenshots_folder).pack(side=tk.LEFT)
        
        # Область статуса
        status_frame = ttk.LabelFrame(main_frame, text="Статус", padding="10")
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="Готов к работе", 
                                     foreground="green", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Текстовое поле для логов
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Информация о системе
        info_frame = ttk.LabelFrame(main_frame, text="Информация о системе", padding="10")
        info_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        system_info = f"Система: {platform.system()} {platform.release()}"
        if platform.system() == "Darwin":
            system_info += f"\nПроцессор: {platform.processor() or 'Unknown'}"
        
        ttk.Label(info_frame, text=system_info, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W)
        
        # Настройка растягивания для главного фрейма
        main_frame.rowconfigure(5, weight=1)
    
    def auto_find_model(self):
        """Автоматически находит последнюю модель"""
        try:
            # Пути для поиска
            search_paths = [
                Path("models"),
                Path("manual/models"),
                Path.cwd() / "models",
                Path.cwd() / "manual" / "models"
            ]
            
            latest_model = None
            latest_time = 0
            
            for search_path in search_paths:
                if search_path.exists():
                    for model_file in search_path.glob("*.pth"):
                        try:
                            file_time = model_file.stat().st_mtime
                            if file_time > latest_time:
                                latest_time = file_time
                                latest_model = str(model_file)
                        except Exception:
                            continue
            
            if latest_model:
                self.model_path.set(latest_model)
                self.log_message(f"✅ Найдена модель: {Path(latest_model).name}")
            else:
                self.log_message("⚠️ Модель не найдена. Выберите модель вручную.")
        except Exception as e:
            self.log_message(f"⚠️ Ошибка поиска модели: {e}")
    
    def browse_model(self):
        """Открывает диалог выбора модели"""
        filename = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("PyTorch модели", "*.pth"), ("Все файлы", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
            self.log_message(f"✅ Выбрана модель: {Path(filename).name}")
    
    def start_recognition(self):
        """Запускает распознавание в отдельном потоке"""
        if self.is_running:
            return
        
        if not self.model_path.get():
            messagebox.showerror("Ошибка", "Выберите файл модели")
            return
        
        if not Path(self.model_path.get()).exists():
            messagebox.showerror("Ошибка", "Файл модели не найден")
            return
        
        try:
            # Создаем распознаватель
            self.recognizer = MacLiveASLRecognition(
                model_path=self.model_path.get(),
                camera_id=self.camera_id.get(),
                target_frames=16,
                use_only_face=self.use_only_face.get(),
                confidence_threshold=self.confidence_threshold.get(),
                save_screenshots=self.save_screenshots.get()
            )
            
            # Запускаем в отдельном потоке
            self.is_running = True
            self.recognition_thread = threading.Thread(target=self.run_recognition, daemon=True)
            self.recognition_thread.start()
            
            # Обновляем интерфейс
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_status("Распознавание запущено", "blue")
            self.log_message("🎬 Распознавание запущено")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить распознавание:\n{e}")
            self.log_message(f"❌ Ошибка запуска: {e}")
    
    def run_recognition(self):
        """Запускает распознавание (выполняется в отдельном потоке)"""
        try:
            if self.recognizer:
                # Перенаправляем вывод в очередь сообщений
                self.message_queue.put(("status", "Инициализация камеры...", "blue"))
                self.recognizer.start_recognition()
        except Exception as e:
            self.message_queue.put(("error", f"Ошибка распознавания: {e}"))
        finally:
            self.message_queue.put(("finished", None))
    
    def stop_recognition(self):
        """Останавливает распознавание"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Обновляем интерфейс
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Остановлено", "orange")
        self.log_message("⏹ Распознавание остановлено")
        
        # Закрываем окна OpenCV (если есть)
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
    
    def open_screenshots_folder(self):
        """Открывает папку со скриншотами"""
        screenshots_dir = Path.home() / "ASL_Screenshots"
        
        if not screenshots_dir.exists():
            messagebox.showinfo("Информация", "Папка скриншотов еще не создана")
            return
        
        # Открываем папку в Finder (macOS)
        if platform.system() == "Darwin":
            os.system(f'open "{screenshots_dir}"')
        else:
            messagebox.showinfo("Путь к скриншотам", str(screenshots_dir))
    
    def update_status(self, message, color="green"):
        """Обновляет статус"""
        self.status_label.config(text=message, foreground=color)
    
    def log_message(self, message):
        """Добавляет сообщение в лог"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def process_messages(self):
        """Обрабатывает сообщения из очереди"""
        try:
            while True:
                try:
                    msg_type, content, *args = self.message_queue.get_nowait()
                    
                    if msg_type == "status":
                        color = args[0] if args else "blue"
                        self.update_status(content, color)
                    elif msg_type == "log":
                        self.log_message(content)
                    elif msg_type == "error":
                        self.update_status("Ошибка", "red")
                        self.log_message(f"❌ {content}")
                        messagebox.showerror("Ошибка", content)
                    elif msg_type == "finished":
                        self.stop_recognition()
                        
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Ошибка обработки сообщений: {e}")
        
        # Планируем следующую проверку
        self.root.after(100, self.process_messages)
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        if self.is_running:
            if messagebox.askokcancel("Выход", "Остановить распознавание и выйти?"):
                self.stop_recognition()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Запускает GUI приложение"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Центрируем окно
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Показываем приветственное сообщение
        self.log_message("🍎 Добро пожаловать в ASL Recognition для macOS!")
        self.log_message("1. Выберите модель (или используйте автоматически найденную)")
        self.log_message("2. Настройте параметры")
        self.log_message("3. Нажмите 'Начать распознавание'")
        self.log_message("4. Для выхода нажмите 'q' в окне камеры или закройте приложение")
        
        self.root.mainloop()

if __name__ == "__main__":
    print("🍎 Запуск GUI приложения ASL Recognition для macOS...")
    
    try:
        app = ASLRecognitionGUI()
        app.run()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        # Показываем сообщение об ошибке
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Критическая ошибка", 
                               f"Не удалось запустить приложение:\n{e}")
        except:
            pass 