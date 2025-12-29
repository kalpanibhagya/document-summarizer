from Imports import *
from CSVParser import CSVParser
from OllamaServer import OllamaServer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CSVSummarizer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("CSV Data Summarizer with RAG")
        self.root.geometry("1000x800")
        self.server = OllamaServer()
        self.conversation = []
        self.current_model = 'gemma3:1b'
        self.embedding_model = 'nomic-embed-text' 
        self.csv_data = None
        self.csv_filename = None
        self.csv_summary = None
        self.embeddings_ready = False
        
        self.setup_ui()
        self.start_server()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):        
        # Control frame
        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        # Model selector
        ctk.CTkLabel(control_frame, text="Model:").pack(side="left", padx=10)
        
        self.model_var = ctk.StringVar(value=self.current_model)
        model_menu = ctk.CTkOptionMenu(
            control_frame,
            values=['gemma3:1b', 'gemma2:2b', 'llama3.2', 'phi3', 'mistral', 'qwen2.5'],
            variable=self.model_var,
            command=self.change_model
        )
        model_menu.pack(side="left", padx=5)
        
        # Buttons
        ctk.CTkButton(
            control_frame,
            text="📁 Upload CSV",
            command=self.load_csv,
            width=120,
            fg_color="#3498db",
            hover_color="#2980b9"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame,
            text="🔮 Generate Embeddings",
            command=self.generate_embeddings,
            width=150,
            fg_color="#e67e22",
            hover_color="#d35400"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame,
            text="📋 Summarize",
            command=self.auto_summarize,
            width=120,
            fg_color="#9b59b6",
            hover_color="#8e44ad"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame,
            text="🗑️ Clear",
            command=self.clear_chat,
            fg_color="red",
            hover_color="darkred",
            width=100
        ).pack(side="left", padx=5)
        
        # Status
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="Initializing...",
            text_color="gray"
        )
        self.status_label.pack(side="right", padx=10)
        
        # Main content area
        content_frame = ctk.CTkFrame(self.root)
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # Left side - CSV Preview
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(
            left_frame,
            text="CSV Data Preview",
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        self.csv_preview = ctk.CTkTextbox(
            left_frame,
            font=("Courier", 10),
            wrap="none"
        )
        self.csv_preview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right side - Analysis
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(
            right_frame,
            text="Analysis & Questions",
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        self.chat_display = ctk.CTkTextbox(
            right_frame,
            font=("Arial", 11),
            wrap="word"
        )
        self.chat_display.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = ctk.CTkFrame(self.root)
        input_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.input_text = ctk.CTkTextbox(
            input_frame,
            height=60,
            font=("Arial", 12)
        )
        self.input_text.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.input_text.bind('<Return>', self.on_enter)
        
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Ask",
            command=self.send_message,
            width=100,
            height=60,
            font=("Arial", 14, "bold")
        )
        self.send_button.pack(side="right")
    
    def start_server(self):
        def start():
            self.update_status("Starting server...")
            if self.server.start():
                self.root.after(0, lambda: self.update_status("✓ Ready"))
                self.root.after(0, lambda: self.add_message("System", "Welcome! Upload a CSV file to begin.", "system"))
            else:
                self.root.after(0, lambda: self.update_status("✗ Failed"))
        
        threading.Thread(target=start, daemon=True).start()
    
    def update_status(self, message):
        self.status_label.configure(text=message)
    
    def add_message(self, sender, message, msg_type="user"):
        prefix = "📤 " if msg_type == "user" else "🤖 " if msg_type == "bot" else "ℹ️ "
        self.chat_display.insert("end", f"{prefix}{sender}: {message}\n\n")
        self.chat_display.see("end")
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.csv_data = CSVParser(file_path)
            self.csv_filename = os.path.basename(file_path)
            self.embeddings_ready = False
            
            # Display preview
            self.csv_preview.delete("1.0", "end")
            
            info_text = f"File: {self.csv_filename}\n"
            info_text += f"Rows: {len(self.csv_data.rows)}\n"
            info_text += f"Columns: {len(self.csv_data.headers)}\n"
            info_text += f"\nColumns: {', '.join(self.csv_data.headers)}\n"
            info_text += "\n" + "="*80 + "\n\n"
            info_text += self.csv_data.get_preview(20)
            
            if len(self.csv_data.rows) > 20:
                info_text += f"\n\n... and {len(self.csv_data.rows) - 20} more rows"
            
            self.csv_preview.insert("1.0", info_text)
            
            self.add_message("System", 
                f"✓ Loaded {self.csv_filename} ({len(self.csv_data.rows)} rows, {len(self.csv_data.headers)} columns)", 
                "system")
            
            self.add_message("System", 
                "💡 Click 'Generate Embeddings' to enable intelligent querying of all data", 
                "system")
            
            self.generate_basic_summary()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            self.add_message("System", f"✗ Error: {str(e)}", "system")
    
    def generate_embeddings(self):
        """Generate embeddings for CSV data"""
        if self.csv_data is None:
            messagebox.showwarning("No Data", "Please upload a CSV file first!")
            return
        
        self.update_status("Generating embeddings...")
        self.add_message("System", "Generating embeddings... This may take a moment.", "system")
        
        def generate():
            try:
                # Create chunks
                num_chunks = self.csv_data.create_chunks(chunk_size=10)
                self.root.after(0, lambda: self.add_message("System", 
                    f"Created {num_chunks} chunks from {len(self.csv_data.rows)} rows", "system"))
                
                # Generate embeddings
                num_embeddings = self.csv_data.generate_embeddings(self.embedding_model)
                
                self.embeddings_ready = True
                self.root.after(0, lambda: self.add_message("System", 
                    f"✓ Generated {num_embeddings} embeddings! You can now ask detailed questions.", "system"))
                self.root.after(0, lambda: self.update_status("✓ Embeddings ready"))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_message("System", f"✗ Error: {str(e)}", "system"))
                self.root.after(0, lambda: self.update_status("✗ Error"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def generate_basic_summary(self):
        """Generate basic statistical summary"""
        if self.csv_data is None:
            return
        
        try:
            summary_parts = []
            summary_parts.append(f"Dataset: {self.csv_filename}")
            summary_parts.append(f"Total Records: {len(self.csv_data.rows)}")
            summary_parts.append(f"Total Columns: {len(self.csv_data.headers)}")
            
            numeric_cols = self.csv_data.get_numeric_columns()
            text_cols = [h for h in self.csv_data.headers if h not in numeric_cols]
            
            if numeric_cols:
                summary_parts.append(f"\nNumeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")
            if text_cols:
                summary_parts.append(f"\nText Columns ({len(text_cols)}): {', '.join(text_cols)}")
            
            if numeric_cols:
                summary_parts.append("\nNumeric Statistics:")
                for col in numeric_cols[:5]:
                    stats = self.csv_data.get_stats(col)
                    if stats:
                        summary_parts.append(
                            f"  {col}: mean={stats['mean']:.2f}, "
                            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
                        )
            
            self.csv_summary = "\n".join(summary_parts)
            
        except Exception as e:
            print(f"Error generating summary: {e}")
    
    def auto_summarize(self):
        """Generate AI summary"""
        if self.csv_data is None:
            messagebox.showwarning("No Data", "Please upload a CSV file first!")
            return
        
        self.add_message("You", "Generate a comprehensive summary", "user")
        self.send_button.configure(state="disabled")
        self.update_status("Analyzing...")
        
        def analyze():
            try:
                import ollama
                
                # Use relevant chunks if embeddings are ready
                if self.embeddings_ready:
                    relevant_chunks = self.csv_data.find_relevant_chunks(
                        "summary statistics overview", 
                        self.embedding_model,
                        top_k=5
                    )
                    context = "\n\n".join(relevant_chunks)
                else:
                    # Fallback to sample data
                    context = json.dumps(self.csv_data.to_dict()[:10], indent=2)
                
                prompt = f"""Analyze this CSV dataset:

                            Filename: {self.csv_filename}
                            Total Rows: {len(self.csv_data.rows)}
                            Columns: {', '.join(self.csv_data.headers)}

                            {context}

                            Statistics:
                            {self.csv_summary}

                            Provide:
                            1. Dataset overview
                            2. Key patterns and insights
                            3. Data quality observations

                            Keep it concise."""
                
                response = ollama.chat(
                    model=self.current_model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                
                answer = response['message']['content']
                self.root.after(0, lambda: self.add_message("Bot", answer, "bot"))
                self.root.after(0, lambda: self.update_status("✓ Ready"))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_message("System", f"Error: {e}", "system"))
                self.root.after(0, lambda: self.update_status("✗ Error"))
            finally:
                self.root.after(0, lambda: self.send_button.configure(state="normal"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def on_enter(self, event):
        if not event.state & 0x1:
            self.send_message()
            return 'break'
    
    def send_message(self):
        if self.csv_data is None:
            messagebox.showwarning("No Data", "Please upload a CSV file first!")
            return
        
        message = self.input_text.get("1.0", "end-1c").strip()
        if not message:
            return
        
        self.input_text.delete("1.0", "end")
        self.add_message("You", message, "user")
        
        self.send_button.configure(state="disabled")
        self.update_status("Thinking...")
        
        def get_response():
            try:
                import ollama
                
                # Use embeddings for retrieval if available
                if self.embeddings_ready:
                    relevant_chunks = self.csv_data.find_relevant_chunks(
                        message, 
                        self.embedding_model,
                        top_k=3
                    )
                    context = "\n\n".join(relevant_chunks)
                    context_info = f"Retrieved {len(relevant_chunks)} relevant data chunks"
                else:
                    # Fallback to sample
                    context = json.dumps(self.csv_data.to_dict()[:5], indent=2)
                    context_info = "Using sample data (generate embeddings for full access)"
                
                prompt = f"""CSV Dataset: {self.csv_filename}
                            Total Rows: {len(self.csv_data.rows)}
                            Columns: {', '.join(self.csv_data.headers)}

                            Relevant Data:
                            {context}

                            Statistics:
                            {self.csv_summary}

                            Question: {message}

                            Answer based on the data above. {context_info}."""
                
                self.conversation.append({'role': 'user', 'content': prompt})
                
                response = ollama.chat(
                    model=self.current_model,
                    messages=self.conversation
                )
                
                answer = response['message']['content']
                self.conversation.append({'role': 'assistant', 'content': answer})
                
                self.root.after(0, lambda: self.add_message("Bot", answer, "bot"))
                self.root.after(0, lambda: self.update_status("✓ Ready"))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_message("System", f"Error: {e}", "system"))
                self.root.after(0, lambda: self.update_status("✗ Error"))
            finally:
                self.root.after(0, lambda: self.send_button.configure(state="normal"))
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def change_model(self, choice):
        self.current_model = choice
        self.add_message("System", f"Switched to {choice}", "system")
    
    def clear_chat(self):
        self.conversation = []
        self.chat_display.delete("1.0", "end")
        self.add_message("System", "Chat cleared!", "system")
    
    def on_closing(self):
        self.server.stop()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

