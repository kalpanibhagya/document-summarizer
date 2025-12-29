from Imports import *

class OllamaServer:
    def __init__(self):
        self.process = None
    
    def start(self, timeout=10):
        """Start Ollama server with timeout"""
        try:
            if self.is_running():
                return True
            
            self.process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_running():
                    return True
                time.sleep(0.5)
            
            return False
            
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False
    
    def is_running(self):
        try:
            response = requests.get('http://localhost:11434', timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
