from Imports import *

class CSVParser:
    """CSV parser with embedding support"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.headers = []
        self.rows = []
        self.chunks = []
        self.embeddings = []
        self.load()
    
    def load(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            self.rows = list(reader)
    
    def create_chunks(self, chunk_size=10):
        """Create text chunks from CSV rows for embedding"""
        self.chunks = []
        
        # Create metadata chunk
        meta_chunk = f"CSV Metadata:\nColumns: {', '.join(self.headers)}\n"
        meta_chunk += f"Total rows: {len(self.rows)}\n"
        meta_chunk += f"Numeric columns: {', '.join(self.get_numeric_columns())}\n"
        self.chunks.append(meta_chunk)
        
        # Create row chunks
        for i in range(0, len(self.rows), chunk_size):
            chunk_rows = self.rows[i:i+chunk_size]
            chunk_text = f"Rows {i+1} to {i+len(chunk_rows)}:\n"
            
            for row in chunk_rows:
                row_dict = dict(zip(self.headers, row))
                chunk_text += json.dumps(row_dict) + "\n"
            
            self.chunks.append(chunk_text)
        
        return len(self.chunks)
    
    def generate_embeddings(self, model='nomic-embed-text'):
        """Generate embeddings for all chunks"""
        import ollama
        
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            try:
                response = ollama.embeddings(
                    model=model,
                    prompt=chunk
                )
                self.embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Error generating embedding for chunk {i}: {e}")
                self.embeddings.append([0] * 768)  # Fallback empty embedding
        
        return len(self.embeddings)
    
    def find_relevant_chunks(self, query, model='nomic-embed-text', top_k=3):
        """Find most relevant chunks for a query"""
        import ollama
        
        if not self.embeddings:
            return []
        
        # Generate query embedding
        try:
            query_response = ollama.embeddings(
                model=model,
                prompt=query
            )
            query_embedding = query_response['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return self.chunks[:top_k]  # Fallback to first chunks
        
        # Calculate cosine similarity
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K chunks
        relevant_chunks = [self.chunks[i] for i, _ in similarities[:top_k]]
        return relevant_chunks
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def get_column(self, col_name):
        """Get all values from a column"""
        try:
            idx = self.headers.index(col_name)
            return [row[idx] for row in self.rows if idx < len(row)]
        except ValueError:
            return []
    
    def get_numeric_columns(self):
        """Identify numeric columns"""
        numeric_cols = []
        for i, header in enumerate(self.headers):
            sample = [row[i] for row in self.rows[:10] if i < len(row) and row[i]]
            if sample and all(self.is_numeric(val) for val in sample):
                numeric_cols.append(header)
        return numeric_cols
    
    def is_numeric(self, value):
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def get_stats(self, col_name):
        """Get statistics for a numeric column"""
        values = [float(v) for v in self.get_column(col_name) if self.is_numeric(v)]
        if not values:
            return None
        
        return {
            'count': len(values),
            'mean': mean(values),
            'median': median(values),
            'min': min(values),
            'max': max(values),
            'std': stdev(values) if len(values) > 1 else 0
        }
    
    def to_dict(self):
        """Convert to dictionary format"""
        return [dict(zip(self.headers, row)) for row in self.rows]
    
    def get_preview(self, n=20):
        """Get first n rows as formatted string"""
        lines = []
        header_line = " | ".join(f"{h:20}" for h in self.headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        for row in self.rows[:n]:
            row_line = " | ".join(f"{str(v):20}" for v in row)
            lines.append(row_line)
        
        return "\n".join(lines)
