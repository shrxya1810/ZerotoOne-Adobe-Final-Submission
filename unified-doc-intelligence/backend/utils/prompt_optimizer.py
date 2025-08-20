"""
Prompt optimization utilities to reduce token usage and improve response quality.
Specifically designed for PDF analysis and knowledge graph generation.
"""
import re
from typing import List, Dict, Tuple

class PromptOptimizer:
    @staticmethod
    def truncate_content(content: str, max_chars: int = 1000) -> str:
        """Smart content truncation preserving meaning."""
        if len(content) <= max_chars:
            return content
        
        # Try to find sentence boundaries
        sentences = re.split(r'[.!?]+', content)
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence) > max_chars:
                break
            truncated += sentence + ". "
        
        return truncated.strip() or content[:max_chars] + "..."
    
    @staticmethod
    def extract_key_sentences(text: str, num_sentences: int = 3) -> str:
        """Extract key sentences for context reduction."""
        sentences = re.split(r'[.!?]+', text)
        # Simple heuristic: longer sentences often contain more information
        key_sentences = sorted(sentences, key=len, reverse=True)[:num_sentences]
        return '. '.join(key_sentences) + '.'
    
    @staticmethod
    def create_concise_prompt(task: str, content: str, max_tokens: int = 500) -> str:
        """Create concise prompts that reduce token usage."""
        # Remove common filler words
        filler_words = ['please', 'kindly', 'would you', 'could you', 'thank you']
        task = ' '.join([word for word in task.split() if word.lower() not in filler_words])
        
        # Use abbreviations for common phrases
        abbreviations = {
            'please provide': 'provide',
            'please analyze': 'analyze',
            'please summarize': 'summarize',
            'based on the following': 'from:',
            'please keep the response': 'keep response',
            'please explain': 'explain',
            'please describe': 'describe'
        }
        
        for long_phrase, short_phrase in abbreviations.items():
            task = task.replace(long_phrase, short_phrase)
        
        # Truncate content intelligently
        truncated_content = PromptOptimizer.truncate_content(content, max_tokens)
        
        return f"{task}\n\nContent: {truncated_content}"
    
    @staticmethod
    def create_pdf_analysis_prompt(pdf_content: str, analysis_type: str = "summary") -> str:
        """Create optimized prompt for PDF analysis."""
        # Extract key content (first 500 chars + key sentences)
        key_content = pdf_content[:500]
        if len(pdf_content) > 500:
            key_sentences = PromptOptimizer.extract_key_sentences(pdf_content[500:], 2)
            key_content += " " + key_sentences
        
        # Create task-specific prompts
        prompts = {
            "summary": "Summarize this document in 2-3 sentences",
            "headings": "Extract main headings and structure",
            "topics": "Identify 3-5 main topics",
            "entities": "Extract key entities, people, organizations",
            "insights": "Provide 2-3 key insights"
        }
        
        task = prompts.get(analysis_type, "Analyze this document")
        
        return f"{task}:\n\n{key_content}"
    
    @staticmethod
    def create_relationship_prompt(doc1_summary: str, doc2_summary: str, similarity: float) -> str:
        """Create optimized prompt for document relationship analysis."""
        # Truncate summaries to save tokens
        doc1_truncated = PromptOptimizer.truncate_content(doc1_summary, 300)
        doc2_truncated = PromptOptimizer.truncate_content(doc2_summary, 300)
        
        prompt = f"""Explain relationship between these documents in 2 sentences:

Doc1: {doc1_truncated}
Doc2: {doc2_truncated}
Similarity: {similarity:.0%}

Focus on shared topics, themes, or methodologies."""
        
        return prompt
    
    @staticmethod
    def create_importance_prompt(doc_summary: str, context: str = "") -> str:
        """Create optimized prompt for document importance analysis."""
        doc_truncated = PromptOptimizer.truncate_content(doc_summary, 400)
        context_truncated = PromptOptimizer.truncate_content(context, 200) if context else ""
        
        prompt = f"""Rate importance (1-10) and explain why:

Document: {doc_truncated}"""
        
        if context_truncated:
            prompt += f"\n\nContext: {context_truncated}"
        
        prompt += "\n\nProvide: 1) Importance score (1-10) 2) Brief reason"
        
        return prompt
    
    @staticmethod
    def create_community_prompt(doc_summaries: List[str], community_id: int) -> str:
        """Create optimized prompt for community analysis."""
        # Combine summaries intelligently
        combined_summary = ""
        for i, summary in enumerate(doc_summaries[:3]):  # Limit to 3 docs
            truncated = PromptOptimizer.truncate_content(summary, 200)
            combined_summary += f"Doc{i+1}: {truncated}\n"
        
        prompt = f"""Analyze this research community:

{combined_summary}

Community ID: {community_id}

Provide: 1) Main research theme 2) Key methodologies 3) Common topics"""
        
        return prompt
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return len(text) // 4
    
    @staticmethod
    def optimize_for_tokens(prompt: str, max_tokens: int = 1000) -> str:
        """Optimize prompt to stay within token limits."""
        estimated_tokens = PromptOptimizer.estimate_tokens(prompt)
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Reduce content proportionally
        reduction_factor = max_tokens / estimated_tokens
        target_length = int(len(prompt) * reduction_factor)
        
        return PromptOptimizer.truncate_content(prompt, target_length)
    
    @staticmethod
    def create_batch_pdf_prompt(pdf_requests: List[Dict]) -> str:
        """Create batch prompt for multiple PDF analysis requests."""
        prompt = "Analyze these documents and respond in JSON format:\n\n"
        
        for i, req in enumerate(pdf_requests):
            content = req.get('content', '')[:300]  # Limit each document
            analysis_type = req.get('type', 'summary')
            
            prompt += f"Doc{i+1} ({analysis_type}): {content}\n\n"
        
        prompt += """Respond with JSON array:
[{"doc_id": 1, "analysis": "result"}, {"doc_id": 2, "analysis": "result"}, ...]"""
        
        return prompt

# Example usage and testing
if __name__ == "__main__":
    # Test prompt optimization
    test_content = "This is a very long document about machine learning algorithms. " * 50
    
    print("Original length:", len(test_content))
    print("Optimized prompt length:", len(PromptOptimizer.create_concise_prompt(
        "Please analyze this document", test_content, 500
    )))
    
    print("Estimated tokens:", PromptOptimizer.estimate_tokens(test_content)) 