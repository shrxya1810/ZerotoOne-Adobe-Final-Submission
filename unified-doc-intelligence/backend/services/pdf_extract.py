"""
PDF extraction service for Challenge 1a - PDF title and heading extraction.
Authoritative source from challenge1a project.
"""
import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import fitz  # PyMuPDF
import json
from pathlib import Path

from ..settings import settings
from ..models.schemas import ExtractResponse, OutlineItem

logger = logging.getLogger(__name__)


class PDFExtractService:
    """
    PDF extraction service that extracts document titles and hierarchical outlines.
    Based on Challenge 1a implementation with Gemini enhancement.
    """
    
    def __init__(self):
        self.gemini_available = bool(settings.GEMINI_API_KEY)
        if self.gemini_available:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini AI model initialized for PDF extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_available = False
        else:
            logger.info("Gemini API key not provided - using basic extraction")
    
    def extract_pdf_structure(self, pdf_path: str) -> ExtractResponse:
        """
        Extract title and outline from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractResponse with title, outline, and metadata
        """
        start_time = time.time()
        filename = os.path.basename(pdf_path)
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            if page_count == 0:
                return ExtractResponse(
                    success=False,
                    filename=filename,
                    title=None,
                    outline=[],
                    page_count=0,
                    processing_time=time.time() - start_time,
                    message="PDF has no pages"
                )
            
            # Extract title
            title = self._extract_title(doc)
            
            # Extract outline/headings
            outline = self._extract_outline(doc, title)
            
            doc.close()
            
            processing_time = time.time() - start_time
            logger.info(f"Extracted {len(outline)} headings from {filename} in {processing_time:.2f}s")
            
            return ExtractResponse(
                filename=filename,
                title=title,
                outline=outline,
                page_count=page_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error extracting PDF structure from {filename}: {e}")
            return ExtractResponse(
                success=False,
                filename=filename,
                title=None,
                outline=[],
                page_count=0,
                processing_time=time.time() - start_time,
                message=f"Extraction error: {str(e)}"
            )
    
    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        """Extract document title from PDF."""
        # Try PDF metadata first
        metadata = doc.metadata
        if metadata.get('title') and len(metadata['title'].strip()) > 3:
            return metadata['title'].strip()
        
        # Analyze first page for title
        if len(doc) == 0:
            return None
        
        first_page = doc[0]
        blocks = first_page.get_text("dict")
        
        # Find largest font size on first page
        max_font_size = 0
        title_candidates = []
        
        for block in blocks.get("blocks", []):
            if block.get("type") != 0:  # Only text blocks
                continue
                
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = span.get("size", 0)
                    text = span.get("text", "").strip()
                    
                    if font_size > max_font_size and len(text) > 3:
                        max_font_size = font_size
                        title_candidates = [text]
                    elif font_size == max_font_size and len(text) > 3:
                        title_candidates.append(text)
        
        # Return the first substantial title candidate
        for candidate in title_candidates:
            if len(candidate) > 10 and not self._is_header_footer(candidate):
                return candidate
        
        return title_candidates[0] if title_candidates else None
    
    def _extract_outline(self, doc: fitz.Document, title: Optional[str] = None) -> List[OutlineItem]:
        """Extract document outline/headings."""
        # Try built-in outline first
        toc = doc.get_toc()
        if toc and len(toc) > 2:  # If there's a substantial TOC
            outline = []
            for level, heading_title, page_num in toc:
                outline.append(OutlineItem(
                    level=level,
                    title=heading_title.strip(),
                    page_number=max(0, page_num - 1)  # Convert to 0-indexed
                ))
            return outline
        
        # Extract headings based on font analysis
        headings = self._extract_headings_by_font(doc, title)
        
        # Enhance with Gemini if available
        if self.gemini_available and headings:
            try:
                enhanced_headings = self._enhance_headings_with_gemini(headings)
                if enhanced_headings:
                    return enhanced_headings
            except Exception as e:
                logger.warning(f"Gemini enhancement failed: {e}")
        
        return headings
    
    def _extract_page_content(self, page: fitz.Page, page_num: int, title: Optional[str] = None) -> Tuple[Dict, List[Dict]]:
        """Extract content from a single page (for parallel processing)."""
        font_analysis = {}
        text_elements = []
        
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if block.get("type") != 0:  # Only text blocks
                continue
            
            for line in block.get("lines", []):
                line_text = ""
                font_size = 0
                is_bold = False
                bbox = line.get("bbox", [0, 0, 0, 0])
                
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    
                    line_text += text
                    font_size = max(font_size, size)
                    is_bold = is_bold or (flags & 2 ** 4)  # Bold flag
                
                line_text = line_text.strip()
                if len(line_text) < 3:
                    continue
                
                # Skip title
                if title and line_text == title:
                    continue
                
                # Record font usage
                font_key = (round(font_size, 1), is_bold)
                if font_key not in font_analysis:
                    font_analysis[font_key] = 0
                font_analysis[font_key] += 1
                
                text_elements.append({
                    "text": line_text,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "page": page_num,
                    "bbox": bbox,
                    "y_position": bbox[1]
                })
        
        return font_analysis, text_elements
    
    def _extract_headings_by_font(self, doc: fitz.Document, title: Optional[str] = None) -> List[OutlineItem]:
        """Extract headings based on font size and formatting analysis with parallel processing."""
        combined_font_analysis = {}
        all_text_elements = []
        
        # Process pages in parallel
        max_workers = min(len(doc), 8)  # Limit to 8 workers for memory efficiency
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit page processing tasks
            future_to_page = {
                executor.submit(self._extract_page_content, doc[page_num], page_num, title): page_num 
                for page_num in range(len(doc))
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                try:
                    font_analysis, text_elements = future.result()
                    
                    # Merge font analysis
                    for font_key, count in font_analysis.items():
                        if font_key not in combined_font_analysis:
                            combined_font_analysis[font_key] = 0
                        combined_font_analysis[font_key] += count
                    
                    # Add text elements
                    all_text_elements.extend(text_elements)
                    
                except Exception as e:
                    page_num = future_to_page[future]
                    logger.warning(f"Error processing page {page_num}: {e}")
        
        # Determine heading font sizes (top 3 most prominent)
        sorted_fonts = sorted(combined_font_analysis.items(), key=lambda x: x[1], reverse=True)
        body_font = sorted_fonts[0][0] if sorted_fonts else (12, False)
        
        heading_fonts = []
        for (size, bold), count in sorted_fonts:
            if size > body_font[0] or (size == body_font[0] and bold and not body_font[1]):
                heading_fonts.append((size, bold))
        
        # Extract headings
        headings = []
        for element in all_text_elements:
            font_key = (round(element["font_size"], 1), element["is_bold"])
            
            if font_key in heading_fonts and self._is_likely_heading(element["text"]):
                level = heading_fonts.index(font_key) + 1
                if level <= 6:  # Max 6 heading levels
                    headings.append(OutlineItem(
                        level=level,
                        title=element["text"],
                        page_number=element["page"],
                        bbox=element["bbox"]
                    ))
        
        # Sort by page and position
        headings.sort(key=lambda h: (h.page_number, -h.bbox[1] if h.bbox else 0))
        
        return headings
    
    def _enhance_headings_with_gemini(self, headings: List[OutlineItem]) -> List[OutlineItem]:
        """Use Gemini AI to refine heading extraction."""
        if not headings or not self.gemini_available:
            return headings
        
        # Prepare prompt with heading candidates
        candidates_text = []
        for i, heading in enumerate(headings):
            candidates_text.append(
                f"- Page {heading.page_number + 1}: \"{heading.title}\" "
                f"(Level: {heading.level})"
            )
        
        prompt = f"""
        You are an expert document analyst. Review these candidate headings extracted from a PDF and determine which ones are actual section heading
        s.

        Candidates:
        {chr(10).join(candidates_text)}

        Return a JSON array of refined headings. Each heading should have:
        - "text": the heading text (clean, no numbering prefixes)
        - "level": hierarchical level (1=H1, 2=H2, etc.)
        - "page": page number (0-indexed)

        Rules:
        - Exclude paragraph text, captions, headers/footers
        - Infer proper hierarchy levels
        - Keep only true section headings

        Example output:
        [
          {{"text": "Introduction", "level": 1, "page": 0}},
          {{"text": "Method", "level": 1, "page": 2}},
          {{"text": "Data Collection", "level": 2, "page": 3}}
        ]
        """
        
        try:
            import google.generativeai as genai
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            refined_data = json.loads(response.text)
            
            # Convert to OutlineItem objects
            refined_headings = []
            for item in refined_data:
                if isinstance(item, dict) and all(k in item for k in ["text", "level", "page"]):
                    refined_headings.append(OutlineItem(
                        level=item["level"],
                        title=item["text"],
                        page_number=item["page"]
                    ))
            
            return refined_headings
            
        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
            return headings
    
    def _is_likely_heading(self, text: str) -> bool:
        """Determine if text is likely a heading."""
        if len(text) > 200:  # Too long for heading
            return False
        
        # Skip common non-heading patterns
        skip_patterns = [
            r'^page \d+',
            r'^\d+$',
            r'^[^\w]*$',
            r'www\.',
            r'@',
            r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP address
        ]
        
        for pattern in skip_patterns:
            import re
            if re.search(pattern, text.lower()):
                return False
        
        # Positive indicators
        if any(word in text.lower() for word in [
            'chapter', 'section', 'introduction', 'conclusion', 
            'method', 'result', 'discussion', 'abstract', 'summary',
            'background', 'analysis', 'findings', 'recommendation'
        ]):
            return True
        
        # Check for heading-like structure
        if text.istitle() or text.isupper():
            return True
        
        # Numbered sections
        import re
        if re.match(r'^\d+\.?\s+\w+', text):
            return True
        
        return len(text.split()) <= 10  # Reasonable heading length
    
    def _extract_page_text(self, page: fitz.Page, page_num: int) -> str:
        """Extract text from a single page (for parallel processing)."""
        try:
            text = page.get_text()
            return f"{text}\n\n--- Page {page_num + 1} ---\n\n"
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num}: {e}")
            return f"\n\n--- Page {page_num + 1} (Error) ---\n\n"
    
    def extract_content_parallel(self, doc: fitz.Document) -> str:
        """Extract full text content with parallel page processing."""
        if len(doc) == 0:
            return ""
        
        # For small documents, use sequential processing to avoid overhead
        if len(doc) <= 3:
            content = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                content += self._extract_page_text(page, page_num)
            return content
        
        # Use parallel processing for larger documents
        max_workers = min(len(doc), 8)  # Limit workers for memory efficiency
        page_contents = [""] * len(doc)  # Pre-allocate to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit page extraction tasks
            future_to_page = {
                executor.submit(self._extract_page_text, doc[page_num], page_num): page_num 
                for page_num in range(len(doc))
            }
            
            # Collect results in order
            for future in as_completed(future_to_page):
                try:
                    page_num = future_to_page[future]
                    page_contents[page_num] = future.result()
                except Exception as e:
                    page_num = future_to_page[future]
                    logger.warning(f"Error processing page {page_num}: {e}")
                    page_contents[page_num] = f"\n\n--- Page {page_num + 1} (Error) ---\n\n"
        
        return "".join(page_contents)
    
    def _is_header_footer(self, text: str) -> bool:
        """Check if text is likely a header or footer."""
        lower_text = text.lower()
        
        # Common header/footer patterns
        patterns = [
            'page',
            'copyright',
            '©',
            'confidential',
            'draft',
            'proprietary',
            'all rights reserved'
        ]
        
        return any(pattern in lower_text for pattern in patterns)


# Global service instance
pdf_extract_service = PDFExtractService()


# Convenience function
def extract_pdf_structure(pdf_path: str) -> ExtractResponse:
    """Extract PDF structure using the global service."""
    return pdf_extract_service.extract_pdf_structure(pdf_path)
