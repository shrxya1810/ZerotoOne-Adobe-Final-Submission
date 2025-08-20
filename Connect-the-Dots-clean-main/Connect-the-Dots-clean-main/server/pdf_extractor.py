import pdfplumber
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from hierarchy_enhancer import HierarchyEnhancer
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- Gemini Integration Start ---
# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except TypeError:
    print("ERROR: GOOGLE_API_KEY not found. Please ensure it is set in your .env file.")
    # You might want to exit or handle this more gracefully
# --- Gemini Integration End ---


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
        self.pages_data = []
        self.hierarchy_enhancer = HierarchyEnhancer()
        
        # Multiline title detection parameters
        self.FONT_SIZE_TOLERANCE = 2.0
        self.MAX_LINE_SPACING = 1.8
        self.MIN_WHITESPACE_ABOVE = 25
        self.MIN_WHITESPACE_BELOW = 15
        self.MAX_TITLE_LINES = 5
        self.POSITION_TOLERANCE = 20
        
        self._extract_pages_data()
        # --- Gemini Integration ---
        # Set up the Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        # --- End Gemini Integration ---

    def close(self):
        """Closes the PDF file object."""
        if self.pdf:
            self.pdf.close()

    def _extract_pages_data(self):
        """Extracts and stores data from each page of the PDF."""
        for i, page in enumerate(self.pdf.pages):
            page_data = {
                'page_num': i + 1,
                'width': page.width,
                'height': page.height,
                'chars': page.chars,
                'lines': page.lines,
                'rects': page.rects
            }
            self.pages_data.append(page_data)

    def _get_char_properties(self, char: Dict) -> Tuple[str, float, bool]:
        """Extracts font name, size, and boldness from a character object."""
        fontname = char.get('fontname', 'default')
        size = round(char.get('size', 0), 2)
        is_bold = 'bold' in fontname.lower()
        return fontname, size, is_bold

    def _get_title_and_font_properties(self) -> Tuple[Optional[str], Dict[float, int]]:
        """
        Analyzes the first page to find the most likely title and font size distribution.
        """
        if not self.pages_data:
            return None, defaultdict(int)

        first_page_chars = self.pages_data[0]['chars']
        if not first_page_chars:
            return None, defaultdict(int)

        font_sizes = defaultdict(int)
        for char in first_page_chars:
            _, size, _ = self._get_char_properties(char)
            font_sizes[size] += 1

        largest_font_size = max(font_sizes, key=font_sizes.get)
        
        title_chars = [c for c in first_page_chars if round(c.get('size', 0), 2) == largest_font_size]
        
        if not title_chars:
            return None, font_sizes

        # Group characters into lines based on y-position
        lines = defaultdict(list)
        for char in title_chars:
            lines[char['y0']].append(char)
        
        # Get the highest line (top of the page)
        title_line_chars = lines[min(lines.keys())]
        title_text = "".join(c['text'] for c in sorted(title_line_chars, key=lambda c: c['x0']))

        return title_text.strip(), font_sizes

    def _extract_headings_from_pages(self, title_font_size: float, font_sizes: Dict[float, int]) -> List[Dict]:
        """
        Extracts headings from all pages based on font properties.
        """
        headings = []
        if not font_sizes:
            return headings

        # Determine heading font sizes (e.g., the top 2-3 sizes excluding title)
        sorted_font_sizes = sorted(font_sizes.keys(), reverse=True)
        heading_font_sizes = [s for s in sorted_font_sizes if s < title_font_size][:3]

        for page_data in self.pages_data:
            page_lines = self._group_chars_into_lines(page_data['chars'])
            for y0, line_chars in page_lines.items():
                line_text = "".join(c['text'] for c in line_chars).strip()
                if not line_text:
                    continue

                line_font_size = round(line_chars[0].get('size', 0), 2)
                if line_font_size in heading_font_sizes:
                    headings.append({
                        "text": line_text,
                        "level": heading_font_sizes.index(line_font_size) + 1,
                        "page": page_data['page_num'],
                        "y_position": y0,
                        "font_size": line_font_size,
                        "is_bold": "bold" in line_chars[0].get('fontname', '').lower()
                    })
        return headings

    def _group_chars_into_lines(self, chars: List[Dict]) -> Dict[float, List[Dict]]:
        """Groups characters into lines based on their y-coordinate."""
        lines = defaultdict(list)
        for char in chars:
            lines[char['y0']].append(char)
        for y0 in lines:
            lines[y0] = sorted(lines[y0], key=lambda c: c['x0'])
        return lines

    def _refine_headings_with_gemini(self, candidate_headings: List[Dict]) -> List[Dict]:
        """
        Uses the Gemini API to refine a list of candidate headings.
        """
        if not candidate_headings:
            return []

        # Convert candidate headings to a simpler format for the prompt
        prompt_headings = [
            f"- Page {h.get('page', 'N/A')}: \"{h.get('text', '')}\" (Font Size: {h.get('font_size', 0):.1f}, Bold: {h.get('is_bold', False)})"
            for h in candidate_headings
        ]
        
        prompt = f"""
        You are an expert document analyst. Your task is to review a list of candidate headings extracted from a PDF and determine which ones are actual headings.

        Here is the list of candidates:
        {chr(10).join(prompt_headings)}

        Please analyze this list and return a valid JSON array containing only the TRUE headings.
        - Each object in the array must have three keys: "text" (string), "level" (integer, 1 for H1, 2 for H2, etc.), and "page" (integer).
        - Infer the hierarchical level (1, 2, 3, etc.) based on numbering, font size, and context.
        - Exclude any text that is part of a paragraph, a list item, a figure caption, a table entry, or a header/footer.
        - Ensure the "text" is clean and does not contain numbering prefixes like "1.1".
        - Preserve the original page number for each heading.

        Example of a perfect response:
        [
          {{ 
            "text": "Introduction",
            "level": 1,
            "page": 1
          }},
          {{
            "text": "Project Background",
            "level": 2,
            "page": 2
          }}
        ]

        Now, provide the refined JSON for the candidate list above.
        """

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            # The response text should be a valid JSON string.
            refined_headings = json.loads(response.text)
            # We need to add back the other metadata if needed, but for now, this matches the required structure.
            # Let's ensure the structure is correct before returning.
            validated_headings = []
            for h in refined_headings:
                if isinstance(h, dict) and 'text' in h and 'level' in h and 'page' in h:
                    validated_headings.append(h)
            return validated_headings
        except Exception as e:
            print(f"An error occurred during Gemini API call: {e}")
            # Fallback to the original, unrefined headings
            return self._classify_and_clean_headings(candidate_headings)

    def extract_data(self) -> Dict:
        """
        Main method to extract title and a refined list of headings from the PDF.
        Adds robust error logging for debugging.
        """
        import traceback
        try:
            title_info = self._extract_document_title_comprehensive()
            title = title_info.get('title', '')

            # Step 1: Get all potential headings using existing logic
            candidate_headings = self.extract_headings_with_title_awareness(title_info)

            # Step 2: Refine the headings using the Gemini model
            print(f"Found {len(candidate_headings)} candidates. Refining with Gemini...")
            refined_outline = self._refine_headings_with_gemini(candidate_headings)
            print(f"Gemini refinement complete. Final outline has {len(refined_outline)} headings.")

            return {
                "title": title,
                "outline": refined_outline
            }
        except Exception as e:
            print("\n[PDFExtractor.extract_data] Exception occurred:")
            traceback.print_exc()
            return {"title": "", "outline": [], "error": str(e)}

    def extract_headings_with_title_awareness(self, title_info: Dict) -> List[Dict]:
        """
        Extracts a list of CANDIDATE headings. The final classification is done by the LLM.
        (This is a simplified version of your existing function to show integration)
        """
        if self._is_form_document():
            return []

        headings = []
        all_lines_data = []
        
        for page_data in self.pages_data:
            page_headings = self._extract_page_headings(page_data)
            headings.extend(page_headings)
            
            lines_data = self._extract_lines_with_metadata(page_data['chars'])
            for line_data in lines_data:
                line_data['page'] = page_data['page_num']
                if 'y_position' not in line_data:
                    line_data['y_position'] = len(all_lines_data) * 14
            all_lines_data.extend(lines_data)
        
        if title_info['title']:
            headings = self._exclude_title_from_headings(headings, title_info)
        
        enhanced_headings = self.hierarchy_enhancer.enhance_headings(headings, all_lines_data)
        filtered_headings = self._filter_false_positive_headers(enhanced_headings, all_lines_data)
        major_headings = self._filter_to_major_headings_only(filtered_headings)
        
        # Return the raw candidates with all metadata for the LLM to use
        return major_headings

    def extract_title(self) -> str:
        """Legacy method - now calls comprehensive title extraction"""
        title_info = self._extract_document_title_comprehensive()
        return title_info['title']
    
    def _extract_document_title_comprehensive(self) -> Dict:
        """Extract document title and return metadata for hierarchy exclusion"""
        
        if not self.pages_data:
            return {'title': '', 'title_metadata': None}
        
        first_page = self.pages_data[0]
        lines_data = self._extract_lines_with_metadata(first_page['chars'])
        
        # Strategy 1: Multiline title detection (NEW - highest priority)
        title_by_multiline = self._extract_title_by_multiline(lines_data)
        
        # Strategy 2: Find absolute largest font on first page
        title_by_max_font = self._find_absolute_largest_font_text(lines_data)
        
        # Strategy 3: Pattern-based (existing logic)
        title_by_pattern = self._extract_title_by_pattern(lines_data)
        
        # Strategy 4: Position + spacing context
        title_by_context = self._extract_title_by_spatial_context(lines_data)
        
        # Strategy 5: Center-aligned prominent text
        title_by_center = self._find_center_prominent_text(lines_data)
        
        # Combine strategies with confidence scoring
        candidates = []
        
        if title_by_multiline:
            candidates.append({
                'text': title_by_multiline['text'],
                'confidence': 0.95,  # Highest confidence for multiline
                'method': 'multiline',
                'metadata': title_by_multiline
            })
        
        if title_by_max_font:
            candidates.append({
                'text': title_by_max_font['text'],
                'confidence': 0.9,
                'method': 'max_font',
                'metadata': title_by_max_font
            })
        
        if title_by_pattern and title_by_pattern != title_by_max_font.get('text') if title_by_max_font else True:
            candidates.append({
                'text': title_by_pattern,
                'confidence': 0.8,
                'method': 'pattern',
                'metadata': self._get_text_metadata(title_by_pattern, lines_data)
            })
        
        if title_by_context and title_by_context['text'] not in [c['text'] for c in candidates]:
            candidates.append({
                'text': title_by_context['text'],
                'confidence': 0.7,
                'method': 'context',
                'metadata': title_by_context
            })
        
        if title_by_center and title_by_center['text'] not in [c['text'] for c in candidates]:
            candidates.append({
                'text': title_by_center['text'],
                'confidence': 0.6,
                'method': 'center',
                'metadata': title_by_center
            })
        
        if not candidates:
            return {'title': '', 'title_metadata': None}
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        
        cleaned_title = self._clean_title_text(best_candidate['text'])
        self._cached_title = cleaned_title
        
        return {
            'title': cleaned_title,
            'title_metadata': best_candidate['metadata'],
            'title_method': best_candidate['method'],
            'title_confidence': best_candidate['confidence'],
            'all_candidates': candidates
        }
    
    def _extract_title_by_font_size(self, lines_data: List[Dict]) -> str:
        """Extract title based on largest font size"""
        title_candidates = []
        
        for line_data in lines_data:
            text = line_data['text'].strip()
            if (text and 
                len(text) > 3 and 
                not self._is_header_footer_metadata(text) and
                not self._is_form_field(text)):
                
                prominence_score = self._calculate_prominence_score(line_data, lines_data)
                title_candidates.append({
                    'text': text,
                    'score': prominence_score,
                    'font_size': line_data['avg_font_size']
                })
        
        if not title_candidates:
            return ""
        
        title_candidates.sort(key=lambda x: x['score'], reverse=True)
        return title_candidates[0]['text']
    
    def _extract_title_by_pattern(self, lines_data: List[Dict]) -> str:
        """Extract title based on common title patterns"""
        for line_data in lines_data[:10]:
            text = line_data['text'].strip()
            
            title_patterns = [
                r'^[A-Z][A-Za-z\s:]+[A-Za-z]$',
                r'^[A-Z][a-z]+\s+[A-Z][a-z\s]+$',
                r'^\w+:\s+.+$',
                r'^[A-Z][A-Z\s]{8,}$',  # ALL CAPS titles
            ]
            
            for pattern in title_patterns:
                if (re.match(pattern, text) and 
                    len(text) > 10 and len(text) < 100 and
                    not self._is_header_footer_metadata(text) and
                    not self._is_form_field(text)):
                    return text
        
        return ""
    
    def _extract_title_by_position(self, lines_data: List[Dict]) -> str:
        """Extract title as first significant text block"""
        for line_data in lines_data[:5]:
            text = line_data['text'].strip()
            
            if (text and 
                len(text) > 5 and
                not self._is_header_footer_metadata(text) and
                not self._is_form_field(text)):
                return text
        
        return ""
    
    def _calculate_prominence_score(self, line_data: Dict, all_lines: List[Dict]) -> float:
        all_font_sizes = [l['avg_font_size'] for l in all_lines if l['text'].strip()]
        avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12
        
        font_score = line_data['avg_font_size'] / avg_font_size
        position_score = 1.2 if line_data['position'] == 'center' else 1.0
        bold_score = 1.3 if line_data['is_bold'] else 1.0
        
        text_len = len(line_data['text'].strip())
        if 10 <= text_len <= 150:
            length_score = 1.2
        elif text_len > 150:
            length_score = 0.8
        else:
            length_score = 1.0
        
        return font_score * position_score * bold_score * length_score
    
    def _is_header_footer_metadata(self, text: str) -> bool:
        text_lower = text.lower()
        patterns = [
            r'page\s+\d+',
            r'\d+\s+of\s+\d+',
            r'version\s+\d+\.\d+',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'www\.',
            r'@',
            r'copyright',
            r'©',
            r'confidential',
            r'draft',
            r'revision\s+history',
            r'table\s+of\s+contents'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _is_form_field(self, text: str) -> bool:
        patterns = [
            r'^\d+\.\s*$',
            r'^Name\s+of\s+',
            r'^Date\s+of\s+',
            r'^Amount\s+of\s+',
            r'^Whether\s+',
            r'^\([a-z]\)\s+',
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _clean_title_text(self, title: str) -> str:
        title = re.sub(r'\s+', ' ', title.strip())
        title = re.sub(r'^[^\w]+|[^\w\s.!?:-]+$', '', title)
        
        return title
    
    def _find_absolute_largest_font_text(self, lines_data: List[Dict]) -> Optional[Dict]:
        """Find text with absolutely largest font size on first page"""
        
        max_font_size = 0
        max_font_text = None
        
        for line_data in lines_data[:15]:
            text = line_data['text'].strip()
            font_size = line_data['avg_font_size']
            
            if (text and len(text) > 3 and len(text) < 200 and
                font_size > max_font_size and
                not self._is_header_footer_metadata(text) and
                not self._is_form_field(text)):
                
                max_font_size = font_size
                max_font_text = {
                    'text': text,
                    'font_size': font_size,
                    'is_bold': line_data['is_bold'],
                    'position': line_data['position'],
                    'y_position': line_data.get('y_position', 0),
                    'page': 0
                }
        
        return max_font_text
    
    def _extract_title_by_spatial_context(self, lines_data: List[Dict]) -> Optional[Dict]:
        """Extract title based on spatial isolation (lots of whitespace around it)"""
        
        for i, line_data in enumerate(lines_data[:10]):
            text = line_data['text'].strip()
            
            if (text and len(text) > 5 and len(text) < 150 and
                not self._is_header_footer_metadata(text) and
                not self._is_form_field(text)):
                
                whitespace_above = self._measure_whitespace_above_line(lines_data, i)
                whitespace_below = self._measure_whitespace_below_line(lines_data, i)
                
                if (whitespace_above > self.MIN_WHITESPACE_ABOVE and 
                    whitespace_below > self.MIN_WHITESPACE_BELOW):
                    
                    return {
                        'text': text,
                        'font_size': line_data['avg_font_size'],
                        'is_bold': line_data['is_bold'],
                        'position': line_data['position'],
                        'y_position': line_data.get('y_position', 0),
                        'page': 0,
                        'whitespace_above': whitespace_above,
                        'whitespace_below': whitespace_below
                    }
        
        return None
    
    def _find_center_prominent_text(self, lines_data: List[Dict]) -> Optional[Dict]:
        """Find center-aligned text with good font size as potential title"""
        
        for line_data in lines_data[:10]:
            text = line_data['text'].strip()
            
            if (text and len(text) > 5 and len(text) < 150 and
                line_data['position'] == 'center' and
                not self._is_header_footer_metadata(text) and
                not self._is_form_field(text)):
                
                return {
                    'text': text,
                    'font_size': line_data['avg_font_size'],
                    'is_bold': line_data['is_bold'],
                    'position': line_data['position'],
                    'y_position': line_data.get('y_position', 0),
                    'page': 0
                }
        
        return None
    
    def _get_text_metadata(self, text: str, lines_data: List[Dict]) -> Optional[Dict]:
        """Get metadata for a specific text string"""
        for line_data in lines_data:
            if line_data['text'].strip() == text:
                return {
                    'text': text,
                    'font_size': line_data['avg_font_size'],
                    'is_bold': line_data['is_bold'],
                    'position': line_data['position'],
                    'y_position': line_data.get('y_position', 0),
                    'page': 0
                }
        return None
    
    def _measure_whitespace_above_line(self, lines_data: List[Dict], index: int) -> float:
        """Measure vertical whitespace above a line"""
        if index == 0:
            return float('inf')
        
        prev_line = lines_data[index - 1]
        current_line = lines_data[index]
        
        current_y = current_line.get('y_position', index * 14)
        prev_y = prev_line.get('y_position', (index - 1) * 14)
        prev_height = prev_line.get('height', 14)
        
        return abs(current_y - (prev_y + prev_height))
    
    def _measure_whitespace_below_line(self, lines_data: List[Dict], index: int) -> float:
        """Measure vertical whitespace below a line"""
        if index >= len(lines_data) - 1:
            return float('inf')
        
        current_line = lines_data[index]
        next_line = lines_data[index + 1]
        
        current_y = current_line.get('y_position', index * 14)
        current_height = current_line.get('height', 14)
        next_y = next_line.get('y_position', (index + 1) * 14)
        
        return abs(next_y - (current_y + current_height))
    
    def _detect_multiline_title_candidates(self, lines_data: List[Dict]) -> List[List[Dict]]:
        """Detect groups of lines that could form multiline titles"""
        try:
            if not lines_data:
                return []

            # Step 1: Identify prominent lines (top 20% font sizes)
            prominent_lines = self._identify_prominent_lines(lines_data[:15])  # First 15 lines only
            if not prominent_lines:
                return []

            # Step 2: Group adjacent prominent lines with similar properties
            line_groups = self._group_adjacent_lines(prominent_lines)
            if not line_groups:
                line_groups = []

            # Step 3: Filter groups to keep only valid multiline title candidates
            valid_groups = []
            for group in line_groups:
                if self._is_valid_title_group(group, lines_data):
                    valid_groups.append(group)

            return valid_groups
        except Exception:
            return []
    
    def _identify_prominent_lines(self, lines_data: List[Dict]) -> List[Dict]:
        """Identify lines with prominent font sizes (top 20%)"""
        
        try:
            if not lines_data:
                return []

            # Get all font sizes and calculate threshold
            font_sizes = [line['avg_font_size'] for line in lines_data if line['text'].strip()]
            if not font_sizes:
                return []

            # Top 20% threshold
            sorted_sizes = sorted(font_sizes, reverse=True)
            threshold_index = max(1, len(sorted_sizes) // 5)  # At least 1 line
            size_threshold = sorted_sizes[threshold_index - 1]

            # Filter lines by size and basic quality checks
            prominent_lines = []
            for line in lines_data:
                text = line['text'].strip()
                if (text and 
                    len(text) > 2 and len(text) < 200 and
                    line['avg_font_size'] >= size_threshold and
                    not self._is_header_footer_metadata(text) and
                    not self._is_form_field(text)):
                    prominent_lines.append(line)

            return prominent_lines
        except Exception:
            return []
    
    def _group_adjacent_lines(self, prominent_lines: List[Dict]) -> List[List[Dict]]:
        """Group adjacent lines that could form multiline titles"""
        
        try:
            if not prominent_lines:
                return []

            groups = []
            current_group = [prominent_lines[0]]

            # Defensive: if only one prominent line, return it as a group
            if len(prominent_lines) == 1:
                return [current_group]

            for i in range(1, len(prominent_lines)):
                prev = prominent_lines[i - 1]
                curr = prominent_lines[i]
                if abs(curr.get('y_position', 0) - prev.get('y_position', 0)) < 30:
                    current_group.append(curr)
                else:
                    groups.append(current_group)
                    current_group = [curr]
            if current_group:
                groups.append(current_group)
            return groups
        except Exception:
            return []
    
    def _score_title_group(self, group: List[Dict]) -> float:
        """Score a title group for quality/likelihood"""
        
        if not group:
            return 0.0
        
        score = 0.0
        
        # Font size score (average of group)
        avg_font_size = sum(line['avg_font_size'] for line in group) / len(group)
        score += avg_font_size * 0.5
        
        # Position bonus (center-aligned gets bonus)
        if all(line.get('position') == 'center' for line in group):
            score += 15.0
        elif all(line.get('position') == 'left' for line in group):
            score += 5.0
        
        # Bold consistency bonus
        bold_lines = sum(1 for line in group if line.get('is_bold', False))
        if bold_lines == len(group):  # All bold
            score += 10.0
        elif bold_lines == 0:  # None bold (consistent)
            score += 5.0
        
        # Line count penalty (2-3 lines optimal)
        if len(group) == 1:
            score -= 5.0  # Prefer multiline
        elif len(group) in [2, 3]:
            score += 8.0  # Optimal
        elif len(group) == 4:
            score += 3.0  # Acceptable
        else:  # 5+ lines
            score -= 10.0  # Penalty
        
        # Font consistency bonus
        font_sizes = [line['avg_font_size'] for line in group]
        font_variance = max(font_sizes) - min(font_sizes)
        if font_variance <= 1.0:
            score += 8.0  # Very consistent
        elif font_variance <= 2.0:
            score += 4.0  # Somewhat consistent
        
        return score
    
    def _combine_title_group(self, group: List[Dict]) -> str:
        """Combine a group of lines into a single title string"""
        
        if not group:
            return ""
        
        # Simple concatenation with space separation
        title_parts = []
        for line in group:
            text = line['text'].strip()
            if text:
                title_parts.append(text)
        
        combined_title = ' '.join(title_parts)
        
        # Clean up the combined title
        return self._clean_title_text(combined_title)
    
    def _extract_title_by_multiline(self, lines_data: List[Dict]) -> Optional[Dict]:
        """Extract title using multiline detection strategy"""
        
        try:
            # Detect multiline title candidates
            title_groups = self._detect_multiline_title_candidates(lines_data)
            if not title_groups:
                return None

            # Score each group and select the best one
            best_group = None
            best_score = 0.0
            for group in title_groups:
                score = self._score_title_group(group)
                if score > best_score:
                    best_score = score
                    best_group = group
            if not best_group:
                return None

            # Combine the group into a single title
            combined_title = self._combine_title_group(best_group)
            if not combined_title:
                return None

            # Create metadata for the multiline title
            avg_font_size = sum(line['avg_font_size'] for line in best_group) / len(best_group)
            positions = [line.get('position', 'left') for line in best_group]
            most_common_position = max(set(positions), key=positions.count)
            y_positions = [line.get('y_position', 0) for line in best_group]
            min_y = min(y_positions) if y_positions else 0
            max_y = max(y_positions) if y_positions else 0
            is_any_bold = any(line.get('is_bold', False) for line in best_group)
            return {
                'text': combined_title,
                'font_size': avg_font_size,
                'is_bold': is_any_bold,
                'position': most_common_position,
                'y_position': min_y,
                'y_position_end': max_y,
                'page': 0,
                'line_count': len(best_group),
                'score': best_score,
                'group_lines': [line['text'].strip() for line in best_group]
            }
        except Exception:
            return None
    
    def _clean_duplicate_chars(self, text: str) -> str:
        """Remove duplicate character sequences like RRRR -> R"""
        if not text:
            return text
        
        import re
        cleaned = re.sub(r'(.)\1{2,}', r'\1', text)
        cleaned = re.sub(r'(\w)\s+\1(\s+\1)+', r'\1', cleaned)
        
        return cleaned
    
    def _is_form_document(self) -> bool:
        all_text_lines = []
        
        for page_data in self.pages_data:
            lines_data = self._extract_lines_with_metadata(page_data['chars'])
            for line_data in lines_data:
                text = line_data['text'].strip()
                if text and len(text) > 3:
                    all_text_lines.append(text)
        
        if not all_text_lines:
            return False
        
        form_patterns = 0
        total_lines = len(all_text_lines)
        
        for text in all_text_lines:
            if (re.match(r'^\d+\.\s+[A-Z]', text) or
                'Name of' in text or
                'Date of' in text or
                'Whether' in text or
                'Application form' in text or
                'S.No' in text or
                'Age' in text and 'Name' in text or
                'Signature' in text):
                form_patterns += 1
        
        form_ratio = form_patterns / total_lines if total_lines > 0 else 0
        
        title_is_form = False
        if hasattr(self, '_cached_title'):
            title_is_form = 'application form' in self._cached_title.lower()
        
        return form_ratio > 0.2 or title_is_form
    
    def extract_headings(self, title_info: Dict) -> List[Dict]:
        """Calls title-aware heading extraction with title_info argument"""
        return self.extract_headings_with_title_awareness(title_info)
    
    
    def _extract_page_headings(self, page_data: Dict) -> List[Dict]:
        chars = page_data['chars']
        page_num = page_data['page_num']
        
        if not chars:
            return []
        
        lines_data = self._extract_lines_with_metadata(chars)
        font_analysis = self._analyze_fonts(chars)
        heading_candidates = []
        
        for line_data in lines_data:
            text = line_data['text'].strip()
            
            if self._is_potential_heading(text, line_data, font_analysis):
                heading_candidates.append({
                    'text': text,
                    'page': page_num,
                    'font_size': line_data['avg_font_size'],
                    'is_bold': line_data['is_bold'],
                    'position': line_data['position'],
                    'raw_level': self._determine_raw_level(text, line_data, font_analysis)
                })
        
        return heading_candidates
    
    def _is_potential_heading(self, text: str, line_data: Dict, font_analysis: Dict) -> bool:
        if len(text) < 3 or len(text) > 200:
            return False
        
        if self._is_header_footer_metadata(text):
            return False
        
        if self._is_form_field_for_outline(text):
            return False
        
        if self._is_table_of_contents_entry(text):
            return False
        
        if self._is_repeated_header_footer(text):
            return False
        
        if self._is_body_text(text, line_data, font_analysis):
            return False
        
        # Special handling for numbered text
        numbering_info = self._extract_numbering_pattern(text)
        if numbering_info:
            return self._is_valid_numbered_heading(text, line_data, font_analysis, numbering_info)
        
        # Non-numbered text uses existing logic
        if self._matches_heading_patterns(text):
            return True
        
        if self._has_heading_typography_dynamic(line_data, font_analysis):
            return True
        
        if self._has_heading_position_strict(text, line_data, font_analysis):
            return True
        
        return False
    
    def _is_valid_numbered_heading(self, text: str, line_data: Dict, font_analysis: Dict, numbering_info: Dict) -> bool:
        """Validate that numbered text is actually a heading, not a table row or list item"""
        
        # Additional checks for numbered text to prevent false positives
        
        # 1. Check for table-like patterns (multiple columns, tabular data)
        if self._looks_like_table_row(text):
            return False
        
        # 2. Check for list items in paragraphs (too much following text)
        if self._looks_like_list_item(text):
            return False
        
        # 3. Check for form enumeration (specific form patterns)
        if self._looks_like_form_enumeration(text):
            return False
        
        # 4. For numbered text, require at least ONE of these heading qualities:
        has_heading_quality = (
            self._has_heading_typography_lenient(line_data, font_analysis) or
            self._has_heading_spacing(text, line_data) or
            self._has_heading_formatting(text, line_data) or
            self._has_section_title_characteristics(text)
        )
        
        return has_heading_quality
    
    def _looks_like_table_row(self, text: str) -> bool:
        """Detect if numbered text looks like a table row"""
        
        # Pattern: "1. Name Age Relationship" or "1. John 25 Son"
        # Multiple distinct values that look like columns
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for table header patterns
        table_headers = ['name', 'age', 'date', 'amount', 'description', 'type', 'status', 'id']
        word_lower = [w.lower() for w in words]
        
        # If contains multiple table-like headers
        header_count = sum(1 for header in table_headers if header in word_lower)
        if header_count >= 2:
            return True
        
        # Pattern: number followed by multiple capitalized single words (like table data)
        if len(words) >= 4:
            # Skip the number part
            remaining_words = words[1:] if re.match(r'^\d+\.?', words[0]) else words
            if len(remaining_words) >= 3:
                # Check if looks like tabular data (multiple short capitalized words)
                short_cap_words = [w for w in remaining_words if len(w) <= 10 and w[0].isupper()]
                if len(short_cap_words) >= 3:
                    return True
        
        return False
    
    def _looks_like_list_item(self, text: str) -> bool:
        """Detect if numbered text is a list item rather than a heading"""
        
        # Very long numbered items are likely list items, not headings
        if len(text) > 150:
            return True
        
        # Check for list item patterns (ends with punctuation, contains sentence structure)
        if text.endswith(('.', ',', ';')) and len(text) > 50:
            return True
        
        # Contains too many common sentence words
        sentence_words = ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'by', 'from', 'that', 'which', 'who']
        words = text.lower().split()
        sentence_word_count = sum(1 for word in words if word in sentence_words)
        
        if len(words) > 10 and sentence_word_count > len(words) * 0.3:
            return True
        
        return False
    
    def _looks_like_form_enumeration(self, text: str) -> bool:
        """Detect form field enumerations"""
        
        # Pattern: "1. Name of the applicant:"
        # Pattern: "2. Date of birth:"
        # Pattern: "3. Whether married or single:"
        
        form_patterns = [
            r'^\d+\.\s+(Name|Date|Age|Address|Phone|Email|Whether|Amount|Details?)\s+(of|for)?\s+',
            r'^\d+\.\s+\w+\s+(name|date|number|address|details?)\s*:?\s*$',
            r'^\d+\.\s+(Signature|Declaration|Certification)\s+',
        ]
        
        for pattern in form_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_heading_typography_lenient(self, line_data: Dict, font_analysis: Dict) -> bool:
        """More lenient typography check for numbered headings"""
        
        font_size = line_data['avg_font_size']
        is_bold = line_data['is_bold']
        avg_size = font_analysis['avg_size']
        
        # For numbered headings, be more lenient with font size requirements
        # Require either: larger font OR bold OR both
        if font_size >= avg_size * 1.05:  # Even slightly larger is OK
            return True
        
        if is_bold:  # Bold is often used for numbered headings
            return True
        
        return False
    
    def _has_heading_spacing(self, text: str, line_data: Dict) -> bool:
        """Check if text has spacing characteristics of headings"""
        
        # This would need access to surrounding lines for whitespace analysis
        # For now, use basic checks
        
        # Headings are typically not too long
        if len(text) > 120:
            return False
        
        # Headings typically don't end with sentence punctuation
        if text.strip().endswith(('.', '!', '?')) and len(text) > 30:
            return False
        
        return True
    
    def _has_heading_formatting(self, text: str, line_data: Dict) -> bool:
        """Check formatting characteristics"""
        
        # Center or left alignment is good for headings
        position = line_data.get('position', 'left')
        if position in ['center', 'left']:
            return True
        
        return False
    
    def _has_section_title_characteristics(self, text: str) -> bool:
        """Check if text has characteristics of section titles"""

        # Reasonable length for a section title
        if not (5 <= len(text) <= 100):
            return False

        # Contains meaningful title words
        title_words = [
            'introduction', 'overview', 'background', 'methodology', 'results', 'conclusion',
            'discussion', 'analysis', 'summary', 'abstract', 'references', 'appendix',
            'definition', 'purpose', 'scope', 'requirements', 'implementation', 'testing'
        ]

        text_lower = text.lower()
        if any(word in text_lower for word in title_words):
            return True

        # Title case or proper capitalization
        words = text.split()
        if len(words) >= 2:
            # Check if most words are capitalized (title case)
            capitalized = sum(1 for word in words[1:] if word and word[0].isupper())  # Skip number
            if capitalized >= len(words[1:]) * 0.7:
                return True
        return False
    
    def _is_table_of_contents_entry(self, text: str) -> bool:
        """Detect table of contents entries with page numbers"""
        if re.search(r'\.\s*\d+\s*$', text):
            return True
        
        if re.search(r'^\d+\.\s+.+\.\s*\d+\s*$', text):
            return True
        
        return False
    
    def _is_repeated_header_footer(self, text: str) -> bool:
        """Detect repeated header/footer elements that appear on multiple pages"""
        if (len(text.split()) <= 2 and 
            len(text) < 20 and
            not re.match(r'^\d+\.', text)):
            return True
        
        return False
    
    def _is_body_text(self, text: str, line_data: Dict, font_analysis: Dict) -> bool:
        """Enhanced body text detection - more nuanced filtering"""
        
        if (text.count('.') > 2 or 
            text.count(',') > 4 or 
            len(text.split()) > 20 or
            len(text) > 150):
            return True
        
        body_patterns = [
            r'^[A-Z][a-z].+[.!?]\s+[A-Z][a-z]',
            r'\b(will be|have been|has been|are being|were being)\b',
            r'\b(during|throughout|within|between|among)\s+the\b',
            r'\b(including|such as|for example|specifically)\b',
        ]
        
        sentence_indicators = 0
        for pattern in body_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sentence_indicators += 1
        
        if sentence_indicators >= 2:
            return True
        
        font_size = line_data['avg_font_size']
        avg_size = font_analysis['avg_size']
        
        if font_size < avg_size * 0.9:
            return True
        
        return False
    
    def _has_heading_typography_dynamic(self, line_data: Dict, font_analysis: Dict) -> bool:
        """Dynamic font size analysis based on document characteristics - less strict"""
        font_size = line_data['avg_font_size']
        is_bold = line_data['is_bold']
        avg_size = font_analysis['avg_size']
        max_size = font_analysis['max_size']
        
        size_range = max_size - avg_size
        
        if size_range < 2:
            h1_threshold = avg_size * 1.2
            h2_threshold = avg_size * 1.15
            h3_threshold = avg_size * 1.1
        else:
            h1_threshold = avg_size + (size_range * 0.4)
            h2_threshold = avg_size + (size_range * 0.25)
            h3_threshold = avg_size + (size_range * 0.1)
        
        if font_size >= h1_threshold:
            return True
        
        elif font_size >= h2_threshold or (is_bold and font_size >= avg_size * 1.05):
            return True
        
        elif (font_size >= h3_threshold and is_bold) or (is_bold and font_size >= avg_size):
            return True
        
        return False
    
    def _has_heading_position_strict(self, text: str, line_data: Dict, font_analysis: Dict) -> bool:
        """Position analysis for headings - more permissive"""
        position = line_data['position']
        font_size = line_data['avg_font_size']
        avg_size = font_analysis['avg_size']
        
        if font_size <= avg_size * 1.05:
            return False
        
        if len(text) > 120 or (text.endswith('.') and len(text) > 50):
            return False
        
        if position == 'left':
            if (len(text) < 100 and 
                font_size >= avg_size * 1.08):
                return True
        
        elif position == 'center':
            if (len(text) < 100 and 
                font_size >= avg_size):
                return True
        
        return False
    
    def _matches_heading_patterns(self, text: str) -> bool:
        """Enhanced pattern recognition for headings"""
        h1_patterns = [
            r'^\d+\.\s+[A-Z][^.]*$',
            r'^[A-Z][A-Z\s]{8,}$',
            r'^[A-Z][a-z\s]{10,}$',
            r'^\d+\s+[A-Z][a-z\s]{5,}$',
        ]
        
        h2_patterns = [
            r'^\d+\.\d+\s+[A-Z][^.]*$',
            r'^[A-Z][a-z\s]{5,}:\s*$',
            r'^\d+\.\d+\s+[A-Z][a-z\s]{3,}$',
        ]
        
        h3_patterns = [
            r'^\d+\.\d+\.\d+\s+[A-Z][^.]*$',
            r'^[A-Z][a-z\s]{3,}:\s*$',
            r'^[A-Z][a-z\s]*[A-Z][a-z\s]*:\s*$',
        ]
        
        h4_patterns = [
            r'^For\s+each\s+[A-Z][a-z\s]*it\s+could\s+mean:\s*$',
            r'^[A-Z][a-z\s]{15,}:\s*$',
        ]
        
        all_patterns = h1_patterns + h2_patterns + h3_patterns + h4_patterns
        
        for pattern in all_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        return False
    
    
    
    
    def _is_form_field_for_outline(self, text: str) -> bool:
        patterns = [
            r'^\d+\.\s+[A-Z][a-z\s]*$',
            r'^\d+\.\s+[A-Z][A-Z\s+]+$',
            r'^[A-Z][a-z\s]*\s+[A-Z][a-z\s]*\s+[A-Z][a-z\s]*$',
            r'^S\.No\s+Name\s+Age',
            r'^\d+\.\s*$',
            r'^Name\s+Age\s+Relationship',
            r'Signature\s+of\s+Government\s+Servant',
            r'LTC\s+is\s+to\s+be\s+availed',
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    
    def _looks_like_body_text(self, text: str) -> bool:
        return (
            text.count('.') > 2 or 
            len(text.split()) > 20 or
            text.count(',') > 3
        )
    
    def _determine_raw_level(self, text: str, line_data: Dict, font_analysis: Dict) -> int:
        """Enhanced heading level determination using visual hierarchy"""
        
        # Priority 1: Section numbering (most reliable)
        section_level = self._get_level_from_section_numbering(text)
        if section_level:
            return section_level
        
        # Priority 2: Visual patterns
        pattern_level = self._get_level_from_visual_patterns(text, line_data, font_analysis)
        if pattern_level:
            return pattern_level
        
        # Priority 3: Font size analysis
        font_level = self._get_level_from_font_size_refined(line_data, font_analysis)
        
        return min(font_level, 6)
    
    def _get_level_from_section_numbering(self, text: str) -> Optional[int]:
        """Determine relative heading level from section numbering patterns"""
        
        text = text.strip()
        
        # Extract numbering pattern and return depth (not absolute level)
        numbering_info = self._extract_numbering_pattern(text)
        if numbering_info:
            return numbering_info['depth']
        
        return None
    
    def _extract_numbering_pattern(self, text: str) -> Optional[Dict]:
        """Extract numbering pattern and determine its hierarchical depth"""
        
        text = text.strip()
        
        # Pattern 1: Multi-level decimal numbering (1.1, 1.1.1, 2.2.1, etc.) - MUST come first
        multi_decimal_match = re.match(r'^(\d+\.\d+(?:\.\d+)*)\s+', text)
        if multi_decimal_match:
            number_part = multi_decimal_match.group(1)
            depth = number_part.count('.') + 1  # 1.1=depth2, 1.1.1=depth3, etc.
            return {
                'type': 'decimal',
                'pattern': number_part,
                'depth': depth,
                'text': text
            }
        
        # Pattern 2: Simple numbering with dot (1., 2., 3., etc.)
        simple_dot_match = re.match(r'^(\d+)\.\s+', text)
        if simple_dot_match:
            return {
                'type': 'decimal',
                'pattern': simple_dot_match.group(1),
                'depth': 1,  # Single number = depth 1
                'text': text
            }
        
        # Pattern 3: Simple numbering without dot (1 Title, 2 Title, etc.)
        simple_match = re.match(r'^(\d+)\s+[A-Z]', text)
        if simple_match:
            return {
                'type': 'decimal',
                'pattern': simple_match.group(1),
                'depth': 1,
                'text': text
            }
        
        # Pattern 4: Roman numerals (I, II, III, IV, V)
        roman_match = re.match(r'^([IVX]+)\.\s+', text)
        if roman_match:
            return {
                'type': 'roman',
                'pattern': roman_match.group(1),
                'depth': 1,
                'text': text
            }
        
        # Pattern 5: Letter numbering (A, B, C or a, b, c)
        letter_match = re.match(r'^([A-Za-z])\.\s+', text)
        if letter_match:
            letter = letter_match.group(1)
            return {
                'type': 'letter',
                'pattern': letter,
                'depth': 2 if letter.isupper() else 3,  # A.=depth2, a.=depth3
                'text': text
            }
        
        # Pattern 6: Mixed alphanumeric (1.A, 1.a, etc.)
        mixed_match = re.match(r'^(\d+\.[A-Za-z])\s+', text)
        if mixed_match:
            return {
                'type': 'mixed',
                'pattern': mixed_match.group(1),
                'depth': 3,
                'text': text
            }
        
        # Pattern 7: Parenthetical numbering ((1), (a), etc.)
        paren_match = re.match(r'^\(([0-9a-zA-Z]+)\)\s+', text)
        if paren_match:
            content = paren_match.group(1)
            return {
                'type': 'parenthetical',
                'pattern': f"({content})",
                'depth': 3 if content.isdigit() else 4,
                'text': text
            }
        
        return None
    
    def _get_level_from_visual_patterns(self, text: str, line_data: Dict, font_analysis: Dict) -> Optional[int]:
        """Determine heading level from visual hierarchy and patterns"""
        text = text.strip()
        font_size = line_data['avg_font_size']
        is_bold = line_data['is_bold']
        position = line_data['position']
        avg_size = font_analysis['avg_size']
        max_size = font_analysis['max_size']
        
        if text.endswith(':') and len(text) <= 25 and len(text.split()) <= 3:
            return 3
        
        if text.endswith('?') or re.search(r'\bcould\s+mean\b', text, re.IGNORECASE):
            return 4
        
        if position == 'center' and font_size >= avg_size * 1.3:
            return 1
        
        if (font_size >= max_size * 0.8 or 
            (position == 'center' and font_size >= avg_size * 1.2)):
            return 1
        
        size_ratio = font_size / avg_size
        
        if size_ratio >= 1.3:
            return 1
        elif size_ratio >= 1.2:
            return 2
        elif size_ratio >= 1.1 or is_bold:
            return 3
        else:
            return 4
    
    def _get_level_from_font_size_refined(self, line_data: Dict, font_analysis: Dict) -> int:
        """Refined font size analysis with better thresholds"""
        font_size = line_data['avg_font_size']
        is_bold = line_data['is_bold']
        avg_size = font_analysis['avg_size']
        max_size = font_analysis['max_size']
        
        size_ratio = font_size / avg_size
        
        if size_ratio >= 1.4 or font_size >= max_size * 0.9:
            return 1
        
        elif size_ratio >= 1.25:
            return 2
        
        elif size_ratio >= 1.15:
            return 3 if is_bold else 4
        
        elif size_ratio >= 1.08 and is_bold:
            return 3
        
        else:
            return 4
    
    
    
    def _classify_and_clean_headings(self, headings: List[Dict]) -> List[Dict]:
        if not headings:
            return []
        
        cleaned_headings = []
        
        for heading in headings:
            raw_level = heading.get('raw_level', 1)
            level = min(raw_level, 6)
            
            cleaned_text = self._clean_heading_text(heading['text'])
            
            if cleaned_text:
                cleaned_headings.append({
                    'level': f"H{level}",
                    'text': cleaned_text,
                    'page': heading['page']
                })
        
        return cleaned_headings
    
    def _clean_heading_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        
        text = re.sub(r':\s*$', '', text)
        
        return text
    
    def _filter_false_positive_headers(self, headings: List[Dict], all_lines_data: List[Dict]) -> List[Dict]:
        """Apply strict isolation filtering to remove false positive headers"""
        
        if not headings or not all_lines_data:
            return headings
        
        # Calculate percentile-based spacing thresholds
        spacing_stats = self._calculate_spacing_percentiles(all_lines_data)
        
        validated_headings = []
        
        for heading in headings:
            # Find the line in all_lines_data that corresponds to this heading
            line_info = self._find_heading_line_context(heading, all_lines_data)
            
            if line_info and self._passes_strict_isolation_test(heading, line_info, spacing_stats):
                validated_headings.append(heading)
        
        return validated_headings
    
    def _calculate_spacing_percentiles(self, all_lines_data: List[Dict]) -> Dict:
        """Calculate spacing statistics for strict isolation testing"""
        
        spacings = []
        
        # Group lines by page for proper spacing calculation
        pages = {}
        for line in all_lines_data:
            page = line.get('page', 0)
            if page not in pages:
                pages[page] = []
            pages[page].append(line)
        
        # Calculate spacing between consecutive lines within each page
        for page_lines in pages.values():
            for i in range(len(page_lines) - 1):
                current = page_lines[i]
                next_line = page_lines[i + 1]
                
                current_y = current.get('y_position', i * 14)
                current_height = current.get('height', 14)
                next_y = next_line.get('y_position', (i + 1) * 14)
                
                spacing = abs(next_y - (current_y + current_height))
                spacings.append(spacing)
        
        if not spacings:
            return {'p50': 5, 'p75': 10, 'p90': 20, 'p95': 30}
        
        spacings.sort()
        n = len(spacings)
        
        return {
            'p50': spacings[int(n * 0.5)],   # Median
            'p75': spacings[int(n * 0.75)],  # 75th percentile
            'p90': spacings[int(n * 0.90)],  # 90th percentile
            'p95': spacings[int(n * 0.95)]   # 95th percentile
        }
    
    def _find_heading_line_context(self, heading: Dict, all_lines_data: List[Dict]) -> Optional[Dict]:
        """Find the line context for a heading in the full document lines"""
        
        heading_text = heading['text'].strip()
        heading_page = heading.get('page', 0)
        
        # Find lines on the same page
        page_lines = [line for line in all_lines_data if line.get('page', 0) == heading_page]
        
        # Find the line that matches this heading text
        for i, line in enumerate(page_lines):
            if line['text'].strip() == heading_text:
                return {
                    'line_data': line,
                    'line_index': i,
                    'page_lines': page_lines,
                    'global_index': all_lines_data.index(line)
                }
        
        return None
    
    def _passes_strict_isolation_test(self, heading: Dict, line_info: Dict, spacing_stats: Dict) -> bool:
        """Apply strict isolation test to validate a heading"""
        
        text = heading['text'].strip()
        line_data = line_info['line_data']
        page_lines = line_info['page_lines']
        line_index = line_info['line_index']
        
        # BYPASS: Strong header evidence - skip isolation test
        if self._has_strong_header_evidence(text, line_data):
            return True
        
        # 1. Basic content quality checks
        if not self._passes_content_quality_check(text):
            return False
        
        # 2. Check for paragraph continuation (major cause of false positives)
        if self._is_paragraph_continuation(text, page_lines, line_index):
            return False
        
        # 3. Measure actual spatial isolation
        spacing_above = self._measure_line_spacing_above(page_lines, line_index)
        spacing_below = self._measure_line_spacing_below(page_lines, line_index)
        
        # 4. Apply spacing requirements (more lenient now)
        return self._meets_spacing_requirements(text, line_data, spacing_above, spacing_below, spacing_stats)
    
    def _has_strong_header_evidence(self, text: str, line_data: Dict) -> bool:
        """Check if text has strong evidence of being a header (bypass isolation test)"""
        
        # 1. Numbered patterns - these are usually reliable headers
        if self._extract_numbering_pattern(text):
            return True
        
        # 2. Bold text with reasonable size
        if (line_data.get('is_bold', False) and 
            line_data.get('avg_font_size', 12) >= 12):
            return True
        
        # 3. Very short text (likely section markers) that's prominent
        if (len(text) <= 15 and 
            (line_data.get('is_bold', False) or 
             line_data.get('avg_font_size', 12) >= 13)):
            return True
        
        # 4. Clear header patterns (existing pattern matches)
        if self._matches_heading_patterns(text):
            return True
        
        # 5. CJK text with parenthetical English (common in multilingual docs)
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+.*\([A-Za-z\s]+\)', text):
            return True
        
        return False
    
    def _passes_content_quality_check(self, text: str) -> bool:
        """Check if text has header-like content quality"""
        
        # Length check
        if len(text) < 2 or len(text) > 120:
            return False
        
        # Should not look like obvious paragraph text
        if (text.count('.') > 2 or 
            text.count(',') > 4 or 
            len(text.split()) > 20):
            return False
        
        # Should not end with sentence-ending punctuation (except colons)
        if text.rstrip().endswith(('.', '!', '?')) and len(text) > 30:
            return False
        
        # Should not contain typical paragraph continuation words
        paragraph_words = ['therefore', 'however', 'moreover', 'furthermore', 'consequently', 'additionally']
        text_lower = text.lower()
        if any(word in text_lower for word in paragraph_words):
            return False
        
        return True
    
    def _is_paragraph_continuation(self, text: str, page_lines: List[Dict], line_index: int) -> bool:
        """Check if this text is a continuation of a paragraph"""
        
        # Check previous line for paragraph flow
        if line_index > 0:
            prev_text = page_lines[line_index - 1]['text'].strip()
            
            # Previous line doesn't end with sentence punctuation - likely continuation
            if prev_text and not prev_text.endswith(('.', '!', '?', ':')):
                return True
            
            # Text starts with lowercase - likely continuation
            if text and text[0].islower():
                return True
            
            # Check for natural sentence flow
            if self._flows_naturally_from_previous(text, prev_text):
                return True
        
        # Check next line for paragraph flow
        if line_index < len(page_lines) - 1:
            next_text = page_lines[line_index + 1]['text'].strip()
            
            # Current line doesn't end with punctuation but next continues - likely paragraph
            if (not text.endswith(('.', '!', '?', ':')) and 
                next_text and next_text[0].islower()):
                return True
        
        return False
    
    def _flows_naturally_from_previous(self, current_text: str, prev_text: str) -> bool:
        """Check if current text flows naturally from previous text"""
        
        # Common continuation indicators
        flow_starters = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor', 'as', 'if', 'when', 'while', 'because']
        
        current_lower = current_text.lower().strip()
        for starter in flow_starters:
            if current_lower.startswith(starter + ' '):
                return True
        
        # Check for relative clauses or dependent clauses
        if current_lower.startswith(('that ', 'which ', 'who ', 'where ', 'when ', 'why ', 'how ')):
            return True
        
        return False
    
    def _meets_spacing_requirements(self, text: str, line_data: Dict, spacing_above: float, spacing_below: float, spacing_stats: Dict) -> bool:
        """Check if spacing meets requirements for headers (more lenient now)"""
        
        # Get typography strength
        font_size = line_data.get('avg_font_size', 12)
        is_bold = line_data.get('is_bold', False)
        
        # Calculate typography score
        typography_score = 0
        if is_bold:
            typography_score += 2
        if font_size >= 14:
            typography_score += 1
        elif font_size >= 13:
            typography_score += 0.5
        
        # More lenient spacing requirements
        if typography_score >= 2:  # Strong typography (bold + large)
            # Very lenient with spacing
            min_spacing = spacing_stats['p50']  # Median
            required_isolation = min_spacing
        elif typography_score >= 1:  # Medium typography
            # Moderate spacing requirements
            min_spacing = spacing_stats['p75']  # 75th percentile
            required_isolation = min_spacing
        else:  # Weak typography
            # Moderate spacing requirements (less strict than before)
            min_spacing = spacing_stats['p90']  # 90th percentile
            required_isolation = min_spacing
        
        # Check if at least one side meets requirements
        meets_above = spacing_above >= required_isolation
        meets_below = spacing_below >= required_isolation
        
        # For very short text (likely standalone headers), require both sides
        if len(text) <= 10:
            return meets_above and meets_below
        
        # For longer text, just need one side OR both sides are above median
        return (meets_above or meets_below) or (
            spacing_above >= spacing_stats['p50'] and 
            spacing_below >= spacing_stats['p50']
        )
    
    def _measure_line_spacing_above(self, page_lines: List[Dict], line_index: int) -> float:
        """Measure spacing above the current line"""
        if line_index == 0:
            return float('inf')  # Top of page
        
        current_line = page_lines[line_index]
        prev_line = page_lines[line_index - 1]
        
        current_y = current_line.get('y_position', line_index * 14)
        prev_y = prev_line.get('y_position', (line_index - 1) * 14)
        prev_height = prev_line.get('height', 14)
        
        return abs(current_y - (prev_y + prev_height))
    
    def _measure_line_spacing_below(self, page_lines: List[Dict], line_index: int) -> float:
        """Measure spacing below the current line"""
        if line_index >= len(page_lines) - 1:
            return float('inf')  # Bottom of page
        
        current_line = page_lines[line_index]
        next_line = page_lines[line_index + 1]
        
        current_y = current_line.get('y_position', line_index * 14)
        current_height = current_line.get('height', 14)
        next_y = next_line.get('y_position', (line_index + 1) * 14)
        
        return abs(next_y - (current_y + current_height))
    
    def _filter_to_major_headings_only(self, headings: List[Dict]) -> List[Dict]:
        """Filter to only allow H1, H2, H3 level headers (remove H4, H5, H6)"""
        
        major_headings = []
        
        for heading in headings:
            text = heading['text'].strip()
            
            # Apply stricter criteria for what qualifies as a major heading
            if self._qualifies_as_major_heading(text, heading):
                major_headings.append(heading)
        
        return major_headings
    
    def _qualifies_as_major_heading(self, text: str, heading: Dict) -> bool:
        """Check if a heading qualifies as H1, H2, or H3 level"""
        
        # 1. Reject obvious list items and descriptive text
        if self._is_list_item_or_description(text):
            return False
        
        # 2. Check for strong heading indicators
        has_strong_indicators = self._has_major_heading_indicators(text, heading)
        
        # 3. Apply length and content quality checks
        content_quality = self._has_major_heading_content_quality(text)
        
        return has_strong_indicators and content_quality
    
    def _is_list_item_or_description(self, text: str) -> bool:
        """Identify text that's clearly a list item or descriptive text"""
        
        # Long descriptive sentences
        if (len(text) > 80 and 
            (text.count(' ') > 12 or text.count(',') > 2)):
            return True
        
        # Starts with list indicators but is too long/descriptive
        if (re.match(r'^\d+\.\s+', text) and 
            len(text) > 60 and 
            any(word in text.lower() for word in ['who', 'that', 'which', 'have', 'been', 'will', 'are', 'would'])):
            return True
        
        # Descriptive phrases that start with common patterns
        descriptive_starters = [
            'the following', 'the above', 'each of the', 'all of the',
            'this section', 'this chapter', 'this document', 'in order to',
            'it is important', 'please note', 'as shown in'
        ]
        
        text_lower = text.lower()
        if any(text_lower.startswith(starter) for starter in descriptive_starters):
            return True
        
        # Instructions or explanatory text
        instruction_patterns = [
            r'must be \w+', r'should be \w+', r'will be \w+',
            r'are required to', r'are expected to', r'are used in'
        ]
        
        if any(re.search(pattern, text_lower) for pattern in instruction_patterns):
            return True
        
        return False
    
    def _has_major_heading_indicators(self, text: str, heading: Dict) -> bool:
        """Check for indicators of H1/H2/H3 level headers"""
        
        # 1. Clear numbered section patterns (1., 1.1, 1.1.1 but not deep nesting)
        numbering_info = self._extract_numbering_pattern(text)
        if numbering_info and numbering_info['depth'] <= 3:
            # Additional check: numbered text should be concise
            if len(text) <= 50 or (len(text) <= 80 and text.count(' ') <= 8):
                return True
        
        # 2. Strong typography (bold + larger font)
        is_bold = heading.get('is_bold', False)
        font_size = heading.get('font_size', 12)
        
        if is_bold and font_size >= 13:
            return True
        
        # 3. Very large font (even without bold)
        if font_size >= 16:
            return True
        
        # 4. Short, capitalized text that looks like section titles
        if (len(text) <= 30 and 
            (text.isupper() or text.istitle()) and
            not text.endswith(('.', ',', ';'))):
            return True
        
        return False
    
    def _has_major_heading_content_quality(self, text: str) -> bool:
        """Check content quality for major headers"""
        
        # Length should be reasonable for a header
        if len(text) < 3 or len(text) > 100:
            return False
        
        # Should not have too many function words (articles, prepositions)
        function_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()
        
        if len(words) > 10:
            function_word_ratio = sum(1 for word in words if word in function_words) / len(words)
            if function_word_ratio > 0.4:  # Too many function words - likely descriptive text
                return False
        
        # Should not end with sentence punctuation
        if text.rstrip().endswith(('.', '!', '?')) and len(text) > 20:
            return False
        
        # Should not contain complex sentence structures
        complex_indicators = ['therefore', 'however', 'moreover', 'furthermore', 'consequently', 
                            'nevertheless', 'although', 'whereas', 'provided that']
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in complex_indicators):
            return False
        
        return True
    
    def _exclude_title_from_headings(self, headings: List[Dict], title_info: Dict) -> List[Dict]:
        """Remove title text from heading candidates to avoid duplication"""
        
        title_text = title_info['title'].lower().strip()
        title_metadata = title_info.get('title_metadata', {})
        
        filtered_headings = []
        
        for heading in headings:
            heading_text = heading['text'].lower().strip()
            
            # Exact match
            if heading_text == title_text:
                continue
            
            # Fuzzy match (85% similarity)
            if self._text_similarity(heading_text, title_text) > 0.85:
                continue
            
            # For multiline titles, check against individual lines
            if (title_metadata and title_metadata.get('group_lines')):
                is_part_of_multiline_title = False
                for title_line in title_metadata['group_lines']:
                    title_line_clean = title_line.lower().strip()
                    if (heading_text == title_line_clean or 
                        self._text_similarity(heading_text, title_line_clean) > 0.85):
                        is_part_of_multiline_title = True
                        break
                
                if is_part_of_multiline_title:
                    continue
            
            # Position + font size match (likely same element)
            if (title_metadata and 
                heading.get('page', 0) == 0):
                
                # For multiline titles, check against spatial bounds
                if title_metadata.get('y_position_end'):
                    title_start_y = title_metadata.get('y_position', 0)
                    title_end_y = title_metadata.get('y_position_end', 0)
                    heading_y = heading.get('y_position', 0)
                    
                    # Check if heading falls within title's vertical bounds
                    if (title_start_y <= heading_y <= title_end_y + 20 and
                        abs(heading.get('font_size', 0) - title_metadata.get('font_size', 0)) < 2):
                        continue
                
                # Original single-line check
                elif (abs(heading.get('font_size', 0) - title_metadata.get('font_size', 0)) < 1 and
                      abs(heading.get('y_position', 0) - title_metadata.get('y_position', 0)) < 20):
                    continue
            
            filtered_headings.append(heading)
        
        return filtered_headings
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity ratio"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_title_aware_hierarchy(self, headings: List[Dict], title_info: Dict, all_lines_data: List[Dict]) -> List[Dict]:
        """Apply font hierarchy that's aware of the document title"""
        
        if not headings:
            return headings
        
        # Step 1: Apply relative section numbering hierarchy (highest priority)
        headings = self._apply_relative_section_numbering(headings)
        
        # Step 2: For non-numbered headings, use font-based hierarchy
        title_font_size = 0
        if title_info.get('title_metadata'):
            title_font_size = title_info['title_metadata'].get('font_size', 0)
        
        font_hierarchy = self._analyze_heading_font_hierarchy(headings, title_font_size)
        headings = self._assign_relative_heading_levels(headings, font_hierarchy)
        
        return headings
    
    def _apply_relative_section_numbering(self, headings: List[Dict]) -> List[Dict]:
        """Apply relative section numbering hierarchy to headings"""
        
        # Step 1: Extract all numbered headings with their patterns
        numbered_headings = []
        for heading in headings:
            numbering_info = self._extract_numbering_pattern(heading['text'])
            if numbering_info:
                heading['numbering_info'] = numbering_info
                numbered_headings.append(heading)
        
        if not numbered_headings:
            return headings  # No numbered headings to process
        
        # Step 2: Analyze the numbering hierarchy in the document
        depth_mapping = self._create_relative_depth_mapping(numbered_headings)
        
        # Step 3: Apply the relative mapping to numbered headings
        for heading in headings:
            if 'numbering_info' in heading:
                depth = heading['numbering_info']['depth']
                if depth in depth_mapping:
                    heading['raw_level'] = depth_mapping[depth]
                else:
                    heading['raw_level'] = min(depth, 6)  # Fallback
        
        return headings
    
    def _create_relative_depth_mapping(self, numbered_headings: List[Dict]) -> Dict[int, int]:
        """Create mapping from numbering depth to relative heading levels"""
        
        # Collect all depths found in the document
        depths_found = set()
        for heading in numbered_headings:
            depths_found.add(heading['numbering_info']['depth'])
        
        # Sort depths to create relative mapping
        sorted_depths = sorted(depths_found)
        
        # Create mapping: smallest depth → H1, next → H2, etc.
        depth_mapping = {}
        for i, depth in enumerate(sorted_depths):
            # Map to H1, H2, H3, etc. but cap at H6
            heading_level = min(i + 1, 6)
            depth_mapping[depth] = heading_level
            
        return depth_mapping
    
    def _analyze_heading_font_hierarchy(self, headings: List[Dict], title_font_size: float) -> Dict:
        """Analyze font hierarchy among headings, considering title context"""
        
        # Calculate prominence scores only for non-numbered headings
        prominence_scores = []
        non_numbered_count = 0
        
        for heading in headings:
            # Skip numbered headings - they have their own hierarchy
            if 'numbering_info' in heading:
                continue
                
            non_numbered_count += 1
            font_size = heading.get('font_size', 12)
            is_bold = heading.get('is_bold', False)
            
            # Base prominence
            prominence = font_size
            
            # Bold bonus
            if is_bold:
                prominence += font_size * 0.2
            
            # Position bonus
            if heading.get('position') == 'center':
                prominence += font_size * 0.1
            
            # Avoid promoting headings too close to title font size
            if title_font_size > 0 and font_size >= title_font_size * 0.9:
                prominence *= 0.8
            
            prominence_scores.append(prominence)
        
        # Create tiers from unique prominence levels
        unique_scores = sorted(set(prominence_scores), reverse=True) if prominence_scores else []
        
        # Group similar scores into tiers (within 1pt difference)
        tiers = []
        for score in unique_scores:
            if not tiers or score < tiers[-1] - 1.0:
                tiers.append(score)
        
        return {
            'prominence_tiers': tiers[:6],
            'title_font_size': title_font_size,
            'heading_count': non_numbered_count
        }
    
    def _assign_relative_heading_levels(self, headings: List[Dict], font_hierarchy: Dict) -> List[Dict]:
        """Assign H1-H6 levels based on relative font prominence in document"""
        
        prominence_tiers = font_hierarchy['prominence_tiers']
        
        for heading in headings:
            # Skip numbered headings - they already have levels assigned
            if 'numbering_info' in heading and 'raw_level' in heading:
                continue
                
            # Calculate heading's prominence score
            font_size = heading.get('font_size', 12)
            is_bold = heading.get('is_bold', False)
            
            base_score = font_size
            bold_bonus = font_size * 0.2 if is_bold else 0
            position_bonus = font_size * 0.1 if heading.get('position') == 'center' else 0
            heading_prominence = base_score + bold_bonus + position_bonus
            
            # Find which tier this heading belongs to
            level = 6
            for i, tier_score in enumerate(prominence_tiers):
                if abs(heading_prominence - tier_score) <= 1.0:
                    level = i + 1
                    break
                elif heading_prominence > tier_score:
                    level = i + 1
                    break
            
            # Apply contextual adjustments
            level = self._apply_contextual_level_adjustments(heading, level)
            
            heading['raw_level'] = min(level, 6)
            heading['prominence_score'] = heading_prominence
        
        return headings
    
    def _apply_contextual_level_adjustments(self, heading: Dict, current_level: int) -> int:
        """Apply contextual rules to refine levels"""
        
        text = heading['text']
        
        # Numbered hierarchy (1.2.3 structure)
        if re.match(r'^\d+\.\d+\.\d+', text):
            suggested_level = text.count('.')
            current_level = max(current_level, suggested_level)
        
        # Position boost for center-aligned
        if heading.get('position') == 'center' and current_level > 1:
            current_level = max(1, current_level - 1)
        
        # ALL CAPS boost (but cap at H2)
        if text.isupper() and len(text) > 3:
            current_level = max(1, min(current_level, 2))
        
        # Colon endings suggest subsections
        if text.endswith(':') and current_level < 3:
            current_level = max(current_level, 3)
        
        return current_level
    
    def _extract_lines_with_metadata(self, chars: List[Dict]) -> List[Dict]:
        if not chars:
            return []
        
        chars_sorted = sorted(chars, key=lambda x: (x.get('top', 0), x.get('x0', 0)))
        
        lines = []
        current_line_chars = []
        current_top = None
        tolerance = 3
        
        for char in chars_sorted:
            char_top = char.get('top', 0)
            
            if current_top is None or abs(char_top - current_top) <= tolerance:
                current_line_chars.append(char)
                current_top = char_top
            else:
                if current_line_chars:
                    lines.append(self._process_line_chars(current_line_chars))
                current_line_chars = [char]
                current_top = char_top
        
        if current_line_chars:
            lines.append(self._process_line_chars(current_line_chars))
        
        return [line for line in lines if line['text'].strip()]
    
    def _process_line_chars(self, chars: List[Dict]) -> Dict:
        if not chars:
            return {'text': '', 'avg_font_size': 12, 'is_bold': False, 'position': 'left'}
        
        chars_sorted = sorted(chars, key=lambda x: x.get('x0', 0))
        text_parts = []
        prev_x1 = None
        prev_char = None
        
        for char in chars_sorted:
            char_text = char.get('text', '')
            char_x0 = char.get('x0', 0)
            char_x1 = char.get('x1', char_x0)
            
            if (prev_char and 
                abs(char_x0 - prev_char.get('x0', 0)) < 1 and 
                char_text == prev_char.get('text', '')):
                continue
            
            if prev_x1 is not None and char_x0 - prev_x1 > 3:
                text_parts.append(' ')
            
            text_parts.append(char_text)
            prev_x1 = char_x1
            prev_char = char
        
        text = ''.join(text_parts).strip()
        
        text = self._clean_duplicate_chars(text)
        
        font_sizes = [char.get('size', 12) for char in chars if char.get('text', '').strip()]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        
        bold_chars = sum(1 for char in chars if 'bold' in char.get('fontname', '').lower())
        is_bold = bold_chars > len(chars) * 0.5
        
        left_positions = [char.get('x0', 0) for char in chars]
        avg_left = sum(left_positions) / len(left_positions) if left_positions else 0
        
        # Determine position based on relative placement (more flexible)
        # Get page width from character positions
        all_x_positions = [char.get('x0', 0) for char in chars] + [char.get('x1', 0) for char in chars]
        page_width = max(all_x_positions) - min(all_x_positions) if all_x_positions else 600
        
        # Use proportional thresholds instead of hardcoded values
        left_threshold = page_width * 0.15   # 15% from left = left-aligned
        right_threshold = page_width * 0.70  # 70% from left = right-aligned
        
        if avg_left < left_threshold:
            position = 'left'
        elif avg_left > right_threshold:
            position = 'right'
        else:
            position = 'center'
        
        # Get y_position (top of the line)
        y_positions = [char.get('top', 0) for char in chars]
        avg_y = sum(y_positions) / len(y_positions) if y_positions else 0
        
        # Get line height
        heights = [char.get('height', 14) for char in chars if char.get('height')]
        avg_height = sum(heights) / len(heights) if heights else 14
        
        return {
            'text': text,
            'avg_font_size': avg_font_size,
            'is_bold': is_bold,
            'position': position,
            'y_position': avg_y,
            'height': avg_height
        }
    
    def _analyze_fonts(self, chars: List[Dict]) -> Dict:
        font_sizes = []
        bold_count = 0
        total_chars = 0
        
        for char in chars:
            if char.get('text', '').strip():
                font_sizes.append(char.get('size', 0))
                if 'bold' in char.get('fontname', '').lower():
                    bold_count += 1
                total_chars += 1
        
        if not font_sizes:
            return {'avg_size': 12, 'max_size': 12, 'common_sizes': [12]}
        
        avg_size = sum(font_sizes) / len(font_sizes)
        max_size = max(font_sizes)
        
        return {
            'avg_size': avg_size,
            'max_size': max_size,
            'bold_ratio': bold_count / total_chars if total_chars > 0 else 0
        }
    
    def close(self):
        """Closes the PDF file object."""
        if self.pdf:
            self.pdf.close()