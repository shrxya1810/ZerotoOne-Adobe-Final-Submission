import re
from typing import List, Dict, Optional


class HierarchyEnhancer:
    """Enhanced hierarchy detection using sequential and spatial context analysis"""
    
    def __init__(self):
        # Tunable parameters
        self.SEQUENTIAL_THRESHOLD = 50  # Max distance between sequential headers (px)
        self.MIN_ABOVE_SPACING = 20     # Minimum whitespace above header (px)
        self.MIN_BELOW_SPACING = 10     # Minimum whitespace below header (px)
        self.SPATIAL_HEADER_THRESHOLD = 0.6  # Confidence threshold for spatial detection
        self.LARGE_WHITESPACE_THRESHOLD = 40  # Threshold for major section breaks (px)
        self.AVERAGE_LINE_HEIGHT = 14   # Average line height for calculations
    
    def enhance_headings(self, headings: List[Dict], lines_data: List[Dict]) -> List[Dict]:
        """Main method to enhance heading detection"""
        
        # Step 1: Detect spatial context headers
        spatial_headings = self._detect_spatial_context_headers(lines_data)
        
        # Step 2: Merge with existing headings
        all_headings = self._merge_heading_candidates(headings, spatial_headings)
        
        # Step 3: Apply sequential hierarchy logic
        all_headings = self._detect_sequential_header_groups(all_headings)
        
        # Step 4: Resolve conflicts and finalize
        final_headings = self._resolve_heading_conflicts(all_headings)
        
        return final_headings
    
    def _detect_spatial_context_headers(self, lines_data: List[Dict]) -> List[Dict]:
        """Detect headers based on spatial context and following body text"""
        
        spatial_headers = []
        
        for i, line in enumerate(lines_data):
            if self._has_header_spatial_pattern(line, lines_data, i):
                confidence = self._calculate_spatial_confidence(line, lines_data, i)
                
                if confidence > self.SPATIAL_HEADER_THRESHOLD:
                    spatial_headers.append({
                        'text': line['text'],
                        'page': line.get('page', 0),
                        'font_size': line.get('avg_font_size', 12),
                        'is_bold': line.get('is_bold', False),
                        'position': line.get('position', 'left'),
                        'confidence': confidence,
                        'detection_method': 'spatial_context',
                        'y_position': line.get('y_position', i * self.AVERAGE_LINE_HEIGHT),
                        'raw_level': self._infer_level_from_context(line, lines_data, i)
                    })
        
        return spatial_headers
    
    def _has_header_spatial_pattern(self, line: Dict, all_lines: List[Dict], index: int) -> bool:
        """Check if line has spatial characteristics of a header"""
        
        text = line['text'].strip()
        
        # Basic text filters
        if (len(text) < 3 or len(text) > 200 or 
            self._is_obviously_body_text(text) or
            self._is_metadata_or_form(text)):
            return False
        
        # Check spatial characteristics
        whitespace_above = self._measure_whitespace_above(line, all_lines, index)
        whitespace_below = self._measure_whitespace_below(line, all_lines, index)
        has_following_body = self._has_following_body_text(line, all_lines, index)
        text_characteristics = self._check_header_text_characteristics(text)
        
        return (whitespace_above >= self.MIN_ABOVE_SPACING and
                whitespace_below >= self.MIN_BELOW_SPACING and
                whitespace_above > whitespace_below and
                has_following_body and
                text_characteristics)
    
    def _measure_whitespace_above(self, line: Dict, all_lines: List[Dict], index: int) -> float:
        """Measure vertical whitespace above a line"""
        if index == 0:
            return float('inf')  # Top of page
        
        prev_line = all_lines[index - 1]
        
        # Calculate vertical gap
        current_y = line.get('y_position', index * self.AVERAGE_LINE_HEIGHT)
        prev_y = prev_line.get('y_position', (index - 1) * self.AVERAGE_LINE_HEIGHT)
        prev_height = prev_line.get('height', self.AVERAGE_LINE_HEIGHT)
        
        gap = abs(current_y - (prev_y + prev_height))
        
        # Check for empty lines above
        empty_lines_above = self._count_empty_lines_above(all_lines, index)
        
        return gap + (empty_lines_above * self.AVERAGE_LINE_HEIGHT)
    
    def _measure_whitespace_below(self, line: Dict, all_lines: List[Dict], index: int) -> float:
        """Measure vertical whitespace below a line"""
        if index >= len(all_lines) - 1:
            return float('inf')  # Bottom of page
        
        next_line = all_lines[index + 1]
        
        current_y = line.get('y_position', index * self.AVERAGE_LINE_HEIGHT)
        current_height = line.get('height', self.AVERAGE_LINE_HEIGHT)
        next_y = next_line.get('y_position', (index + 1) * self.AVERAGE_LINE_HEIGHT)
        
        gap = abs(next_y - (current_y + current_height))
        
        return gap
    
    def _count_empty_lines_above(self, all_lines: List[Dict], index: int) -> int:
        """Count consecutive empty lines above current line"""
        count = 0
        for i in range(index - 1, -1, -1):
            if not all_lines[i]['text'].strip():
                count += 1
            else:
                break
        return count
    
    def _has_following_body_text(self, line: Dict, all_lines: List[Dict], index: int) -> bool:
        """Check if header is followed by body text within next few lines"""
        
        for i in range(index + 1, min(index + 6, len(all_lines))):
            next_line = all_lines[i]
            text = next_line['text'].strip()
            
            if not text:  # Skip empty lines
                continue
                
            if self._is_definite_body_text(text):
                return True
            
            # Stop if we hit another potential header
            if self._looks_like_header(next_line):
                break
        
        return False
    
    def _is_definite_body_text(self, text: str) -> bool:
        """Check if text is definitely body text (sentences)"""
        return (len(text) > 50 or 
                text.count('.') > 1 or 
                text.count(',') > 2 or
                len(text.split()) > 15 or
                bool(re.search(r'\b(the|and|that|this|with|from|they|have|will|been)\b', text.lower())))
    
    def _is_obviously_body_text(self, text: str) -> bool:
        """Quick check for obvious body text"""
        return (text.count('.') > 2 or 
                text.count(',') > 4 or 
                len(text.split()) > 20 or
                len(text) > 150)
    
    def _is_metadata_or_form(self, text: str) -> bool:
        """Check if text is metadata or form field"""
        text_lower = text.lower()
        patterns = [
            r'page\s+\d+', r'\d+\s+of\s+\d+', r'version\s+\d+\.\d+',
            r'www\.', r'@', r'copyright', r'©', r'confidential',
            r'^\d+\.\s*$', r'^Name\s+of\s+', r'^Date\s+of\s+'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in patterns)
    
    def _looks_like_header(self, line: Dict) -> bool:
        """Quick check if line looks like a header"""
        text = line['text'].strip()
        return (len(text) < 100 and 
                not text.endswith('.') and
                line.get('avg_font_size', 12) >= 11 and
                not self._is_obviously_body_text(text))
    
    def _check_header_text_characteristics(self, text: str) -> bool:
        """Check if text has characteristics typical of headers"""
        
        # Length check
        if not (3 <= len(text) <= 120):
            return False
        
        # Should not end with sentence punctuation (except colons)
        if text.endswith(('.', '!', '?')) and not text.endswith(':'):
            return False
        
        # Should not have too many sentence indicators
        if text.count('.') > 1 or text.count(',') > 3:
            return False
        
        # Should start with capital letter or number
        if not (text[0].isupper() or text[0].isdigit()):
            return False
        
        return True
    
    def _calculate_spatial_confidence(self, line: Dict, all_lines: List[Dict], index: int) -> float:
        """Calculate confidence score for spatial header detection"""
        
        confidence = 0.0
        
        # Whitespace ratio (more above than below)
        whitespace_above = self._measure_whitespace_above(line, all_lines, index)
        whitespace_below = self._measure_whitespace_below(line, all_lines, index)
        
        if whitespace_above > whitespace_below * 1.5:
            confidence += 0.3
        
        # Font size relative to following text
        font_boost = self._compare_font_with_following_text(line, all_lines, index)
        confidence += font_boost * 0.2
        
        # Text length (headers are typically concise)
        text_len = len(line['text'].strip())
        if 5 <= text_len <= 80:
            confidence += 0.2
        elif text_len <= 120:
            confidence += 0.1
        
        # Bold text bonus
        if line.get('is_bold', False):
            confidence += 0.15
        
        # Position bonus (left-aligned or centered)
        if line.get('position') in ['left', 'center']:
            confidence += 0.1
        
        # Following body text confirmation
        if self._has_following_body_text(line, all_lines, index):
            confidence += 0.25
        
        return min(confidence, 1.0)
    
    def _compare_font_with_following_text(self, line: Dict, all_lines: List[Dict], index: int) -> float:
        """Compare font size with following text"""
        current_font = line.get('avg_font_size', 12)
        
        # Get average font size of next few lines
        following_fonts = []
        for i in range(index + 1, min(index + 4, len(all_lines))):
            next_line = all_lines[i]
            if next_line['text'].strip():
                following_fonts.append(next_line.get('avg_font_size', 12))
        
        if not following_fonts:
            return 0.0
        
        avg_following_font = sum(following_fonts) / len(following_fonts)
        
        # Return ratio (capped at 1.0)
        return min(current_font / avg_following_font - 1.0, 1.0)
    
    def _infer_level_from_context(self, line: Dict, all_lines: List[Dict], index: int) -> int:
        """Infer heading level based on spatial and contextual clues"""
        
        # Start with font-based level
        font_size = line.get('avg_font_size', 12)
        is_bold = line.get('is_bold', False)
        
        # Simple font-based level
        if font_size >= 16 or is_bold and font_size >= 14:
            font_level = 1
        elif font_size >= 14 or is_bold and font_size >= 12:
            font_level = 2
        elif font_size >= 12 or is_bold:
            font_level = 3
        else:
            font_level = 4
        
        # Adjust based on whitespace magnitude
        whitespace_magnitude = self._measure_whitespace_above(line, all_lines, index)
        
        if whitespace_magnitude > self.LARGE_WHITESPACE_THRESHOLD:
            font_level = max(1, font_level - 1)
        
        return min(font_level, 6)
    
    def _detect_sequential_header_groups(self, headings: List[Dict]) -> List[Dict]:
        """Detect groups of sequential headers and assign hierarchy"""
        
        if not headings:
            return headings
        
        # Sort headings by page and position
        headings.sort(key=lambda h: (h.get('page', 0), h.get('y_position', 0)))
        
        header_groups = []
        current_group = []
        
        for i, heading in enumerate(headings):
            if i == 0:
                current_group = [heading]
                continue
            
            prev_heading = headings[i-1]
            
            # Check if headers are sequential
            if self._are_headers_sequential(prev_heading, heading):
                current_group.append(heading)
            else:
                # End current group, start new one
                if len(current_group) > 1:
                    header_groups.append(current_group)
                current_group = [heading]
        
        # Process last group
        if len(current_group) > 1:
            header_groups.append(current_group)
        
        # Assign sequential hierarchy to groups
        for group in header_groups:
            for i, heading in enumerate(group[:4]):  # Max H4
                heading['level'] = f"H{i+1}"
                heading['sequential_hierarchy'] = True
        
        return headings
    
    def _are_headers_sequential(self, header1: Dict, header2: Dict) -> bool:
        """Check if two headers are sequential (close together, minimal text between)"""
        
        # Must be on same page
        if header1.get('page', 0) != header2.get('page', 0):
            return False
        
        # Check vertical distance
        y1 = header1.get('y_position', 0)
        y2 = header2.get('y_position', 0)
        
        if abs(y2 - y1) > self.SEQUENTIAL_THRESHOLD:
            return False
        
        return True
    
    def _merge_heading_candidates(self, traditional_headings: List[Dict], spatial_headings: List[Dict]) -> List[Dict]:
        """Merge traditional and spatial heading candidates, avoiding duplicates"""
        
        merged = list(traditional_headings)
        
        for spatial_heading in spatial_headings:
            # Check for duplicates (same text, similar position)
            is_duplicate = False
            for existing in merged:
                if (spatial_heading['text'].lower().strip() == existing['text'].lower().strip() and
                    spatial_heading.get('page', 0) == existing.get('page', 0)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(spatial_heading)
        
        return merged
    
    def _resolve_heading_conflicts(self, headings: List[Dict]) -> List[Dict]:
        """Resolve conflicts between different detection methods"""
        
        # For now, just remove exact duplicates and sort by confidence
        unique_headings = []
        seen_texts = set()
        
        # Sort by confidence (higher first)
        headings.sort(key=lambda h: h.get('confidence', 0.5), reverse=True)
        
        for heading in headings:
            text_key = (heading['text'].lower().strip(), heading.get('page', 0))
            if text_key not in seen_texts:
                unique_headings.append(heading)
                seen_texts.add(text_key)
        
        return unique_headings