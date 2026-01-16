"""
Google Docs Content Fetcher
Fetches content from public Google Docs without authentication
"""

import requests
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict


class GoogleDocsFetcher:
    """Fetch and parse Google Docs content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_doc_id(self, url: str) -> Optional[str]:
        """Extract document ID from Google Docs URL"""
        patterns = [
            r'/document/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def fetch_document(self, url: str) -> Dict[str, any]:
        """
        Fetch Google Doc content
        Returns: {
            'success': bool,
            'content': str,
            'title': str,
            'error': str (if failed)
        }
        """
        try:
            # Extract document ID
            doc_id = self.extract_doc_id(url)
            if not doc_id:
                return {
                    'success': False,
                    'error': 'Invalid Google Docs URL. Please provide a valid link.'
                }
            
            # Construct export URL
            export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=html'
            
            # Fetch document
            response = self.session.get(export_url, timeout=15)
            
            if response.status_code == 403:
                return {
                    'success': False,
                    'error': 'Document is private. Please make it "Anyone with the link can view".'
                }
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f'Failed to fetch document. Status code: {response.status_code}'
                }
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.text if title else 'Untitled Document'
            
            # Extract text content
            body = soup.find('body')
            if not body:
                return {
                    'success': False,
                    'error': 'Could not parse document content.'
                }
            
            # Clean and extract text
            text = self._extract_clean_text(body)
            
            if not text or len(text.strip()) < 50:
                return {
                    'success': False,
                    'error': 'Document appears to be empty or too short.'
                }
            
            return {
                'success': True,
                'content': text,
                'title': title,
                'doc_id': doc_id
            }
        
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. Please try again.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching document: {str(e)}'
            }
    
    def _extract_clean_text(self, body_element) -> str:
        """Extract and clean text from HTML body"""
        # Remove script and style elements
        for element in body_element(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Get text
        text = body_element.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        # Join with proper spacing
        text = '\n\n'.join(lines)
        
        return text
    
    def validate_url(self, url: str) -> Dict[str, any]:
        """Quick validation without fetching full content"""
        if not url:
            return {'valid': False, 'error': 'URL is required'}
        
        if 'docs.google.com' not in url:
            return {'valid': False, 'error': 'Not a Google Docs URL'}
        
        doc_id = self.extract_doc_id(url)
        if not doc_id:
            return {'valid': False, 'error': 'Invalid Google Docs URL format'}
        
        return {'valid': True, 'doc_id': doc_id}
