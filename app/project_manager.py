"""
Project Manager for PyPotteryScan
Handles creation, loading, and management of project workspaces
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class ProjectManager:
    """Manages project workspaces with hierarchical folder structure"""
    
    def __init__(self, projects_root: str = "projects"):
        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(exist_ok=True)
    
    def create_project(self, project_name: str, description: str = "") -> Dict:
        """
        Create a new project with folder structure and metadata
        
        Args:
            project_name: Name of the project
            description: Optional project description
            
        Returns:
            Dict with project metadata
        """
        # Sanitize project name for filesystem
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        # Create unique ID based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{safe_name}_{timestamp}"
        
        project_path = self.projects_root / project_id
        
        # Check if project already exists
        if project_path.exists():
            raise ValueError(f"Project already exists: {project_id}")
        
        # Create folder structure
        folders = [
            'original_images',     # Original scanned images
            'thumbnails',          # Cached thumbnails for performance
            'annotations',         # Annotation data (JSON files)
            'cropped_drawings',    # Cropped individual drawings
            'cleaned_drawings',    # Cleaned drawings (no text)
            'ocr_results',         # OCR text extraction results
            'exports',             # Final exports (CSV, Excel)
            'fewshot_examples',    # Few-shot parser examples
        ]
        
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # Create project metadata
        metadata = {
            'project_id': project_id,
            'project_name': project_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'workflow_status': {
                'images_loaded': False,
                'images_count': 0,
                'annotations_completed': 0,
                'ocr_processed': 0,
                'cleaned_count': 0,
                'current_image_index': 0,
                'processed_images': []  # List of processed image filenames
            },
            'settings': {
                'confidence_threshold': 0.5,
            }
        }
        
        # Save metadata
        self._save_metadata(project_path, metadata)
        
        return metadata
    
    def list_projects(self) -> List[Dict]:
        """
        List all available projects
        
        Returns:
            List of project metadata dictionaries
        """
        projects = []
        
        if not self.projects_root.exists():
            return projects
        
        for project_dir in self.projects_root.iterdir():
            if project_dir.is_dir():
                metadata_file = project_dir / 'project.json'
                if metadata_file.exists():
                    try:
                        metadata = self._load_metadata(project_dir)
                        projects.append(metadata)
                    except Exception as e:
                        print(f"Error loading project {project_dir.name}: {e}")
        
        # Sort by last modified (most recent first)
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        
        return projects
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific project
        
        Args:
            project_id: ID of the project
            
        Returns:
            Project metadata or None if not found
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return None
        
        try:
            return self._load_metadata(project_path)
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its contents
        
        Args:
            project_id: ID of the project to delete
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            shutil.rmtree(project_path)
            return True
        except Exception as e:
            print(f"Error deleting project {project_id}: {e}")
            return False
    
    def update_workflow_status(self, project_id: str, status_updates: Dict) -> bool:
        """
        Update workflow status for a project
        
        Args:
            project_id: ID of the project
            status_updates: Dictionary of status fields to update
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            metadata['workflow_status'].update(status_updates)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(project_path, metadata)
            return True
        except Exception as e:
            print(f"Error updating workflow status for {project_id}: {e}")
            return False
    
    def update_settings(self, project_id: str, settings: Dict) -> bool:
        """
        Update project settings
        
        Args:
            project_id: ID of the project
            settings: Dictionary of settings to update
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            metadata['settings'].update(settings)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(project_path, metadata)
            return True
        except Exception as e:
            print(f"Error updating settings for {project_id}: {e}")
            return False
    
    def get_project_path(self, project_id: str, subfolder: str = None) -> Optional[Path]:
        """
        Get the filesystem path for a project or its subfolder
        
        Args:
            project_id: ID of the project
            subfolder: Optional subfolder name (original_images, annotations, etc.)
            
        Returns:
            Path object or None if project doesn't exist
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return None
        
        if subfolder:
            return project_path / subfolder
        
        return project_path
    
    def _load_metadata(self, project_path: Path) -> Dict:
        """Load project metadata from project.json"""
        metadata_file = project_path / 'project.json'
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_metadata(self, project_path: Path, metadata: Dict):
        """Save project metadata to project.json"""
        metadata_file = project_path / 'project.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_images_list(self, project_id: str, folder_type: str = 'original_images') -> List[str]:
        """
        Get list of images in a project folder
        
        Args:
            project_id: ID of the project
            folder_type: Type of folder (original_images, cropped_drawings, etc.)
            
        Returns:
            List of image filenames
        """
        folder_path = self.get_project_path(project_id, folder_type)
        
        if not folder_path or not folder_path.exists():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = []
        
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path.name)
        
        return sorted(images)
    
    def count_files(self, project_id: str, folder_type: str = 'original_images') -> int:
        """
        Count files in a project folder
        
        Args:
            project_id: ID of the project
            folder_type: Type of folder
            
        Returns:
            Number of files
        """
        return len(self.get_images_list(project_id, folder_type))
    
    def save_annotation_data(self, project_id: str, image_name: str, annotation_data: Dict) -> bool:
        """
        Save annotation data for an image
        
        Args:
            project_id: ID of the project
            image_name: Name of the image file
            annotation_data: Annotation data to save
            
        Returns:
            True if successful, False otherwise
        """
        annotations_path = self.get_project_path(project_id, 'annotations')
        
        if not annotations_path:
            return False
        
        try:
            # Create annotation filename (same as image but .json)
            base_name = Path(image_name).stem
            annotation_file = annotations_path / f"{base_name}_annotations.json"
            
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving annotation data for {image_name}: {e}")
            return False
    
    def load_annotation_data(self, project_id: str, image_name: str) -> Optional[Dict]:
        """
        Load annotation data for an image
        
        Args:
            project_id: ID of the project
            image_name: Name of the image file
            
        Returns:
            Annotation data or None if not found
        """
        annotations_path = self.get_project_path(project_id, 'annotations')
        
        if not annotations_path:
            return None
        
        try:
            base_name = Path(image_name).stem
            annotation_file = annotations_path / f"{base_name}_annotations.json"
            
            if not annotation_file.exists():
                return None
            
            with open(annotation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading annotation data for {image_name}: {e}")
            return None
    
    def save_ocr_results(self, project_id: str, results: List[Dict]) -> bool:
        """
        Save OCR results to project (OVERWRITES previous results)
        
        Args:
            project_id: ID of the project
            results: List of OCR result dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        ocr_path = self.get_project_path(project_id, 'ocr_results')
        
        if not ocr_path:
            return False
        
        try:
            # Always use the same filename to overwrite previous results
            results_file = ocr_path / "ocr_results.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving OCR results: {e}")
            return False
    
    def get_latest_ocr_results(self, project_id: str) -> Optional[List[Dict]]:
        """
        Get the OCR results from project
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of OCR results or None if not found
        """
        ocr_path = self.get_project_path(project_id, 'ocr_results')
        
        if not ocr_path or not ocr_path.exists():
            return None
        
        try:
            # Use fixed filename
            results_file = ocr_path / 'ocr_results.json'
            
            if not results_file.exists():
                # Fallback: try to find any old timestamped files
                json_files = sorted(ocr_path.glob('ocr_results_*.json'), reverse=True)
                if not json_files:
                    return None
                results_file = json_files[0]
            
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading OCR results: {e}")
            return None
    
    def save_ocr_corrections(self, project_id: str, corrections: Dict) -> bool:
        """
        Save OCR corrections to project (OVERWRITES previous corrections)
        
        Args:
            project_id: ID of the project
            corrections: Dictionary of corrections (key -> corrected_text)
            
        Returns:
            True if successful, False otherwise
        """
        ocr_path = self.get_project_path(project_id, 'ocr_results')
        
        if not ocr_path:
            return False
        
        try:
            # Use fixed filename for corrections
            corrections_file = ocr_path / "ocr_corrections.json"
            
            with open(corrections_file, 'w', encoding='utf-8') as f:
                json.dump(corrections, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving OCR corrections: {e}")
            return False
    
    def get_ocr_corrections(self, project_id: str) -> Optional[Dict]:
        """
        Get OCR corrections from project
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary of corrections or None if not found
        """
        ocr_path = self.get_project_path(project_id, 'ocr_results')
        
        if not ocr_path or not ocr_path.exists():
            return None
        
        try:
            corrections_file = ocr_path / 'ocr_corrections.json'
            
            if not corrections_file.exists():
                return None
            
            with open(corrections_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading OCR corrections: {e}")
            return None
    
    def save_fewshot_examples(self, project_id: str, examples: List[Dict]) -> bool:
        """
        Save few-shot examples to project
        
        Args:
            project_id: ID of the project
            examples: List of few-shot example dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        fewshot_path = self.get_project_path(project_id, 'fewshot_examples')
        
        if not fewshot_path:
            return False
        
        try:
            # Ensure the fewshot_examples folder exists (for old projects)
            fewshot_path.mkdir(parents=True, exist_ok=True)
            
            examples_file = fewshot_path / "fewshot_examples.json"
            
            with open(examples_file, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving few-shot examples: {e}")
            return False
    
    def get_fewshot_examples(self, project_id: str) -> Optional[List[Dict]]:
        """
        Get few-shot examples from project
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of few-shot examples or None if not found
        """
        fewshot_path = self.get_project_path(project_id, 'fewshot_examples')
        
        if not fewshot_path or not fewshot_path.exists():
            return None
        
        try:
            examples_file = fewshot_path / 'fewshot_examples.json'
            
            if not examples_file.exists():
                return None
            
            with open(examples_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading few-shot examples: {e}")
            return None
