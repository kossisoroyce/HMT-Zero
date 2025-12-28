"""
Persistence layer for nurture states.
Supports JSON file storage for Phase 1 prototype.
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

from .state import NurtureState


class NurtureStore:
    """
    Storage backend for NurtureState instances.
    Uses JSON files for Phase 1 prototype.
    """
    
    def __init__(self, storage_dir: str = "./nurture_data"):
        """
        Initialize the storage backend.
        
        Args:
            storage_dir: Directory to store nurture state files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.storage_dir / "states"
        self.states_dir.mkdir(exist_ok=True)
        self.history_dir = self.storage_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
    
    def _get_state_path(self, instance_id: str) -> Path:
        """Get the file path for a nurture state."""
        return self.states_dir / f"{instance_id}.json"
    
    def _get_history_path(self, instance_id: str) -> Path:
        """Get the file path for interaction history."""
        return self.history_dir / f"{instance_id}_history.json"
    
    def save(self, state: NurtureState) -> bool:
        """
        Save a nurture state to storage.
        
        Args:
            state: NurtureState to save
        
        Returns:
            True if successful
        """
        try:
            path = self._get_state_path(state.instance_id)
            data = state.to_dict()
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving state {state.instance_id}: {e}")
            return False
    
    def load(self, instance_id: str) -> Optional[NurtureState]:
        """
        Load a nurture state from storage.
        
        Args:
            instance_id: ID of the instance to load
        
        Returns:
            NurtureState if found, None otherwise
        """
        try:
            path = self._get_state_path(instance_id)
            
            if not path.exists():
                return None
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            return NurtureState.from_dict(data)
        except Exception as e:
            print(f"Error loading state {instance_id}: {e}")
            return None
    
    def delete(self, instance_id: str) -> bool:
        """
        Delete a nurture state from storage.
        
        Args:
            instance_id: ID of the instance to delete
        
        Returns:
            True if successful
        """
        try:
            path = self._get_state_path(instance_id)
            if path.exists():
                path.unlink()
            
            history_path = self._get_history_path(instance_id)
            if history_path.exists():
                history_path.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting state {instance_id}: {e}")
            return False
    
    def list_instances(self) -> List[str]:
        """
        List all stored instance IDs.
        
        Returns:
            List of instance IDs
        """
        instances = []
        for path in self.states_dir.glob("*.json"):
            instances.append(path.stem)
        return instances
    
    def exists(self, instance_id: str) -> bool:
        """Check if an instance exists in storage."""
        return self._get_state_path(instance_id).exists()
    
    def save_interaction(
        self,
        instance_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Dict
    ) -> bool:
        """
        Save an interaction to history.
        
        Args:
            instance_id: Instance ID
            user_input: User's input
            assistant_response: Assistant's response
            metadata: Interaction metadata
        
        Returns:
            True if successful
        """
        try:
            path = self._get_history_path(instance_id)
            
            # Load existing history
            if path.exists():
                with open(path, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new interaction
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'assistant_response': assistant_response,
                'metadata': metadata
            }
            history.append(interaction)
            
            # Save updated history
            with open(path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving interaction for {instance_id}: {e}")
            return False
    
    def load_history(self, instance_id: str, limit: int = 100) -> List[Dict]:
        """
        Load interaction history for an instance.
        
        Args:
            instance_id: Instance ID
            limit: Maximum number of interactions to return
        
        Returns:
            List of interaction records
        """
        try:
            path = self._get_history_path(instance_id)
            
            if not path.exists():
                return []
            
            with open(path, 'r') as f:
                history = json.load(f)
            
            return history[-limit:]
        except Exception as e:
            print(f"Error loading history for {instance_id}: {e}")
            return []
    
    def get_conversation_history(
        self,
        instance_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in format suitable for context assembly.
        
        Args:
            instance_id: Instance ID
            limit: Maximum number of exchanges
        
        Returns:
            List of {"role": "user"|"assistant", "content": str}
        """
        history = self.load_history(instance_id, limit)
        
        conversation = []
        for item in history:
            conversation.append({
                'role': 'user',
                'content': item['user_input']
            })
            conversation.append({
                'role': 'assistant',
                'content': item['assistant_response']
            })
        
        return conversation[-limit * 2:]  # Last N exchanges


# Default store instance
default_store = NurtureStore()
