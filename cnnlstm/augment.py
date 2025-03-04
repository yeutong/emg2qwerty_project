import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from emg2qwerty.charset import charset

@dataclass
class KeystrokeTransitionAugmentation:
    """Augments keystroke data by occasionally substituting keys with physically
    adjacent keys on a QWERTY keyboard.
    
    Args:
        substitution_prob (float): Probability of substituting a character (default: 0.2)
        max_substitutions (int): Maximum number of substitutions per sequence (default: 3)
        only_adjacent (bool): Whether to only substitute with adjacent keys (default: True)
    """
    substitution_prob: float = 0.2
    max_substitutions: int = 3
    only_adjacent: bool = True
    
    def __post_init__(self):
        # Create QWERTY keyboard layout
        self.keyboard_layout = [
            '`1234567890-=',
            'qwertyuiop[]\\',
            "asdfghjkl;'",
            'zxcvbnm,./'
        ]
        
        # Create adjacency mapping
        self.adjacency_map = self._create_adjacency_map()
        
        # Get character set
        self.char_set = charset()
        
        # Create index-to-char and char-to-index mappings
        self.idx_to_char = {i: c for i, c in enumerate(self.char_set.allowed_chars)}
        self.char_to_idx = {c: i for i, c in enumerate(self.char_set.allowed_chars)}
    
    def _create_adjacency_map(self) -> Dict[str, List[str]]:
        """Create a map of each character to its adjacent characters on the keyboard."""
        adjacency = {}
        
        # Helper to get adjacent positions
        def get_adjacent_positions(row_idx, col_idx):
            adjacent_positions = []
            for r in range(max(0, row_idx-1), min(len(self.keyboard_layout), row_idx+2)):
                for c in range(max(0, col_idx-1), min(len(self.keyboard_layout[r]), col_idx+2)):
                    if r == row_idx and c == col_idx:
                        continue  # Skip the character itself
                    adjacent_positions.append((r, c))
            return adjacent_positions
        
        # Build adjacency map
        for row_idx, row in enumerate(self.keyboard_layout):
            for col_idx, char in enumerate(row):
                adjacent_positions = get_adjacent_positions(row_idx, col_idx)
                adjacency[char] = []
                for r, c in adjacent_positions:
                    adjacent_char = self.keyboard_layout[r][c]
                    adjacency[char].append(adjacent_char)
        
        return adjacency
    
    def _get_substitution(self, char: str) -> str:
        """Get a substitution for the given character."""
        if char not in self.adjacency_map or not self.adjacency_map[char]:
            return char  # No substitution available
        
        # Get possible substitutions
        substitutions = self.adjacency_map[char]
        if not self.only_adjacent:
            # Add some random keys that are not adjacent (for more variety)
            all_chars = [c for row in self.keyboard_layout for c in row]
            non_adjacent = [c for c in all_chars if c != char and c not in substitutions]
            substitutions.extend(np.random.choice(non_adjacent, size=min(3, len(non_adjacent)), replace=False))
        
        # Choose a random substitution
        return np.random.choice(substitutions)
    
    def __call__(self, inputs_and_targets: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply keystroke transition augmentation to the data.
        
        Args:
            inputs_and_targets: Tuple of (inputs, targets) tensors from WindowedEMGDataset
            
        Returns:
            Tuple of (inputs, augmented_targets)
        """
        inputs, targets = inputs_and_targets
        
        # Clone targets to avoid modifying the original
        augmented_targets = targets.clone()
        seq_len = len(targets)
        
        if seq_len == 0:
            return inputs, augmented_targets
            
        # Determine number of substitutions for this sequence
        num_substitutions = min(
            np.random.binomial(seq_len, self.substitution_prob),
            self.max_substitutions
        )
        
        if num_substitutions == 0:
            return inputs, augmented_targets
            
        # Select positions to substitute
        positions = np.random.choice(seq_len, size=num_substitutions, replace=False)
        
        # Apply substitutions
        for pos in positions:
            # Get original character index
            char_idx = targets[pos].item()
            
            # Skip special tokens
            if char_idx >= len(self.char_set.allowed_chars) or char_idx == self.char_set.null_class:
                continue
                
            # Convert to character, find substitution, convert back to index
            original_char = self.idx_to_char[char_idx]
            substitute_char = self._get_substitution(original_char)
            
            # Only substitute if the character is in our charset
            if substitute_char in self.char_to_idx:
                substitute_idx = self.char_to_idx[substitute_char]
                augmented_targets[pos] = substitute_idx
        
        return inputs, augmented_targets
